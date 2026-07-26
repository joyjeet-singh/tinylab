"""
dense_action_adapter.py -- drive a phase2 checkpoint from a planner that emits
two-wide actions.

THE MISMATCH
------------
A checkpoint trained with the corrected pipeline expects, per predictor step:

  * pixels    ImageNet-normalised, not [0, 1]
  * actions   TEN wide -- the five raw actions of the block, concatenated --
              and z-scored by the dataset's own statistics
  * context   history_size frames (3, not the 1 that ckpt_best used)

Our CEMPlanner emits a single two-wide action per step, and the evaluator hands
the model pixels in [0, 1]. Feeding that straight in is exactly the error that
produced the authors'-checkpoint 46% artifact: a model receiving actions it
cannot interpret still plans, it just plans badly, and the number looks like a
result.

WHAT THIS ADAPTER DOES
----------------------
The planner holds one action for `frameskip` environment steps, so the block
of five raw actions it produces is that action repeated five times. That is the
encoding this adapter uses -- and then applies the SAME z-score the loader
applied during training, computed from the same column with the same NaN rows
dropped.

Neither choice is assumed. `verify_phase2_driving.py` measures the repeat
encoding against controls and against the true dense actions from the file
before any planning number is taken seriously.

The three transforms are read from the run's manifest (`loader_convention`),
so a checkpoint trained without them is driven without them. The manifest is
the source of truth, which is the whole point of recording it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def action_stats(h5_path, drop_nan: bool = True):
    """Per-dimension mean/std of the raw action column.

    Must match tworoom_data._action_stats exactly, including dropping NaN rows
    (le-wm/utils.py:29 does the same). verify_phase2_driving.py cross-checks
    this against the loader's own computation rather than trusting the comment.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(str(Path(h5_path).expanduser()), "r") as f:
        a = np.asarray(f["action"][:], dtype=np.float64)
    if drop_nan:
        a = a[~np.isnan(a).any(axis=1)]
    mu = a.mean(0).astype(np.float32)
    sd = a.std(0).astype(np.float32)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def _action_width(inner) -> int:
    """Read the model's action width off its WEIGHTS, not an attribute.

    ToyJEPA does not keep action_dim as an attribute, so getattr silently
    returned the default 2 and the adapter tiled nothing. The first layer of
    the action encoder is the ground truth.
    """
    enc = getattr(inner, "action_encoder", None)
    for attr in ("patch_embed", "proj", "0"):
        layer = getattr(enc, attr, None) if enc is not None else None
        w = getattr(layer, "weight", None)
        if w is not None and w.dim() >= 2:
            return int(w.shape[1])
    for p in (enc.parameters() if enc is not None else []):
        if p.dim() >= 2:
            return int(p.shape[1])
    raise ValueError("could not determine the model's action width from its "
                     "weights -- inspect action_encoder by hand")


class DenseActionAdapter(nn.Module):
    """A dense-action checkpoint, wearing the planner's interface."""

    def __init__(self, inner, *, frameskip=5, imagenet_pixels=True,
                 zscore_actions=True, action_mu=None, action_sd=None):
        super().__init__()
        self.inner = inner
        self.frameskip = int(frameskip)
        self.imagenet_pixels = bool(imagenet_pixels)
        self.zscore_actions = bool(zscore_actions)
        self.history_size = int(getattr(inner, "history_size", 1))
        self.action_width = _action_width(inner)
        self.register_buffer(
            "_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer(
            "_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1))
        if zscore_actions:
            if action_mu is None or action_sd is None:
                raise ValueError("z-scoring requested but no action statistics "
                                 "given -- pass action_stats(h5)")
            self.register_buffer("_amu", torch.as_tensor(action_mu).float())
            self.register_buffer("_asd", torch.as_tensor(action_sd).float())

    def encode(self, info):
        px = info["pixels"]
        if self.imagenet_pixels:
            px = (px - self._mean) / self._std
        return self.inner.encode({**info, "pixels": px})

    def predict(self, emb, act_emb):
        return self.inner.predict(emb, act_emb)

    def action_encoder(self, act):
        """(B, T, 2) from the planner -> (B, T, action_width) for the model."""
        b, t, d = act.shape
        if self.action_width == d:
            wide = act
        else:
            reps = self.action_width // d
            if reps * d != self.action_width:
                raise ValueError(f"cannot tile {d}-wide actions into "
                                 f"{self.action_width}")
            # the planner holds one action for `frameskip` env steps, so the
            # block it produces IS that action repeated
            wide = act.repeat(1, 1, reps)
        if self.zscore_actions:
            reps = wide.shape[-1] // self._amu.shape[0]
            wide = (wide - self._amu.repeat(reps)) / self._asd.repeat(reps)
        return self.inner.action_encoder(wide)


def convention_from_manifest(run_dir) -> dict:
    """What the training run actually did, per its manifest."""
    mf = json.loads((Path(run_dir) / "manifest.json").read_text())
    conv = mf.get("loader_convention")
    if conv is None:
        # pre-provenance-patch runs: everything off, which is what they did
        return {"dense_actions": False, "imagenet_pixels": False,
                "zscore_actions": False, "_source": "absent (assumed all off)"}
    return {**conv, "_source": "manifest"}


def wrap_if_needed(model, run_dir, h5_path, frameskip=5, verbose=True):
    """Wrap `model` iff its manifest says it was trained with the new pipeline.

    A checkpoint trained without the transforms is returned untouched, so the
    same evaluator drives old and new checkpoints correctly without a flag
    anyone has to remember to pass.
    """
    conv = convention_from_manifest(run_dir)
    needs = conv.get("imagenet_pixels") or conv.get("zscore_actions") \
        or conv.get("dense_actions")
    if not needs:
        if verbose:
            print(f"  loader convention ({conv['_source']}): all off -- "
                  f"driving the model directly, no adapter")
        return model
    mu = sd = None
    if conv.get("zscore_actions"):
        mu, sd = action_stats(h5_path)
    wrapped = DenseActionAdapter(
        model, frameskip=frameskip,
        imagenet_pixels=bool(conv.get("imagenet_pixels")),
        zscore_actions=bool(conv.get("zscore_actions")),
        action_mu=mu, action_sd=sd)
    wrapped.eval()
    wrapped.requires_grad_(False)
    if verbose:
        print(f"  loader convention ({conv['_source']}): "
              f"dense={conv.get('dense_actions')} "
              f"imagenet={conv.get('imagenet_pixels')} "
              f"zscore={conv.get('zscore_actions')}")
        print(f"  ADAPTER ACTIVE: pixels "
              f"{'ImageNet' if wrapped.imagenet_pixels else 'raw'}, "
              f"actions 2 -> {wrapped.action_width} by repeat"
              f"{' + z-score' if wrapped.zscore_actions else ''}, "
              f"history_size {wrapped.history_size}")
    return wrapped
