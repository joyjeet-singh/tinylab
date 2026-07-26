"""
authors_adapter.py -- present the authors' released checkpoint through the
exact interface our CEMPlanner and evaluator already use.

Everything about HOW to drive their model comes from authors_driving_spec.json,
which was measured rather than assumed:

    pixels  : ImageNet-normalised (raw [0,1] scored FIVE TIMES WORSE than
              predicting no change at all -- this is not in their config.json)
    actions : the two real dimensions placed in the recorded slots of a
              ten-wide vector, the rest zero
    history : 3 frames (read from predictor.pos_embedding)

Our evaluator hands the model pixels in [0,1] and actions of width 2, exactly
as it does for our own checkpoint. The adapter converts on the way in, so the
evaluator, the planner, the domain guard and the protocol are all untouched.
The ONLY thing that changes between our calibration run and our own result is
the weights -- which is the whole point of the experiment.

The three methods below are the three calls our planner makes:
    encode({"pixels": ...})["emb"]      action_encoder(actions)
    predict(emb, act_emb)
plus the history_size attribute.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

VIT_SIZES = {"tiny": (192, 12, 3), "small": (384, 12, 6),
             "base": (768, 12, 12), "large": (1024, 24, 16),
             "huge": (1280, 32, 16)}
TARGET_MAP = {
    "stable_worldmodel.wm.lewm.LeWM": "jepa.JEPA",
    "stable_worldmodel.wm.lewm.module.Predictor": "module.ARPredictor",
    "stable_worldmodel.wm.lewm.module.Embedder": "module.Embedder",
    "stable_worldmodel.wm.lewm.module.MLP": "module.MLP",
}


def _remap(o, m):
    if isinstance(o, dict):
        return {k: (m.get(v, v) if k == "_target_" and isinstance(v, str)
                    else _remap(v, m)) for k, v in o.items()}
    return o


class AuthorsAdapter(nn.Module):
    """Their model, wearing our interface."""

    def __init__(self, inner, spec):
        super().__init__()
        self.inner = inner
        self.spec = spec
        self.history_size = int(spec["history_size"])
        self.action_width = int(spec["action_width"])
        self.slots = list(spec["action_slots"])
        # The planner emits a PER-STEP action which the evaluator then holds
        # for `frameskip` environment steps. If the model was trained on the
        # block SUM, the equivalent input is frameskip x that action. This
        # factor is measured (action_scale_check.py), never assumed.
        self.action_scale = float(spec.get("action_scale", 1.0))
        # "slots"  : put the action in the recorded slots, zero elsewhere
        # "repeat" : tile the action frameskip times to fill the whole vector
        #            -- the reference keeps actions DENSE and reshapes them to
        #            (history_len, frameskip * action_dim), so a constant
        #            action held for frameskip steps IS the repeated vector
        self.action_encoding = spec.get("action_encoding", "slots")
        self.frameskip = int(spec.get("frameskip", 5))
        self.register_buffer(
            "_mean", torch.tensor(spec["imagenet_mean"]).view(1, 1, 3, 1, 1))
        self.register_buffer(
            "_std", torch.tensor(spec["imagenet_std"]).view(1, 1, 3, 1, 1))

    def encode(self, info):
        px = info["pixels"]
        if self.spec.get("pixel_convention") == "imagenet":
            px = (px - self._mean) / self._std
        return self.inner.encode({**info, "pixels": px})

    def predict(self, emb, act_emb):
        return self.inner.predict(emb, act_emb)

    def action_encoder(self, act):
        """(B, T, 2) from our planner -> (B, T, action_width) for theirs."""
        b, t, d = act.shape
        if self.action_encoding == "repeat":
            reps = self.action_width // d
            wide = act.repeat(1, 1, reps)
            if wide.shape[-1] < self.action_width:
                pad = act.new_zeros(b, t, self.action_width - wide.shape[-1])
                wide = torch.cat([wide, pad], dim=-1)
        else:
            wide = act.new_zeros(b, t, self.action_width)
            wide[..., self.slots] = act * self.action_scale
        return self.inner.action_encoder(wide)


def load_authors_model(spec_path="authors_driving_spec.json",
                       lewm=None, dest=None, verbose=True):
    spec = json.loads(Path(spec_path).read_text())

    try:
        import stable_worldmodel.wm.lewm  # noqa: F401
        mapping = {}
    except Exception:
        lewm = Path(lewm or (Path.home() / "le-wm")).expanduser()
        if not (lewm / "jepa.py").exists():
            raise SystemExit(
                f"need the le-wm clone at {lewm} (this install lacks "
                f"stable_worldmodel.wm.lewm)")
        sys.path.insert(0, str(lewm))
        mapping = TARGET_MAP

    if dest is None:
        candidates = []
        try:
            from stable_worldmodel.data.utils import get_cache_dir
            try:
                candidates.append(Path(get_cache_dir()) / "hf_tworooms")
            except TypeError:
                candidates.append(Path(get_cache_dir(None)) / "hf_tworooms")
        except Exception:
            pass
        candidates += [Path.home() / ".stable_worldmodel" / "hf_tworooms",
                       Path.home() / ".stable-wm" / "hf_tworooms",
                       Path(spec_path).resolve().parent / "hf_tworooms"]
        dest = next((d for d in candidates
                     if (d / "config.json").exists()
                     and (d / "weights.pt").exists()), None)
        if dest is None:
            raise SystemExit(
                "could not find the authors' config.json + weights.pt. "
                "Looked in:\n  " + "\n  ".join(str(c) for c in candidates) +
                "\nPass the right folder with dest=... or re-run "
                "authors_ckpt_fetch.py to download them.")
    dest = Path(dest).expanduser()

    from hydra.utils import instantiate, get_class
    from transformers import ViTConfig, ViTModel

    cfg = json.loads((dest / "config.json").read_text())
    c = _remap(cfg, mapping)
    e = cfg["encoder"]
    h, layers, heads = VIT_SIZES[e["size"]]
    encoder = ViTModel(ViTConfig(hidden_size=h, num_hidden_layers=layers,
                                 num_attention_heads=heads,
                                 intermediate_size=h * 4,
                                 image_size=e["image_size"],
                                 patch_size=e["patch_size"]),
                       add_pooling_layer=False, use_mask_token=False)
    parts = {k: instantiate(c[k]) for k in
             ("predictor", "action_encoder", "projector", "pred_proj")
             if k in c}
    inner = get_class(c["_target_"])(encoder=encoder, **parts)

    sd = torch.load(dest / "weights.pt", map_location="cpu",
                    weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    inner.load_state_dict(sd, strict=True)
    inner.eval()
    inner.requires_grad_(False)

    gates = sum(float(v.abs().sum()) for k, v in inner.state_dict().items()
                if "adaLN_modulation" in k)
    if gates == 0:
        raise SystemExit("the conditioning path is all zeros -- these weights "
                         "are untrained and would ignore every action")

    model = AuthorsAdapter(inner, spec)
    model.eval()
    model.requires_grad_(False)
    if verbose:
        n = sum(p.numel() for p in inner.parameters()) / 1e6
        print(f"  authors' checkpoint: {n:.2f} M params, "
              f"history_size {model.history_size}, "
              f"actions -> {model.action_encoding} into {model.action_width}"
              f"{'' if model.action_encoding == 'repeat' else f' x{model.action_scale:g}'}, "
              f"pixels {spec.get('pixel_convention')}")
        print(f"  driving spec: {spec_path} ({spec.get('note', '')})")
    return model
