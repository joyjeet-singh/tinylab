"""
toy_sigreg.py -- the anti-collapse term, and proof that it works.

The problem, restated
--------------------
Train on prediction alone and the model finds a cheat: output nearly the same
summary for every picture. Then "predict the next summary" is trivial -- the
answer is always the same. Loss looks wonderful. The model is useless. We
reproduced this on the toy world in 100 steps: loss fell 245x while the spread
of the summaries fell 10x. That is COLLAPSE.

The fix, in plain terms
-----------------------
"Make the summaries different" is too vague -- the model finds another cheat.
SIGReg demands something specific: viewed from EVERY direction, the cloud of
summaries must look like a standard bell curve.

Why a bell curve? It has a definite width. Identical summaries have zero width.
You cannot be a bell curve and be identical at once, so the cheat is blocked by
construction rather than by a penalty the model can negotiate around.

How it checks (the "sketch" part)
---------------------------------
Asking "is this 64-dimensional cloud bell-shaped?" directly is hard. So instead:
pick ~1024 random directions, squash the cloud flat onto each one, and check
each of those simple 1-D shadows for bell-shape. If it looks like a bell from a
thousand random angles, it is one.

The 1-D test is Epps-Pulley: compare the cloud's "characteristic function" to a
bell curve's, across a grid of frequencies, and integrate the mismatch. In plain
terms: a distribution has a fingerprint made of averaged cosines and sines at
different frequencies. A bell curve's fingerprint is known exactly. Measure the
gap. Zero gap = it is a bell curve.

Faithful to le-wm/module.py SIGReg. Read with train.py, which reveals two
details worth stating loudly:

  1. THE TARGET IS NOT DETACHED. Reference line: (pred_emb - tgt_emb).pow(2).mean()
     Gradient flows into BOTH sides. So the encoder is actively pulled toward
     making summaries easy to predict -- which makes collapse MORE tempting, not
     less. This is why SIGReg has to carry real weight.

  2. SIGReg IS COMPUTED PER TIMESTEP. Reference passes emb.transpose(0, 1),
     i.e. (T, B, D). The bell-curve test runs across the BATCH at each timestep
     separately, then averages. Not over all frames pooled together.

Run `python toy_sigreg.py` for the tests, including the one that matters:
train WITHOUT it (watch collapse), train WITH it (watch it not).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer. Faithful to the reference.

    Returns a number that is ~0 when the summaries look like a standard bell
    curve from every direction, and grows as they depart from it. Collapsed
    summaries (all identical) score very high.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        # The frequency grid for the Epps-Pulley test: 17 points from 0 to 3.
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)

        # Trapezoid rule weights for integrating over that grid: interior points
        # count double, the two endpoints count once.
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        # A standard bell curve's fingerprint at frequency t is exp(-t^2/2).
        # It doubles as the weighting window, so high frequencies (which are
        # noisy with finite samples) count for less.
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: (T, B, D) -- time first, as the reference passes it.
              The test runs across B (the batch) at each timestep, then averages.
        """
        # 1. random directions to squash the cloud onto, each of unit length
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))

        # 2. squash: (T,B,D) @ (D,P) -> (T,B,P), then spread over frequencies
        x_t = (proj @ A).unsqueeze(-1) * self.t          # (T, B, P, knots)

        # 3. the cloud's fingerprint, averaged over the batch (dim -3 = B),
        #    compared against the bell curve's known fingerprint
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()

        # 4. integrate the mismatch over frequencies, scale by batch size
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()      # average over directions and time


def lewm_loss(model, batch_out: dict, sigreg: SIGReg, ctx_len: int = 3,
              n_preds: int = 1, lambd: float = 0.09) -> dict:
    """
    The whole LeWM objective. Faithful to reference train.py lejepa_forward.

    Note what is NOT here: no stop-gradient, no EMA teacher, no frozen encoder,
    no six tunable coefficients. One prediction term, one bell-curve term, one
    knob (lambda). That simplicity IS the paper's contribution.

    model:      a ToyJEPA (we call model.predict)
    batch_out:  the dict returned by model.encode(batch)
    """
    emb = batch_out["emb"]                  # (B, T, D)
    act_emb = batch_out["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]              # the label -- NOT detached
    pred_emb = model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))       # (T, B, D) -- per timestep
    return {"pred_loss": pred_loss, "sigreg_loss": sigreg_loss,
            "loss": pred_loss + lambd * sigreg_loss}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def _test_statistic_behaviour():
    """Does the score actually recognise a bell curve, and reject collapse?"""
    torch.manual_seed(0)
    sig = SIGReg(num_proj=1024)
    T, B, D = 4, 128, 64

    cases = {
        "a real bell curve (what we want)": torch.randn(T, B, D),
        "collapsed: every summary identical": torch.zeros(T, B, D) + torch.randn(1, 1, D),
        "nearly collapsed (tiny spread)": torch.randn(T, B, D) * 0.01,
        "too spread out (5x too wide)": torch.randn(T, B, D) * 5.0,
        "flat/uniform, not bell-shaped": (torch.rand(T, B, D) - 0.5) * 3.46,
        "off-centre bell curve": torch.randn(T, B, D) + 2.0,
    }
    print("what the score says about different clouds of summaries")
    print("  (lower = more bell-curve-like; this is what training pushes down)")
    print()
    for name, x in cases.items():
        print(f"  {name:36s} -> {sig(x).item():10.3f}")

    good = sig(torch.randn(T, B, D)).item()
    bad = sig(torch.zeros(T, B, D) + torch.randn(1, 1, D)).item()
    assert bad > good * 10, "collapse is not penalised much more than a bell curve"
    print()
    print(f"  collapse scores {bad/good:.0f}x worse than a bell curve. PASS")


def _test_gradient_pushes_away_from_collapse():
    """
    Does following the gradient actually un-collapse a collapsed cloud?

    NOTE, and this is a real property worth knowing: SIGReg's gradient is
    WEAKEST exactly where collapse is worst. Measured on this toy:

        spread 0.01 -> gradient 2.0e-04     (collapsed: very weak pull)
        spread 0.50 -> gradient 4.0e-03     (20x stronger)
        spread 1.00 -> gradient 5.9e-04     (at the target: little pull needed)

    So SIGReg is a good PREVENTER but a poor RESCUER. It must be present from
    step one. Let the model collapse first and you land in a flat region that is
    slow to escape. This also means plain SGD struggles here -- steps are sized
    by the gradient, which is tiny. Adam (and AdamW, which the real training
    loop uses) adapts its own step size and escapes easily.
    """
    torch.manual_seed(0)
    sig = SIGReg(num_proj=1024)

    print()
    print("how strong is the pull, at different amounts of collapse?")
    print(f"  {'spread':>8s} {'score':>9s} {'gradient':>11s}")
    for s in [0.01, 0.1, 0.5, 1.0]:
        x = (torch.randn(2, 128, 64) * s).requires_grad_(True)
        loss = sig(x)
        loss.backward()
        print(f"  {s:8.2f} {loss.item():9.3f} {x.grad.abs().mean().item():11.2e}")
    print("  -> weakest pull exactly where collapse is worst. Prevent, don't rescue.")

    # Start from NEARLY collapsed -- tiny spread, but points not exactly equal.
    # This is what a real encoder produces. See the note below on why PERFECT
    # collapse is a different (and unreachable) case.
    x = (torch.randn(2, 128, 64) * 0.01).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)     # Adam, not SGD -- see the note above

    print()
    print("can it rescue a nearly-collapsed cloud? (optimising summaries directly)")
    for step in range(601):
        loss = sig(x)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0:
            print(f"  step {step:3d}  score {loss.item():9.3f}   "
                  f"spread {x.std().item():.4f}")
    assert x.std().item() > 0.5, "gradient did not restore spread"
    print(f"  PASS -- spread restored to {x.std().item():.4f}, "
          "essentially 1.0 (a standard bell curve)")

    # -- the symmetry finding, worth knowing -------------------------------
    print()
    print("one more property, found by accident and worth keeping:")
    perfect = (torch.zeros(2, 128, 64) + torch.randn(1, 1, 64) * 0.01).requires_grad_(True)
    sig(perfect).backward()
    var_perfect = perfect.grad.reshape(-1, 64).std(0).mean().item()
    near = (torch.randn(2, 128, 64) * 0.01).requires_grad_(True)
    sig(near).backward()
    var_near = near.grad.reshape(-1, 64).std(0).mean().item()
    print(f"  PERFECT collapse (all points identical): gradient varies "
          f"across points by {var_perfect:.1e}")
    print(f"  NEAR collapse (tiny but real spread)   : gradient varies "
          f"across points by {var_near:.1e}")
    print("  -> At perfect collapse the gradient is IDENTICAL for every point, so")
    print("     it can only shift the whole cloud, never pull it apart. Perfect")
    print("     collapse is a true fixed point that SIGReg cannot escape. It is")
    print("     also unreachable in practice: a real encoder always gives slightly")
    print("     different outputs for different pictures, and that is enough.")


def _test_the_real_thing():
    """The test that matters: train on the toy world, with and without SIGReg."""
    import torch.nn.functional as F
    from tworoom_data import TwoRoomIndex, TwoRoomClips, ClipSpec
    from toy_model import ToyJEPA

    spec = ClipSpec(history=3, frameskip=5)
    idx = TwoRoomIndex("toy_tworoom.h5", spec)
    ds = TwoRoomClips("toy_tworoom.h5", idx)

    def get_batch(n, rng):
        picks = rng.choice(len(ds), size=n, replace=False)
        items = [ds[int(p)] for p in picks]
        return {"pixels": torch.tensor(np.stack([i["pixels"] for i in items])),
                "action": torch.tensor(np.stack([i["action"] for i in items]))}

    def run(lambd, steps=300, seed=0):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = ToyJEPA(embed_dim=64, history_size=3)
        sig = SIGReg(num_proj=512)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        hist = []
        for step in range(steps):
            b = get_batch(64, rng)
            out = model.encode(b)
            emb, act_emb = out["emb"], out["act_emb"]
            ctx_emb, ctx_act = emb[:, :3], act_emb[:, :3]
            tgt_emb = emb[:, 1:]                     # NOT detached -- per reference
            pred_emb = model.predict(ctx_emb, ctx_act)

            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            sig_loss = sig(emb.transpose(0, 1))
            loss = pred_loss + lambd * sig_loss

            opt.zero_grad(); loss.backward(); opt.step()
            if step % 75 == 0 or step == steps - 1:
                spread = emb.reshape(-1, 64).std(0).mean().item()
                hist.append((step, pred_loss.item(), sig_loss.item(), spread))
        return hist

    print()
    print("=" * 70)
    print("THE TEST THAT MATTERS: the same model, trained two ways")
    print("=" * 70)
    for lambd, label in [(0.0, "WITHOUT SIGReg (lambda=0)"),
                         (0.09, "WITH SIGReg (lambda=0.09, the config value)")]:
        print()
        print(f"  {label}")
        print(f"    {'step':>5s} {'pred_loss':>11s} {'bell score':>11s} {'spread':>9s}")
        hist = run(lambd)
        for step, pl, sl, sp in hist:
            print(f"    {step:5d} {pl:11.5f} {sl:11.3f} {sp:9.5f}")
        final_spread = hist[-1][3]
        if lambd == 0.0:
            collapsed_spread = final_spread
            print(f"    -> COLLAPSED: spread fell to {final_spread:.5f}. The model")
            print("       cheats -- near-identical summaries make prediction trivial.")
        else:
            print(f"    -> spread held at {final_spread:.5f} "
                  f"({final_spread/collapsed_spread:.1f}x the collapsed run)")
            assert final_spread > collapsed_spread * 3, \
                "SIGReg did not prevent collapse -- something is wrong"
            print("       PASS -- SIGReg prevents the cheat.")
    ds.close()


if __name__ == "__main__":
    _test_statistic_behaviour()
    _test_gradient_pushes_away_from_collapse()
    _test_the_real_thing()
    print()
    print("All checks passed.")
