"""
toy_model.py -- the world model, sized for the 32x32 toy world.

Written from scratch, but structurally faithful to the reference (le-wm/jepa.py
and le-wm/module.py). Where we deviate, it is marked DEVIATION and explained.

The four parts, in plain terms
------------------------------
1. ENCODER: picture -> a short list of numbers ("summary"). The reference uses a
   ViT-Tiny (5.5M numbers) on 224x224 images and keeps only the CLS token -- one
   summary vector per frame. At 32x32 a ViT is overkill, so we use a small
   convolutional net that produces the same SHAPE of thing.
   -> DEVIATION (toy only): CNN instead of ViT-Tiny. The real run swaps it back.

2. ACTION ENCODER: "move by [-0.35, 1.0]" -> a bundle of numbers the predictor
   can use. Faithful to the reference's Embedder.

3. PREDICTOR: the actual world model. Takes the last few summaries plus the
   actions, predicts the NEXT summary. Not the next picture -- the next summary.
   That is the whole efficiency trick. Faithful to the reference's ARPredictor:
   a causal transformer where the action conditions each block (AdaLN-zero).

4. PROJECTORS: two of them, and this is worth getting right.
   - `projector` sits right after the encoder. Its output IS the embedding --
     both the thing the predictor consumes AND the thing the anti-collapse term
     acts on. (I had earlier described it as giving SIGReg a separate view. That
     was WRONG; reading jepa.py shows encode() applies it inline and stores the
     result as `emb`.)
   - `pred_proj` sits after the predictor, on its output.
   Both are MLPs with a normalisation layer in the middle.

What "history" means here
-------------------------
The predictor sees `history_size` frames of context. Note the unresolved
conflict, carried from the pre-registration: the paper says history length 1 for
Two-Room; the repo config sets history_size: 3 globally. We follow the repo (3),
because the repo is the reference and the paper is the description.

Run `python toy_model.py` for a smoke test: shapes end to end, parameter counts,
and a check that gradients actually flow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. ENCODER  (picture -> summary)
# ---------------------------------------------------------------------------
class ViTEncoder(nn.Module):
    """
    A REAL ViT-Tiny -- the reference's actual encoder -- with the patch size
    adjusted so the geometry works at whatever image size we hand it.

    The reference uses patch_size 14 on 224x224 images = 256 patches, and keeps
    only the CLS token as the summary. At 32x32 a 14-pixel patch does not fit,
    so we use patch_size 4 -> 64 patches. Everything else (12 layers, 3 heads,
    192-dim, CLS token) is exactly the reference's.

    Why this matters, and why it is NOT the same as the CNN stand-in: a ViT is
    a different animal -- position embeddings, patch geometry, CLS token
    handling, and memory shape all differ. Those are the parts most likely to
    break on the rented box. Testing them at 32px costs nothing.

    Measured cost, single CPU core, forward+backward:
        32px  patch4  ( 64 tokens): ~2.4 s per batch-16 step
        224px patch14 (256 tokens): ~2.4 s per batch-4  step  (~600 ms/clip)
    A ViT's cost tracks TOKENS, not pixels -- patch size absorbs the
    resolution. 224px is only ~4x the work of 32px, not ~49x.
    """

    def __init__(self, embed_dim: int = 192, img_size: int = 32,
                 patch_size: int = 4, depth: int = 12, heads: int = 3):
        super().__init__()
        from transformers import ViTConfig, ViTModel
        cfg = ViTConfig(hidden_size=embed_dim, num_hidden_layers=depth,
                        num_attention_heads=heads, intermediate_size=embed_dim * 4,
                        image_size=img_size, patch_size=patch_size)
        self.vit = ViTModel(cfg, add_pooling_layer=False)

    def forward(self, x):
        # CLS token only -- one summary vector per frame, as the reference does
        return self.vit(x).last_hidden_state[:, 0]


class ToyEncoder(nn.Module):
    """
    Small convolutional net: (B, 3, 32, 32) -> (B, embed_dim).

    DEVIATION (toy only): a cheap stand-in for ViT-Tiny, kept because it trains
    fast enough to reach a real R^2 overnight on a laptop. Use ViTEncoder to
    test the real thing.
    """

    def __init__(self, embed_dim: int = 64, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),      # 32 -> 16
            nn.GroupNorm(4, width), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),   # 16 -> 8
            nn.GroupNorm(4, width * 2), nn.GELU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),  # 8 -> 4
            nn.GroupNorm(4, width * 4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, embed_dim),
        )

    def forward(self, x):
        return self.net(x)          # (B, embed_dim)


# ---------------------------------------------------------------------------
# 2. ACTION ENCODER  (faithful to reference module.Embedder)
# ---------------------------------------------------------------------------
class ActionEmbedder(nn.Module):
    """(B, T, action_dim) -> (B, T, emb_dim). Mirrors the reference exactly."""

    def __init__(self, input_dim: int = 2, smoothed_dim: int = 10,
                 emb_dim: int = 64, mlp_scale: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float().permute(0, 2, 1)      # (B, D, T) for Conv1d
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)              # back to (B, T, D)
        return self.embed(x)


# ---------------------------------------------------------------------------
# 3. PREDICTOR  (faithful to reference module.ARPredictor)
# ---------------------------------------------------------------------------
def modulate(x, shift, scale):
    """AdaLN-zero: shift and scale a normalised signal by the conditioning."""
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """Causal self-attention: each frame may look at earlier frames only."""

    def __init__(self, dim, heads=4, dim_head=32, dropout=0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads, self.dropout = heads, dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, T, _ = x.shape
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConditionalBlock(nn.Module):
    """
    One transformer block where the ACTION steers the computation (AdaLN-zero).

    The action produces six numbers per channel: shift/scale/gate for the
    attention half and the same for the feed-forward half. The gates start at
    exactly zero, so at initialisation the block does nothing at all and the
    signal passes straight through. It learns how much to intervene.
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN[-1].weight, 0)     # zero-init: identity at start
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x, c):
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.adaLN(c).chunk(6, dim=-1)
        x = x + g_a * self.attn(modulate(self.norm1(x), sh_a, sc_a))
        x = x + g_m * self.mlp(modulate(self.norm2(x), sh_m, sc_m))
        return x


class ARPredictor(nn.Module):
    """
    Summaries + actions -> next summaries.

    Learned position markers tell the model which frame is which. Attention is
    causal, so frame t can only see frames <= t -- the model cannot cheat by
    peeking at the future it is meant to predict.
    """

    def __init__(self, *, num_frames, input_dim, hidden_dim, output_dim=None,
                 depth=4, heads=4, dim_head=32, mlp_dim=256, dropout=0.1):
        super().__init__()
        output_dim = output_dim or input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.in_proj = (nn.Linear(input_dim, hidden_dim)
                        if input_dim != hidden_dim else nn.Identity())
        self.cond_proj = (nn.Linear(input_dim, hidden_dim)
                          if input_dim != hidden_dim else nn.Identity())
        self.blocks = nn.ModuleList([
            ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = (nn.Linear(hidden_dim, output_dim)
                         if hidden_dim != output_dim else nn.Identity())

    def forward(self, x, c):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.in_proj(x)
        c = self.cond_proj(c)
        for b in self.blocks:
            x = b(x, c)
        return self.out_proj(self.norm(x))


# ---------------------------------------------------------------------------
# 4. PROJECTORS  (faithful to reference module.MLP)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Linear -> normalise -> activate -> Linear. Used for both projectors."""

    def __init__(self, input_dim, hidden_dim, output_dim=None,
                 norm_fn=nn.BatchNorm1d, act_fn=nn.GELU):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), norm, act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim))

    def forward(self, x):
        return self.net(x)          # expects (B*T, D) -- flattened, per reference


# ---------------------------------------------------------------------------
# THE WHOLE MODEL  (faithful to reference jepa.JEPA)
# ---------------------------------------------------------------------------
class ToyJEPA(nn.Module):
    """
    encode(): pictures -> summaries (via encoder THEN projector), actions -> bundles
    predict(): summaries + action bundles -> predicted next summaries (via
               predictor THEN pred_proj)

    Note what `emb` is: the projector's output. It is BOTH what the predictor
    consumes AND what the anti-collapse term will act on. They are the same
    tensor -- there is no separate view.
    """

    def __init__(self, embed_dim=64, action_dim=2, history_size=3,
                 depth=4, heads=4, dim_head=32, mlp_dim=256,
                 proj_hidden=256, dropout=0.1, enc_width=32,
                 encoder="cnn", img_size=32, patch_size=4,
                 enc_depth=12, enc_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_size = history_size

        if encoder == "vit":
            self.encoder = ViTEncoder(embed_dim, img_size, patch_size,
                                      enc_depth, enc_heads)
        else:
            self.encoder = ToyEncoder(embed_dim, enc_width)
        self.action_encoder = ActionEmbedder(action_dim, 10, embed_dim)
        self.predictor = ARPredictor(
            num_frames=history_size + 1,     # +1: room for the frame we predict
            input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim,
            depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
            dropout=dropout)
        self.projector = MLP(embed_dim, proj_hidden, embed_dim)
        self.pred_proj = MLP(embed_dim, proj_hidden, embed_dim)

    def encode(self, info: dict) -> dict:
        """pixels (B,T,3,H,W) -> emb (B,T,D);  action (B,T,A) -> act_emb (B,T,D)"""
        px = info["pixels"].float()
        B, T = px.shape[:2]
        px = px.reshape(B * T, *px.shape[2:])          # flatten time into batch
        h = self.encoder(px)                            # (B*T, D)
        emb = self.projector(h)                         # (B*T, D)  <- this IS `emb`
        info["emb"] = emb.reshape(B, T, -1)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb):
        """(B,T,D) + (B,T,D) -> (B,T,D). Frame t predicts frame t+1."""
        B = emb.size(0)
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(preds.reshape(B * preds.size(1), -1))
        return preds.reshape(B, -1, self.embed_dim)

    def forward(self, info: dict) -> dict:
        info = self.encode(info)
        info["pred"] = self.predict(info["emb"], info["act_emb"])
        return info


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
def _smoke():
    torch.manual_seed(0)
    B, T, D = 8, 4, 64          # 4 frames per clip = history 3 + 1 predicted

    model = ToyJEPA(embed_dim=D, history_size=3)
    counts = {n: sum(p.numel() for p in m.parameters())
              for n, m in [("encoder", model.encoder),
                           ("action_encoder", model.action_encoder),
                           ("predictor", model.predictor),
                           ("projector", model.projector),
                           ("pred_proj", model.pred_proj)]}
    total = sum(p.numel() for p in model.parameters())
    print("parameter counts")
    for n, c in counts.items():
        print(f"  {n:16s}: {c/1e3:8.1f}k")
    print(f"  {'TOTAL':16s}: {total/1e6:8.3f}M   (reference LeWM: ~15-18M at 224px)")

    info = {"pixels": torch.rand(B, T, 3, 32, 32),
            "action": torch.randn(B, T, 2).clamp(-1, 1)}

    print()
    print("shapes end to end")
    out = model(info)
    print(f"  pixels   {tuple(info['pixels'].shape)}")
    print(f"  action   {tuple(info['action'].shape)}")
    print(f"  emb      {tuple(out['emb'].shape)}      <- summaries (projector output)")
    print(f"  act_emb  {tuple(out['act_emb'].shape)}      <- action bundles")
    print(f"  pred     {tuple(out['pred'].shape)}      <- predicted next summaries")
    assert out["emb"].shape == (B, T, D)
    assert out["pred"].shape == (B, T, D)

    print()
    print("the prediction task: frame t predicts frame t+1")
    loss = F.mse_loss(out["pred"][:, :-1], out["emb"][:, 1:].detach())
    print(f"  pred[:, :-1] vs emb[:, 1:]  ->  loss {loss.item():.4f}")

    print()
    print("do gradients flow?")
    loss.backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    print(f"  at init, {len(dead)} parameters get no gradient.")
    print("  This is EXPECTED, not a bug: the AdaLN gates start at exactly zero,")
    print("  so the action's contribution is multiplied by 0 and nothing upstream")
    print("  of the gates (the action encoder) receives any gradient yet. The")
    print("  gates learn to open, and then it does. We verify that below.")

    # nudge the gates open, as a few training steps would, then re-check
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "adaLN" in n and "weight" in n:
                p.add_(torch.randn_like(p) * 0.01)

    info3 = {"pixels": torch.rand(B, T, 3, 32, 32),
             "action": torch.randn(B, T, 2).clamp(-1, 1)}
    out3 = model(info3)
    loss3 = F.mse_loss(out3["pred"][:, :-1], out3["emb"][:, 1:].detach())
    model.zero_grad()
    loss3.backward()

    dead3 = [n for n, p in model.named_parameters()
             if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    act_grad = sum(p.grad.abs().sum().item()
                   for p in model.action_encoder.parameters() if p.grad is not None)
    print(f"  once the gates open: {len(dead3)} parameters without gradient")
    print(f"  action encoder gradient: {act_grad:.3e}  (was exactly 0 at init)")
    assert len(dead3) == 0, f"still dead after gates open: {dead3[:5]}"
    assert act_grad > 0, "action encoder never receives gradient -- wiring is broken"
    print("  PASS -- every part is reachable; the action really does steer prediction")
    model.zero_grad()

    print()
    print("causality check: does frame t see the future?")
    print("  (in eval mode -- see the note below on why that matters)")
    model.eval()
    info2 = {"pixels": info["pixels"].clone(), "action": info["action"].clone()}
    info2["pixels"][:, -1] = torch.rand(B, 3, 32, 32)     # change ONLY the last frame
    with torch.no_grad():
        outA = model({"pixels": info["pixels"].clone(),
                      "action": info["action"].clone()})
        outB = model(info2)
    d_first = (outA["pred"][:, 0] - outB["pred"][:, 0]).abs().max().item()
    d_last = (outA["pred"][:, -1] - outB["pred"][:, -1]).abs().max().item()
    print(f"  changing the LAST frame moves pred[0]  by {d_first:.2e}")
    print(f"  changing the LAST frame moves pred[-1] by {d_last:.2e}")
    assert d_first < 1e-6, "frame 0's prediction saw the future -- causality broken"
    print("  PASS -- early predictions cannot see later frames (attention is causal)")
    model.train()

    print()
    print("  NOTE, and it matters later: in TRAIN mode, changing the last frame DOES")
    print("  nudge pred[0] slightly. That is not an attention leak -- it is")
    print("  BatchNorm in the projectors. The reference flattens time into the")
    print("  batch before the projector, so BatchNorm pools statistics over all")
    print("  frames of all clips together, and every frame's summary depends a")
    print("  little on every other. This is faithful to the reference, not a bug.")
    print("  Worth knowing before it surprises you while debugging.")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    _smoke()
