"""
model.py -- the small reasoning model for the reachability task.

Plain-language summary
----------------------
The model keeps a little bundle of numbers ("state") for every dot. It works in
three parts:

  1. Encoder -- reads the raw network once and gives every dot a starting state.
     Two versions:
       * 'flat'  : each dot only sees its own row of the connection table, fed
                   through a small network. No built-in notion of "neighbour".
                   (This is the weak reader we suspect causes the known failures.)
       * 'gnn'   : dots exchange information with the dots they're wired to, a
                   few times, so the starting state already reflects local shape.

  2. Reasoning loop -- the same small step, repeated. Each repeat lets a dot pull
     in what its neighbours know, so information spreads one ring outward per
     repeat. The number of repeats is a dial we can change at test time, because
     the step reuses the same weights every time. Two versions of the step:
       * 'plain'      : a generic learned update.
       * 'structured' : a physics-shaped update that splits the state into two
                        coupled halves and nudges them in a stable, reversible
                        way -- biased to keep spreading cleanly over many repeats
                        instead of blowing up. (First-pass stand-in for your
                        port-Hamiltonian step; to be aligned with your exact v3
                        formulation once we see it.)

  3. Two read-outs -- one says yes/no "can you reach dot B", the other marks, dot
     by dot, which dots have been reached (this second one is the "frontier"
     signal that sat at guess-rate in your earlier round).

Networks here are small and fixed size, so we carry the connection table as a
plain dense grid -- short, readable, and fine on modest hardware.

Run `python model.py` for a smoke test: it builds a tiny model, runs it on a
real batch at several repeat-counts, and takes a few training steps to confirm
the loss actually goes down end to end.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:                       # works as a package (from the notebook)
    from reachability import graphs
except ImportError:        # works when run inside this folder
    import graphs


# ---------------------------------------------------------------------------
# turning a batch of questions into tensors
# ---------------------------------------------------------------------------
def _reachable_set(n_nodes, edges, s):
    """All dots reachable from s (used as the per-dot 'frontier' target)."""
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    seen = [False] * n_nodes
    seen[s] = True
    q = deque([s])
    while q:
        node = q.popleft()
        for nxt in adj[node]:
            if not seen[nxt]:
                seen[nxt] = True; q.append(nxt)
    return seen


def collate(instances, n_nodes):
    """List of Instances -> dict of tensors, all shape-aligned on n_nodes."""
    B = len(instances)
    adj = torch.zeros(B, n_nodes, n_nodes)
    src = torch.zeros(B, dtype=torch.long)
    tgt = torch.zeros(B, dtype=torch.long)
    label = torch.zeros(B)
    dist = torch.full((B,), -1, dtype=torch.long)
    frontier = torch.zeros(B, n_nodes)          # per-dot reached-from-s target
    for i, inst in enumerate(instances):
        for u, v in inst.edges:
            adj[i, u, v] = 1.0; adj[i, v, u] = 1.0
        s, t = inst.query
        src[i], tgt[i] = s, t
        label[i] = inst.reachable
        dist[i] = inst.distance
        frontier[i] = torch.tensor(_reachable_set(n_nodes, inst.edges, s),
                                    dtype=torch.float32)
    return {"adj": adj, "src": src, "tgt": tgt,
            "label": label, "dist": dist, "frontier": frontier}


def _norm_adj(adj, add_self):
    """Degree-normalised neighbour averaging, so state sizes stay stable."""
    if add_self:
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        adj = adj + eye
    deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
    return adj / deg


def _mlp(d_in, d_hidden, d_out, n_layers):
    layers, d = [], d_in
    for _ in range(max(1, n_layers)):
        layers += [nn.Linear(d, d_hidden), nn.GELU()]; d = d_hidden
    layers += [nn.Linear(d, d_out)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------
class FlatEncoder(nn.Module):
    """Each dot sees only its own connection row -> a starting state. No mixing."""
    def __init__(self, n_nodes, width, layers):
        super().__init__()
        self.net = _mlp(n_nodes, width, width, layers)

    def forward(self, adj):                       # adj: [B,N,N]
        return self.net(adj)                       # -> [B,N,width]


class GNNEncoder(nn.Module):
    """A few rounds of neighbour exchange -> a structure-aware starting state."""
    def __init__(self, n_nodes, width, layers):
        super().__init__()
        self.embed = nn.Linear(1, width)           # start from degree signal
        self.rounds = nn.ModuleList(
            [_mlp(2 * width, width, width, 1) for _ in range(max(1, layers))])

    def forward(self, adj):
        deg = adj.sum(-1, keepdim=True)            # [B,N,1]
        h = self.embed(deg)
        A = _norm_adj(adj, add_self=True)
        for layer in self.rounds:
            msg = torch.bmm(A, h)                   # average of neighbours
            h = h + layer(torch.cat([h, msg], dim=-1))
        return h


# ---------------------------------------------------------------------------
# reasoning steps (one repeat of the loop)
# ---------------------------------------------------------------------------
class PlainStep(nn.Module):
    """Generic learned update: pull in neighbour states, revise your own."""
    def __init__(self, width):
        super().__init__()
        self.update = _mlp(2 * width, width, width, 1)

    def forward(self, h, A):
        msg = torch.bmm(A, h)
        return h + self.update(torch.cat([h, msg], dim=-1))


class StructuredStep(nn.Module):
    """
    Physics-shaped update. Split state into two halves q, p and nudge them in a
    coupled, reversible (leapfrog) way, with the 'force' coming from neighbours.
    Biased toward stable spreading across many repeats.
    """
    def __init__(self, width):
        super().__init__()
        assert width % 2 == 0, "structured step needs an even width"
        self.half = width // 2
        self.force = _mlp(self.half, width, self.half, 1)
        self.log_dt = nn.Parameter(torch.tensor(0.0))   # learnable step size

    def _force(self, q, A):
        return self.force(torch.bmm(A, q))              # neighbour-driven force

    def forward(self, h, A):
        q, p = h[..., :self.half], h[..., self.half:]
        dt = torch.exp(self.log_dt).clamp(max=1.0)
        p = p + 0.5 * dt * self._force(q, A)
        q = q + dt * p
        p = p + 0.5 * dt * self._force(q, A)
        return torch.cat([q, p], dim=-1)


# ---------------------------------------------------------------------------
# the whole model
# ---------------------------------------------------------------------------
class ReachabilityModel(nn.Module):
    def __init__(self, encoder="gnn", reasoning="structured", width=128,
                 layers=2, n_nodes=32, share_weights=True, default_loops=6):
        super().__init__()
        self.n_nodes = n_nodes
        self.share_weights = share_weights
        self.default_loops = default_loops

        self.encoder = (FlatEncoder(n_nodes, width, layers) if encoder == "flat"
                        else GNNEncoder(n_nodes, width, layers))
        self.source_marker = nn.Parameter(torch.randn(width) * 0.1)

        step_cls = PlainStep if reasoning == "plain" else StructuredStep
        if share_weights:                       # one reused step -> loops are a dial
            self.step = step_cls(width)
        else:                                   # fixed separate steps (deep baseline)
            self.steps = nn.ModuleList([step_cls(width) for _ in range(default_loops)])

        self.reach_head = nn.Linear(width, 1)
        self.frontier_head = nn.Linear(width, 1)

    def _apply_step(self, h, A, i):
        return self.step(h, A) if self.share_weights else self.steps[i](h, A)

    def forward(self, batch, n_loops=None):
        adj = batch["adj"]
        h = self.encoder(adj)
        # inject the start signal at the source dot
        onehot = F.one_hot(batch["src"], self.n_nodes).float().unsqueeze(-1)
        h = h + onehot * self.source_marker

        A = _norm_adj(adj, add_self=False)
        T = n_loops or self.default_loops
        if not self.share_weights:
            T = len(self.steps)
        for i in range(T):
            h = self._apply_step(h, A, i)

        # read the target dot's final state for yes/no
        idx = batch["tgt"].view(-1, 1, 1).expand(-1, 1, h.size(-1))
        h_t = h.gather(1, idx).squeeze(1)
        reach_logit = self.reach_head(h_t).squeeze(-1)      # [B]
        frontier_logits = self.frontier_head(h).squeeze(-1)  # [B,N]
        return {"reach_logit": reach_logit, "frontier_logits": frontier_logits}


def loss_fn(out, batch, frontier_weight=0.5):
    reach = F.binary_cross_entropy_with_logits(out["reach_logit"], batch["label"])
    front = F.binary_cross_entropy_with_logits(out["frontier_logits"], batch["frontier"])
    return reach + frontier_weight * front, reach.item(), front.item()


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
def _smoke():
    torch.manual_seed(0)
    N = 32
    data = graphs.generate_dataset(64, graphs.TRAIN_DISTANCES, n_nodes=N, data_seed=0)
    batch = collate(data, N)

    for enc in ("flat", "gnn"):
        for rea in ("plain", "structured"):
            m = ReachabilityModel(encoder=enc, reasoning=rea, width=64,
                                  layers=2, n_nodes=N)
            for T in (1, 3, 8):
                out = m(batch, n_loops=T)
                assert out["reach_logit"].shape == (64,)
                assert out["frontier_logits"].shape == (64, N)
            print(f"  {enc:>4}/{rea:<10} forward OK at loops 1/3/8")

    print("Can it learn on a small batch? (loss should drop)")
    m = ReachabilityModel(encoder="gnn", reasoning="structured", width=64,
                          layers=2, n_nodes=N)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = last = None
    for step in range(60):
        out = m(batch, n_loops=6)
        loss, r, f = loss_fn(out, batch)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0: first = loss.item()
        last = loss.item()
    print(f"  loss {first:.3f} -> {last:.3f}  (reach {r:.3f}, frontier {f:.3f})")
    assert last < first, "loss did not decrease"
    print("All checks passed.")


if __name__ == "__main__":
    _smoke()
