"""
eval_grid.py -- the loops-by-distance diagnostic.

Plain-language summary
----------------------
Run the trained model on the test questions many times over, once for each
number of reasoning-loops we allow, and record how often it's right at each
true distance. The result is a grid: loops across the top, distance down the
side. If the model really spreads information one ring per loop, we should see a
clean triangle -- right whenever loops are at least the distance. Where the
triangle is dented tells us whether a failure is a 'not enough loops' problem or
a 'can't represent it at all' problem.

No training here, and nothing typed by hand -- every number comes from running
the model on held-out questions.
"""
from __future__ import annotations

import torch

try:
    from reachability import model as M
except ImportError:
    import model as M


def loops_by_distance(net, data, loops, n_nodes, batch_size=256):
    """Accuracy at each (loop count, true distance). Returns a nested dict."""
    net.eval()
    dists = sorted({d.distance for d in data if d.reachable})
    # pre-group indices so we don't rebuild batches per loop-count
    reach_by_d = {d: [i for i, x in enumerate(data)
                      if x.reachable and x.distance == d] for d in dists}
    unreach = [i for i, x in enumerate(data) if not x.reachable]

    def accuracy(idxs, want, T):
        if not idxs:
            return None
        correct = 0
        with torch.no_grad():
            for s in range(0, len(idxs), batch_size):
                chunk = [data[i] for i in idxs[s:s + batch_size]]
                batch = M.collate(chunk, n_nodes)
                pred = (torch.sigmoid(net(batch, n_loops=T)["reach_logit"]) > 0.5)
                correct += (pred.long() == want).sum().item()
        return correct / len(idxs)

    grid = {}
    for T in loops:
        grid[T] = {d: accuracy(reach_by_d[d], 1, T) for d in dists}
        grid[T]["unreach"] = accuracy(unreach, 0, T)
    net.train()
    return {"loops": loops, "distances": dists, "grid": grid}


def render(result):
    """A plain text heatmap: rows = distance, columns = loop count."""
    loops, dists, grid = result["loops"], result["distances"], result["grid"]
    head = "dist \\ loops |" + "".join(f"{T:>5}" for T in loops)
    lines = [head, "-" * len(head)]
    for d in dists:
        row = f"    d={d:<7}|" + "".join(
            (f"{grid[T][d]*100:>4.0f} " if grid[T][d] is not None else "   . ")
            for T in loops)
        lines.append(row)
    urow = f"  unreach   |" + "".join(
        (f"{grid[T]['unreach']*100:>4.0f} " if grid[T]['unreach'] is not None else "   . ")
        for T in loops)
    lines.append(urow)
    return "\n".join(lines)
