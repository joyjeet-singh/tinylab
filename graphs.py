"""
graphs.py -- data generator for the k-hop reachability task.

Plain-language summary
----------------------
We make small networks of dots joined by lines, then ask one question about
each: "Can you walk from dot A to dot B along the lines?" (yes / no).

The one knob that sets difficulty is how many steps the shortest walk takes.
To get clean, balanced coverage of every distance -- including the long ones a
small random network almost never produces -- we *plant* a path of the exact
length we want between A and B, then hang extra dots and a few extra lines
around it as distractors, being careful never to add a line that would create
a shorter route. So the true A-to-B distance stays exactly what we asked for.

For "no" questions we build two separate clusters and put A in one and B in the
other, so there is genuinely no path -- but each cluster is big enough that a
model has to actually explore to be sure, rather than just counting clusters.

Every dot is given a fresh random name on every example, so a model can't
memorise "dot 7 is usually reachable" -- it has to trace the structure.

Everything is driven by a data seed, kept separate from any training seed, so
the dataset is reproducible on its own.

Run `python graphs.py` for a smoke test that generates a batch and checks, with
an independent shortest-path computation, that every planted distance is exact.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Distances we train on vs. test on. Test distances are strictly longer than
# any seen in training -- that gap is the "harder than trained" test.
TRAIN_DISTANCES = [1, 2, 3, 4, 5]
TEST_DISTANCES = [6, 7, 8, 9, 10, 11, 12]


@dataclass
class Instance:
    """One question. `edges` are undirected (each pair listed once)."""
    n_nodes: int
    edges: list[tuple[int, int]]
    query: tuple[int, int]          # (A, B)
    reachable: int                  # 1 = yes, 0 = no
    distance: int                   # shortest steps A->B, or -1 if unreachable
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# shortest-path check (an independent oracle, used to VERIFY the generator)
# ---------------------------------------------------------------------------
def _adjacency(n_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def shortest_distance(n_nodes: int, edges: list[tuple[int, int]],
                      s: int, t: int) -> int:
    """Steps on the shortest walk from s to t, or -1 if none. Plain breadth-first search."""
    if s == t:
        return 0
    adj = _adjacency(n_nodes, edges)
    seen = [False] * n_nodes
    seen[s] = True
    frontier = deque([(s, 0)])
    while frontier:
        node, dist = frontier.popleft()
        for nxt in adj[node]:
            if nxt == t:
                return dist + 1
            if not seen[nxt]:
                seen[nxt] = True
                frontier.append((nxt, dist + 1))
    return -1


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def _relabel(n_nodes: int, edges: list[tuple[int, int]], query: tuple[int, int],
             rng: np.random.Generator) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Give every dot a fresh random name, so dot identity carries no signal."""
    perm = rng.permutation(n_nodes)
    new_edges = [(int(perm[u]), int(perm[v])) for u, v in edges]
    # canonicalise each undirected pair + shuffle line order too
    new_edges = [tuple(sorted(e)) for e in new_edges]
    rng.shuffle(new_edges)
    s, t = query
    return new_edges, (int(perm[s]), int(perm[t]))


def _would_shortcut(n_nodes: int, edges: list[tuple[int, int]],
                    candidate: tuple[int, int], s: int, t: int,
                    target_d: int) -> bool:
    """True if adding `candidate` makes the A-to-B distance shorter than target_d."""
    test_edges = edges + [candidate]
    return shortest_distance(n_nodes, test_edges, s, t) < target_d


def make_reachable_instance(n_nodes: int, distance: int, extra_edges: int,
                            rng: np.random.Generator) -> Instance:
    """A 'yes' question whose true A-to-B distance is exactly `distance`."""
    assert distance >= 1
    assert distance + 1 <= n_nodes, "network too small for this distance"

    # 1) plant the backbone path: 0 - 1 - ... - distance
    backbone = list(range(distance + 1))
    edges: list[tuple[int, int]] = [(backbone[i], backbone[i + 1])
                                    for i in range(distance)]
    s, t = backbone[0], backbone[-1]

    # 2) hang every remaining dot off the current graph as a pendant (a tree).
    #    Tree attachments never create a shortcut, so A-B distance stays exact.
    attached = list(backbone)
    for node in range(distance + 1, n_nodes):
        anchor = int(rng.choice(attached))
        edges.append(tuple(sorted((anchor, node))))
        attached.append(node)

    # 3) sprinkle a few extra lines, but only ones that don't shorten A->B.
    added, attempts = 0, 0
    while added < extra_edges and attempts < 20 * (extra_edges + 1):
        attempts += 1
        a, b = int(rng.integers(n_nodes)), int(rng.integers(n_nodes))
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair in edges:
            continue
        if _would_shortcut(n_nodes, edges, pair, s, t, distance):
            continue
        edges.append(pair)
        added += 1

    edges, (s, t) = _relabel(n_nodes, edges, (s, t), rng)
    true_d = shortest_distance(n_nodes, edges, s, t)
    assert true_d == distance, f"planting failed: wanted {distance}, got {true_d}"
    return Instance(n_nodes, edges, (s, t), 1, distance,
                    meta={"extra_edges": added})


def make_unreachable_instance(n_nodes: int, extra_edges: int,
                              rng: np.random.Generator) -> Instance:
    """A 'no' question: A and B sit in two separate, non-trivial clusters."""
    assert n_nodes >= 4
    # split the dots into two clusters
    half = n_nodes // 2
    group_a = list(range(half))
    group_b = list(range(half, n_nodes))

    edges: list[tuple[int, int]] = []

    def grow_tree(group: list[int]) -> None:
        # random spanning tree over the group, so it's one connected blob
        if len(group) < 2:
            return
        order = list(group)
        rng.shuffle(order)
        for i in range(1, len(order)):
            anchor = int(rng.choice(order[:i]))
            edges.append(tuple(sorted((anchor, order[i]))))

    grow_tree(group_a)
    grow_tree(group_b)

    # extra lines allowed only WITHIN a cluster (can never connect the two)
    added, attempts = 0, 0
    while added < extra_edges and attempts < 20 * (extra_edges + 1):
        attempts += 1
        group = group_a if rng.random() < 0.5 else group_b
        if len(group) < 2:
            continue
        a, b = int(rng.choice(group)), int(rng.choice(group))
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair in edges:
            continue
        edges.append(pair)
        added += 1

    s = int(rng.choice(group_a)) if group_a else 0
    t = int(rng.choice(group_b)) if group_b else n_nodes - 1

    edges, (s, t) = _relabel(n_nodes, edges, (s, t), rng)
    assert shortest_distance(n_nodes, edges, s, t) == -1, "clusters got connected"
    return Instance(n_nodes, edges, (s, t), 0, -1,
                    meta={"comp_a_size": half, "comp_b_size": n_nodes - half})


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------
def generate_dataset(n_examples: int, distances: list[int], n_nodes: int = 32,
                     frac_reachable: float = 0.5, extra_edges: int = 3,
                     data_seed: int = 0) -> list[Instance]:
    """A balanced batch: half 'yes' spread evenly over `distances`, half 'no'."""
    rng = np.random.default_rng(data_seed)
    n_yes = int(round(n_examples * frac_reachable))
    n_no = n_examples - n_yes

    out: list[Instance] = []
    for i in range(n_yes):
        d = distances[i % len(distances)]
        out.append(make_reachable_instance(n_nodes, d, extra_edges, rng))
    for _ in range(n_no):
        out.append(make_unreachable_instance(n_nodes, extra_edges, rng))

    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# a starting-point tokeniser (swap freely once the model interface is fixed)
# ---------------------------------------------------------------------------
class Vocab:
    """Tiny fixed vocabulary: a few special tokens, then one id per dot name."""
    def __init__(self, n_nodes: int):
        self.PAD, self.EDGE, self.QUERY, self.BOS = 0, 1, 2, 3
        self.node_offset = 4
        self.size = self.node_offset + n_nodes

    def node(self, i: int) -> int:
        return self.node_offset + i


def to_tokens(inst: Instance, vocab: Vocab) -> list[int]:
    """Serialise a question to a list of token ids: edges, then the A/B query."""
    toks = [vocab.BOS]
    for u, v in inst.edges:
        toks += [vocab.node(u), vocab.node(v), vocab.EDGE]
    s, t = inst.query
    toks += [vocab.QUERY, vocab.node(s), vocab.node(t)]
    return toks


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
def _smoke() -> None:
    print("Generating a mixed batch (train distances) ...")
    data = generate_dataset(400, TRAIN_DISTANCES, n_nodes=32, data_seed=0)

    dist_counts: dict[int, int] = {}
    yes = no = 0
    for inst in data:
        # re-verify EVERY instance with the independent oracle
        d = shortest_distance(inst.n_nodes, inst.edges, *inst.query)
        assert d == inst.distance, "label/oracle mismatch"
        if inst.reachable:
            yes += 1
            dist_counts[inst.distance] = dist_counts.get(inst.distance, 0) + 1
        else:
            no += 1

    print(f"  total={len(data)}  yes={yes}  no={no}")
    print(f"  yes-by-distance: {dict(sorted(dist_counts.items()))}")

    print("Checking the long TEST distances plant exactly ...")
    rng = np.random.default_rng(1)
    for d in TEST_DISTANCES:
        inst = make_reachable_instance(64, d, extra_edges=4, rng=rng)
        got = shortest_distance(inst.n_nodes, inst.edges, *inst.query)
        print(f"  wanted d={d:>2}  measured d={got:>2}  {'OK' if got == d else 'FAIL'}")
        assert got == d

    print("Tokeniser sample ...")
    v = Vocab(32)
    toks = to_tokens(data[0], v)
    print(f"  vocab size={v.size}  first question -> {len(toks)} tokens")
    print("All checks passed.")


if __name__ == "__main__":
    _smoke()
