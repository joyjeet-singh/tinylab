"""Minimal example: plan with the released checkpoint and the temporal cost.

The released weights plan well at short horizons and badly at long ones --
94.0% of goals 25 frames away, 26.0% of goals 100 frames away. That gap is not
the model. It is the planner's objective, which minimises squared Euclidean
distance between embeddings; that quantity stops distinguishing states beyond
about 80 arena units and reverses beyond about 120.

Scoring candidates with a learned steps-to-reach head instead lifts the long
horizon to 98.0% with no change to the encoder or predictor.

This file is the smallest thing that shows the difference. It is not the
evaluation harness -- for measured numbers use followup_planner_eval.py, which
is what produced the figures above.

    ./.venv/bin/python followup/plan_with_temporal_cost.py \\
        --run runs/<phase2 run dir> --head followup/temporal_head_phase2.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn


class TemporalHead(nn.Module):
    """(z_t, z_goal) -> predicted steps to reach. Symmetric by construction.

    Trained only on how many frames apart two observed states were. No
    position, no reward, no privileged state: the supervision is available
    from any recorded trajectory.
    """

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Softplus())

    def forward(self, za, zb):
        return 0.5 * (self.net(torch.cat([za, zb], -1))
                      + self.net(torch.cat([zb, za], -1))).squeeze(-1)

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu")
        head = cls(blob["dim"])
        head.load_state_dict(blob["state_dict"])
        head.eval()
        return head


def cem_cost_latent(imagined, goal):
    """The published objective: squared L2 between embeddings.

    Blind beyond ~80 arena units, and inverted beyond ~120 -- moving away from
    the goal can lower this.
    """
    return (imagined - goal).pow(2).sum(-1)


def cem_cost_temporal(imagined, goal, head):
    """Predicted steps to reach. Charges 24% more to cross the dividing wall
    at matched spatial separation, which squared latent distance does not."""
    return head(imagined, goal.expand(len(imagined), -1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="followup/temporal_head_phase2.pt")
    args = ap.parse_args()

    head = TemporalHead.load(args.head)
    d = head.net[0].in_features // 2
    print(f"loaded temporal head, embedding dim {d}")

    # A synthetic sanity check: the head must be symmetric and non-negative,
    # and must return something larger for a pair it considers farther apart.
    torch.manual_seed(0)
    za, zb = torch.randn(4, d), torch.randn(4, d)
    with torch.no_grad():
        ab, ba = head(za, zb), head(zb, za)
    assert torch.allclose(ab, ba, atol=1e-5), "head is not symmetric"
    assert (ab >= 0).all(), "steps-to-reach must be non-negative"
    print(f"symmetric and non-negative; sample predictions {ab.tolist()}")
    print("\nTo plan with it, swap the CEM objective:")
    print("  cost = cem_cost_temporal(imagined_final_embedding, goal_embedding, head)")
    print("instead of")
    print("  cost = cem_cost_latent(imagined_final_embedding, goal_embedding)")
