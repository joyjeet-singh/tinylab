"""REINFORCE on CartPole, single file. The one-sentence algorithm:
nudge every action taken toward "more likely", with the volume knob set
by the reward-to-go that followed it, compared against the episode's
own average.

Usage: python reinforce.py --config configs/reinforce.yaml --seed 0
"""
import argparse

import gymnasium as gym
import torch
import torch.nn as nn
import yaml

from lablog import start_run
from seed import set_seed


def build_policy(obs_dim, n_actions, hidden):
    layers, d = [], obs_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, n_actions))   # one score per possible action
    return nn.Sequential(*layers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = args.seed
    set_seed(cfg["seed"])

    env = gym.make(cfg["env"])
    run_dir, log, close = start_run(
        cfg, tag=cfg["env"].split("-")[0].lower(),
        extra={"gymnasium_version": gym.__version__})

    policy = build_policy(env.observation_space.shape[0],
                          env.action_space.n, cfg["model"]["hidden"])
    optimizer = torch.optim.Adam(policy.parameters(),
                                 lr=cfg["training"]["learning_rate"])
    gamma = cfg["training"]["gamma"]

    # The env has its OWN dice, separate from torch's. Seed it once here;
    # every later reset() continues its stream deterministically.
    obs, _ = env.reset(seed=cfg["seed"])

    recent = []                       # console convenience only: last 50 returns
    for ep in range(cfg["training"]["episodes"]):
        log_probs, entropies, rewards = [], [], []
        done = False
        while not done:
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()                 # luck, on purpose
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())       # the policy's indecision
            obs, r, terminated, truncated, _ = env.step(int(action))
            rewards.append(float(r))
            done = terminated or truncated  # simplification: treat alike (note)

        # Reward-to-go, computed backwards: G = r + gamma * G.
        # Credit flows only forward in time: an action is judged solely
        # by what happened after it.
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        R = torch.tensor(returns)

        # Baseline + scale: compare to the episode's own average so that
        # below-average moments push their actions DOWN. Subtracting a
        # constant cannot bias the direction (probabilities sum to 1);
        # it only steadies the nudges.
        R = (R - R.mean()) / (R.std(unbiased=False) + 1e-8)

        # Minus sign: optimizers descend, we want to ASCEND probability
        # of well-rewarded actions. The single most common bug this week.
        loss = -(torch.stack(log_probs) * R).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_return = sum(rewards)
        log({"kind": "episode", "episode": ep, "return": ep_return,
             "length": len(rewards),
             "entropy": round(torch.stack(entropies).mean().detach().item(), 6),
             "loss": round(loss.detach().item(), 6)})

        recent.append(ep_return)
        if len(recent) > 50:
            recent.pop(0)
        if (ep + 1) % 50 == 0:
            print(f"episode {ep + 1}: last-50 avg return "
                  f"{sum(recent) / len(recent):.1f}")

        obs, _ = env.reset()

    log({"kind": "done", "episodes": cfg["training"]["episodes"]})
    close()


if __name__ == "__main__":
    main()
