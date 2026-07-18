# Week 2 spec — reinforcement learning from scratch

Goal: implement REINFORCE, then PPO, as single readable files inside the
tinylab scaffold, on CartPole (discrete actions) and Pendulum (continuous).
CleanRL is consulted only AFTER each build, as an answer key.

Why this week exists: world models are trained on trajectories, and planning
is judged by control performance. RL is where you learn what trajectories,
returns, and credit assignment feel like from the inside. It also foreshadows
RLVR-style reasoning training (Lane B).

## The one big conceptual shift (read before coding)

In Week 1, the dataset sat in a folder. It never changed. In RL, THE MODEL'S
OWN BEHAVIOR GENERATES THE DATA. The policy acts, the world responds, and
those experiences are the only training material. Two consequences:

1. Non-stationarity: as the policy improves, the data distribution changes
   under your feet. Yesterday's experiences describe a worse policy's world.
2. Credit assignment: reward arrives at the end of (or during) a long episode.
   Which of the 200 actions deserves the credit? This is THE problem; every
   algorithm this week is one answer to it.

REINFORCE's answer, in plain words: after an episode, nudge the knobs so that
every action you took becomes more probable, scaled by how good the total
outcome after that action was. Good outcome -> repeat those choices more.
Bad outcome (relative to average) -> repeat them less. The gradient of
log-probability is the "how do I make this action more likely" direction;
the return is the volume knob on that nudge.

## Setup

- `pip install gymnasium` then IMMEDIATELY pin the version that installed
  into requirements.txt (pip freeze | grep -i gymnasium >> requirements.txt).
- Gymnasium API notes (this trips everyone):
  - `obs, info = env.reset(seed=...)` — the ENV has its own seed, separate
    from torch's. Both must be set from the run seed for determinism.
  - `obs, reward, terminated, truncated, info = env.step(action)` — five
    return values. `terminated` = the task ended (pole fell). `truncated` =
    time limit hit (CartPole-v1 cuts episodes at 500 steps). For REINFORCE
    treat both as episode end (known simplification — note it in the code).
- Refactor first: extract the manifest+JSONL logging from train.py into a
  shared `lablog.py` (functions: `start_run(cfg, extra_manifest) -> run_dir,
  log_fn`). train.py and the RL scripts both import it. Commit the refactor
  alone, then verify Week 1 still reproduces (rerun mlp seed 0, diff the
  metrics against the old run's file). Only then start RL.

## Part A — REINFORCE on CartPole-v1 (~120 lines, rl/reinforce.py)

Recipe card `configs/reinforce.yaml`:
  env: CartPole-v1, episodes: 1500, gamma: 0.99, lr: 0.01,
  hidden: [128], seed: 0 (overridden per run), log_every: 10

Components:
- Policy net: MLP obs(4) -> 128 -> ReLU -> 2 logits. Sample actions with
  `torch.distributions.Categorical(logits=...)`; store log_prob of the
  action actually taken.
- Play one full episode, collecting (log_prob, reward) per step.
- Returns: reward-to-go with discount gamma — G_t = r_t + gamma*G_{t+1},
  computed backwards. Then NORMALIZE returns across the episode
  ((G - mean) / (std + 1e-8)). This is the variance fix that makes
  REINFORCE trainable; without it, learning is agonizing. (The mean
  subtraction is a "baseline": comparing to average doesn't bias the
  direction of the nudge, it only steadies it.)
- Loss = -(log_probs * normalized_returns).sum() — the minus sign because
  optimizers descend and we want ascent. THE most common bug this week is
  this sign.
- One optimizer step per episode.

Logging (per episode, to metrics.jsonl): episode index, raw return (sum of
undiscounted rewards), episode length, policy entropy (mean of the
Categorical entropy over the episode — the policy's remaining indecision).

analyze.py additions: a second mode or new script `analyze_rl.py` that reads
eval-style records and reports, per seed: final-100-episode mean return,
episodes-to-first-crossing of return 400, and a 50-episode moving-average
curve dumped as CSV for plotting later.

### Checkpoints (write predictions BEFORE running — the Week 1 habit)
1. Untrained policy baseline: log 20 episodes with random init, no training.
   Known physics: random CartPole scores ~20-25.
2. Learning signal: moving-average return should leave the random band and
   cross 100 within the first few hundred episodes.
3. Target: 100-episode average >= 475 ("solved") OR sustained >400 —
   REINFORCE is noisy; not every seed solves it. Run 3 seeds.
4. Reproducibility: rerun seed 0, diff metrics.jsonl — must be identical
   (env seeded + torch seeded + CPU = fully deterministic).
5. Expect the seed spread to be MUCH wider than Week 1's. That is the
   lesson: RL variance dwarfs supervised variance. Grade your prediction.

### Failure signatures (check in this order)
- Return flat at ~20 forever: sign error in loss, or optimizer stepping on
  stale gradients (missing zero_grad), or returns not normalized.
- Return rises then collapses to ~9 with entropy ~0: learning rate too
  high — the policy became deterministic too early and can't explore back.
- Return improves but rerun differs: env not seeded via reset(seed=...).

## Part B — PPO on CartPole, then Pendulum-v1 (~250 lines, rl/ppo.py)

Build only after Part A works and is committed. Concepts to introduce, in
plain words, in this order:
- Actor-critic: add a value head (state -> expected future return). The
  critic is a learned baseline, replacing per-episode normalization.
- Advantage = "how much better than expected did this turn out" (use GAE
  with lambda 0.95; explain as a smoothed advantage estimate).
- The PPO clip: reuse each batch of experience for several gradient epochs,
  but clip the probability ratio to [1-eps, 1+eps] (eps=0.2) so the policy
  can't sprint away from the data that was collected under the old policy.
  Plain words: squeeze more learning from each batch, with a leash.
- Truncation now matters: bootstrap the value of the final state when an
  episode was truncated rather than terminated.
- Pendulum: continuous actions. Policy outputs mean + learned log_std of a
  Normal; sample, then clip to [-2, 2] (note the clipping bias in a
  comment; it's acceptable here). Random policy scores about -1200 per
  episode; decent learned policies reach -300 to -150. Rewards are dense
  and negative — do not panic at negative returns.

Checkpoints: PPO solves CartPole faster and more reliably than REINFORCE
(3/3 seeds expected); Pendulum reaches better than -400 average. Then, and
only then, open CleanRL's ppo.py and diff conceptually: what did they add
that you skipped (advantage normalization, value clipping, lr annealing,
parallel envs)? Write a short docs/week2_notes.md on what each extra buys.

## Paper pairing for the week
Read the REINFORCE lineage lightly (any good tutorial derivation) and spend
the deep-read slot on the PPO paper (Schulman et al. 2017), actively: for
each trick, ask "what failure is this preventing?" — you will have just met
several of those failures personally in Part A.

## Discipline reminders
- All numbers from logs via the analyzer. No means in your head.
- Prediction line before every multi-seed run, graded after.
- One heredoc (or downloaded file) per new file; py_compile before running.
- Commit at each checkpoint, not at the end of the week.
