# Rental Requirements — Vast.ai (TwoRoom real-data run)

Reconstructed from project memory (2026-07-18 rental decision).

## Shopping filter

| Field | Requirement | Why |
|---|---|---|
| Provider | **Vast.ai** | only provider where CPU cores are a *shoppable* axis — and the data path is the bottleneck |
| Instance | **On-demand (NOT interruptible)** | a preemption mid-run wastes the meter; this run is short and cheap enough not to gamble |
| GPU | **RTX 3090** preferred (~$0.15/hr); **4090** fallback | 17.89M params fits trivially — GPU is not the constraint, so buy the cheap one that clears |
| **CPU cores** | **≥ 32** | the real bottleneck. ~25× Blosc read amplification + ~77 MB/batch hand-offs; loading needs cores, not a bigger GPU |
| RAM | **≥ 64 GB** | headroom for the 12 GB file + worker copies + prefetch |
| Disk | **≥ 60 GB** | 12 GB file + checkpoints + logs, with margin |
| Reliability | **≥ 99%** | avoid a flaky host eating the run |
| Sort | **by price** | take the cheapest box that clears every filter above |

## Budget

- Envelope: **$30–50**.
- Projected run cost: **$2–7** (well inside the envelope).
- Hard ceiling for the runbook's Gate C abort: **$50**.

## Platform note

The rental is **Linux → `fork`** multiprocessing. That is the path fully verified in the
container; the Mac's `spawn` path was where the `torch.equal`/NaN red herring lived, and
it's now resolved. The loader's private RNG generator keeps resume deterministic with
workers on, on fork.

## Teardown

**Destroy the instance when done — do not stop it.** Stopped instances still bill for
storage. Pull all artifacts off the box first (see runbook Phase 3).
