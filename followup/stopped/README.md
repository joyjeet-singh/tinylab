# Stopped runs — NOT results

Every directory here is a planning run that was deliberately stopped before
completing its 50 episodes. The episode logs are real but the success rates
are partial, and an incomplete run is not a measurement: the episode draw is
ordered, so any prefix is a biased sample. We learned that the hard way in
the parent project, where the first twelve episodes of a run happened to be
exactly its baseline's failures.

Nothing here may be quoted. The completed runs are the siblings of this
directory with 50/50 episodes.

| directory | stopped at | why |
|---|---|---|
| `probe_h10_off100` | 3/50 | planning horizon 10 with the probe cost; superseded once the temporal cost reached 98.0% at horizon 5, so the extra lookahead had nothing left to buy |
| `sub3_off100` | 3/50 | oracle subgoal decomposition. It worked in a smoke test, but subgoals came from the recorded trajectory — privileged information a deployed planner would not have. The probe and temporal costs need none, so this was dropped |
| `temporal_authors_off100` | 8/50 | the v1 temporal head on the authors' weights. Superseded by the v2 head once the v1 head was shown to degrade 69% on imagined embeddings, which is the distribution the planner scores |
