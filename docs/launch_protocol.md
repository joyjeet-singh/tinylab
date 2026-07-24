# Launch protocol — every dollar spent on purpose

The problem this solves: three paid runs, and each needed follow-up work whose
sole job was diagnosing the previous run. The pattern behind every one of
those burns is the same: the failures all fell OUTSIDE the coverage of our
executable gates, and everything INSIDE gate coverage (resume equivalence,
data fingerprints, md5 retrieval) never failed once. The fix is therefore not
"be more careful" -- checklists decay -- it is to widen gate coverage until
every foreseeable failure class has a machine that refuses to let it launch.

Two design principles, learned the expensive way:

1. ONLY EXECUTABLE CHECKS COUNT. A gate is a script that fails loudly, or an
   abort criterion written down BEFORE launch with a number in it. "I'll keep
   an eye on it" is not a gate.
2. DESIGN GATES AROUND DECISIONS, NOT CODE. Before a run: write down the
   decision the run will feed, list every input that decision consumes, and
   require that each input's validity is checked by a machine. Run 1 burned
   us not because training failed but because the DECISION ("score the best
   checkpoint") consumed an artifact (ckpt_best.pt) that nothing verified
   would exist.

## The failure record, and the gate that now covers each

| What burned us | Run | Gate that would have caught it |
|---|---|---|
| Loss oscillated 10 epochs (LR too high) | 0 | A1: early-descent abort criterion, first 30 min on the box |
| Best checkpoint never saved; good epoch lost | 1 | G1: artifact audit against the decision table, rehearsed free |
| Patch double-applied; schedule silently cyclic | 2 | G2: expected-behavior assertion (simulate the LR sequence, assert it); plus: duplicate banner line was visible in minute one -- A2 log review vs the expected card |
| Eval harness ran in the wrong world for 3 runs | all | G3: domain guard -- evals refuse to score off-manifold inputs |
| Run artifacts lost to a failed browser download | (pre) | Already law: retrieval-first, pull+md5 before destroy |

## Stage 0 — free, on the Mac, before renting anything

G1. DRY-RUN IDENTITY + ARTIFACT AUDIT. In a FRESH CLONE (catches uncommitted
    dependencies -- the view scripts sat untracked for days), run the entire
    pipeline at miniature scale on the toy: train a few epochs, run every
    eval the decision table names, then run an audit script that asserts the
    run directory contains every artifact the pre-registered decision
    consumes (ckpt_best.pt, per-epoch step0 lines in metrics.jsonl, ...).
    The audit script is committed and reused verbatim in Stage 1 and 2.

G2. EXPECTED-BEHAVIOR ASSERTIONS for any run-affecting code change. Not
    "the patch applied" but "the code now does what we intend": simulate the
    scheduler for the planned epochs with a dummy optimizer and assert the
    exact LR sequence; assert construction/step counts. Patches stay
    anchored, idempotent, loud -- and are followed by a global-invariant
    verify, not just anchor uniqueness (two DIFFERENT patch wordings each
    passed their own asserts; only a global count catches that).

G3. DOMAIN GUARD in every evaluation. Before printing a single number, an
    eval encodes a small sample of its own input frames and a small sample of
    the training file's frames, and refuses to score if the eval inputs sit
    far off the training cloud (nearest-neighbor distance >> within-training
    spacing) or if the data fingerprint mismatches the manifest. This is
    Check B, miniaturized and made a precondition. It converts the class of
    failure that cost us three runs' worth of planner numbers into a
    two-minute automatic refusal.

G4. THE EXPECTED CARD. One page, auto-generated and committed with the
    prereg before launch: per-epoch expected LR, expected artifact list,
    the abort criteria with numbers, and the exact first-50-lines the launch
    log should contain. Nothing "kept in mind"; everything on the card.

## Stage 1 — first ~30 minutes on the rented box (~$0.30), gates with teeth

A1. EARLY DESCENT. Pre-committed: if training loss has not dropped below
    [value on the card] by step [N on the card], abort and investigate free.
    Run 0's oscillation was visible this early; it cost a full run to learn.
A2. LOG VS CARD. Read the first 50 log lines against the expected card.
    Anything printed twice, any LR that disagrees with the card's sequence,
    any missing line: abort. (Run 2's doubled scheduler banner was in
    minute one.)
A3. LIVE ARTIFACT AUDIT. After epoch 0's eval, run the G1 audit script on
    the live run dir. ckpt_best.pt exists, step0 logged, or abort.

A run only counts against the experiment budget after Stage 1 passes.
An abort costs cents and produces a bug report, not a wasted run.

## Stage 2 — unchanged, already law

Retrieval proven before launch; tar + rsync + md5 before destroy; the G1
audit re-run on the pulled tarball as the final act before the box dies.

## What this cannot do, said honestly

Gates are regression tests for process: they make every PAST failure class
unrepeatable and every FORESEEABLE one cheap. They cannot catch a failure
class nobody has conceived of yet -- the eval-world gap needed the concept
"measure where the eval's inputs come from" to exist before it could be a
gate, and that concept came from the failure. The commitment is therefore
not "no run will ever surprise us" but: no run fails twice for the same
reason, no foreseeable failure reaches the paid stage, and every surprise
gets converted into a gate the same week it is understood.
