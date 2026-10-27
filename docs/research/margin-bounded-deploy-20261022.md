# Margin-bounded deployment: development outcome

**Date:** 2026-10-22. **Branch:** `blue/margin-bounded-override` (worktree
`worktrees/margin`), based on `origin/blue/finite-red-check` (`0dffc4a`).
Implementation: `blue/training/harness_margin.py`,
`blue/tests_blue/test_harness_margin.py` (7 tests passing together with the
anchor pilot's tests). Read-only inputs from `worktrees/blue-finite-red`'s
frozen artifacts; scorer `blue/results/gpt_harness/scorer.pkl`
(`94904879...`), matching the pilot manifests.

## Question

The anchored pilot's frozen greedy policy was bit-identical to guard-only
(zero deviation, 72 paired episodes). Its own doc prescribed measuring
candidate logit margins at saved decisions and pre-registering one bounded
deviation rule. Does a deployment rule that overrides Lancer only when the
actor's margin is confident, capped at K per episode, unlock a better policy
from the already-trained checkpoints?

## Rule

`bounded_choice(margin, tau, budget)`: keep Lancer's candidate unless
`margin >= tau` (>= 0) and the episode budget (K) is not exhausted.
Nonfinite margins raise. Recordings include the Blue-visible Lancer scores,
combined/anchored logits, pick, teacher and budget state. Rollouts are
identical to Lancer except at the deviating decisions: repair counts equal
Lancer's wherever no override fires.

## Diagnostic passes (teacher-forced; margins only, no deviations)

`diag_a/m{0,1,2}` anchor arm at mass 0.9; `diag_current/c{0,1,2}` current arm
at bonus 0.25; seeds 7665-7688. Both reproduce the frozen mean exactly
(−150.21 = guard-only), giving an artifact-identity check for free.

1. **Anchor arm is inert.** All margins at mass 0.9 are deeply negative
   (max −0.86, median −4.3; exactly 0 of 46,239 decisions in reach of any
   tau). Recorded reference probabilities allow exact margin recompute at
   other masses: at mass 0.75 the max live margin is +0.24 (confirmed live
   under `--mass 0.75 --tau 2.0 --budget 3`, 0 overrides). The anchored
   training withdrew the actor head to near-zero residuals, so no
   deployment mass lets it deviate usefully. Note: an earlier offline
   margin table claiming 3.4 p.c. at tau 2.0 was wrong (probability used
   where its logarithm belonged); the live runs are authoritative.
2. **Current arm has logit mass but only where it loses.** Combined-score
   margins under teacher-forced rollouts: c1 p99 +0.22 (max 0.24), c0 max
   +0.07, c2 max +0.06. A shared threshold can only be exploited by c1.

## Pre-registered development evaluation (current arm, `dev_current/c{0,1,2}`)

Frozen before the run in `prereg_current_dev.json`: policy-version
current-residual-v1, bonus 0.25, tau 0.20, budget 3, 24 seeds 7665-7688,
native return; no per-seed tau tuning.

| seed | mean | vs Lancer | vs guard | overrides | vs free-greedy |
|---|---:|---:|---:|---:|---:|
| c0 | −150.21 | +1.58 | 0.00 | 0 | +25.50 |
| c1 | −154.54 | −2.75 | −4.33 | 70 (2.92/ep, ≤3 ✓) | +14.50 |
| c2 | −150.21 | +1.58 | 0.00 | 0 | +60.91 |
| arm (df=2) | | **+0.14** CI [−6.08, +6.35] | **−1.44** CI [−7.66, +4.77] | | +33.64 CI [−26.61, +93.89] |

## Decision

The mechanism works exactly as specified (budget respected; identical rollouts
when no override fires; the earlier 85 p.c. sampled-override chaos is contained
to ≤3 confident moves per episode, worth +33.7 return vs the free-greedy arm),
**but there is no positive learning to harvest**: the one checkpoint with
enough margin to deviate (c1) loses about 4 points per episode against the
guard it mostly copies. The dev-confirmation gate (exploratory superiority
over guard-only) is not met, so the planned fresh-block confirmation is
cancelled and no new seed block will be spent on this cohort.

## Interpretation, and what this closes

Three independent arms of this project now measure the same phenomenon:
label-led imitation lands at or just above Lancer/guard; the learned residuals
either collapse to the teacher (anchored arm, this margin study) or actively
lose when they deviate (current arm, c1; the six native-reward actor runs; the
shaping matrix). No existing checkpoint of this cohort contains
positive-margin information under native reward. The margin rule is retained
as reusable deployment machinery, but its value is containment, not
improvement.

## Costs and verification

- 12 simulator jobs (3 diag anchor + 1 anchor dev run, 3 diag current + 3
  eval current + baselines reused from the frozen pilot), no retraining.
- Tests: `pytest blue/tests_blue/test_harness_margin.py
  blue/tests_blue/test_harness_anchor.py` — 7 passed. Earlier bash runs of
  the pilot's anchor tests confirmed determinism (identical means across
  fresh invocations).
- Known limits: teacher-forced margins are the deployment-matched margin set,
  not the actor-policy margin set; one dev block; n=3 training seeds; no
  claim of generalization. 7809-8200 untouched; no fresh seed block spent.
