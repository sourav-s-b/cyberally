# Blue program summary (general index)

## Input decision audit, 2026-10-13

Branch `blue/input-decision-audit` from `6b608e5` (architecture-longrun work).
The eight reused finite-Red development episodes 8221–8228 were traced with
the exact frozen architecture configuration and scorer. Guard-only returns
reproduced the architecture study; raw risk ranking and a fixed freshness-
adjusted risk ranking both underperformed guard-only on the eight-episode
mean. The latter reduced one large loss but created others. No long training
or final-seed evaluation was run. See
[`blue-input-decision-audit-20261013.md`](../research/blue-input-decision-audit-20261013.md)
for cells, mechanism evidence, artifact hashes, limitations, and the next
decision-to-completion data-collection task. This branch adds the audit and
opt-in diagnostic input, but does not change the frozen training policy.
Focused tests passed; full training tests were not needed for this audit.

PR review: verify only Blue-visible fields reach candidate scoring, confirm
the privileged labels are attached after action selection, verify frozen
hash and seed guards, and retain the two failed ranking rules as diagnostic
rather than claiming a deployment improvement. No dependent branch migration
or shared contract change is required.

## Inspection-yield follow-up, 2026-10-13

Same branch, dependent on the decision audit above. A request-to-completion
collector now saves pre-request Blue-visible data and observed `Analyse`
results in separate episode JSONL files; privileged Red-session truth lives
in a distinct diagnostic file. Twenty-four frozen finite-Red episodes were
collected locally, with 12/4/8 whole-episode train/validation/reused-dev
partitions. A fixed gradient-boosted predictor of new confirmations on
ordinary selectable sweep decisions reached 0.866 episode-macro PR-AUC on
the eight reused development episodes, versus 0.735 for the existing
compromise-risk score on that different target. Three public-objective and
evidence-age consequence proxies added only +0.001 PR-AUC (4/8 episodes
better), so they are not promoted to the actor. This is selected-action
prediction, not a return improvement or counterfactual host ranking.

Full methods, per-partition counts, artifact hashes and limitations:
[`blue-inspection-yield-20261013.md`](../research/blue-inspection-yield-20261013.md).
The next Blue task is a broader, pre-registered training-data policy to
observe more legal investigation choices, followed by a paired *policy*
comparison. No long Kaggle run, final-seed evaluation, shared contract
change or dependent branch migration occurred. Focused tests and live
episode/file-integrity checks passed. PR review should focus on the action
request/completion join, label isolation, sweep-only fit filter, whole-
episode partition guard, and noncausal interpretation of the consequence
proxies.

## Next session: read `blue-handoff-20261004.md` first

[`blue-handoff-20261004.md`](blue-handoff-20261004.md) is the authoritative
starting point for the next session. It carries the verified matrix numbers, the
six claim corrections two external reviews forced, the settled statistical design
(power formula, decision rules, freeze order, five named controls), the P0-P7
phase list with rationale, the cross-role blockers, and a copy-pasteable resume
prompt. `blue-session.md` remains the long-form session log; the handoff
supersedes it wherever they disagree about the max-age guard conclusion.

## Research audit, 2026-10-03

On `blue/mappo-training` at `a6eab7e` (main dependency `6f1d1ca`),
[`blue-rl-feasibility-audit-20261003.md`](../research/blue-rl-feasibility-audit-20261003.md)
reviews primary sources against the current Blue code and experiment records.
It identifies an invalid FQE regression (logged actions ignored), an unused IQL
target Q, lost final offline reward, extraction/deployment mask mismatch, and
linear actor context that cancels in same-command host comparisons. The latest
FQE note's value-indistinguishability conclusion is retracted pending a valid
evaluation method. No new policy result or simulator check is claimed.
Next: audit/fix the offline data and FQE equations, then compare episode-valid
extraction and contextual host scoring on development seeds before more RL.
This research document changes no shared schema or runtime contract and needs
no dependent branch migration. Exact tests run: documentation `git diff --check`;
training/simulator tests skipped because this handoff changes no executable
code. PR review should verify paper links and algebra, and keep 8201+ final
seeds untouched.

Objective: a CAGE-4 Blue defender that beats the round-robin teacher
(-93.5 ± 27.8, 400-step native episodes) and holds up off-distribution.
Branch `blue/mappo-training` (pushed, tracks origin). Details live in
`docs/proposals/` (15 records + manifests), `docs/research/`, and the
chronological `blue-session.md`. This page is the short index — update it
at each meaningful handoff. (Prior foundation-era content superseded;
see git history.)

## What was tried, in order

1. **Literature** (`docs/research/`): sparse-reward/forgetting review
   (warmup tested: null), then architecture review — official CAGE-4
   analysis shows the podium was all sweep-and-remediate heuristics.
2. **Hybrid rules + ordering** (proposal 01, CLOSED): parity proven
   byte-identical; lancer constants hit -63.2 regression but failed
   held-out (-110.1 ± 130.4, incl. a -414 seed); v2 regularized to parity;
   snapshot risk scorer (AUC 0.67) collapsed coverage (-191); risk×recency
   hybrids failed the held-out gate. Fixed ordering cannot win robustly.
3. **Factorized actor + distill ladder** (proposals 02/10/11): host-then-
   command head (zero vendor edits) distills the teacher at 21.8% vs 12.2%
   flat; aux host-loss hurts; 1-layer slot attention reaches **teacher
   parity on held-out with no RL** (-93.9 ± 45.8 vs -85.1 ± 46.1).
   Every MAPPO fine-tune diverges (flat and factorized); KL-to-teacher
   (recorded vendor patch) only slows it. Fine-tune line closed absent a
   better distill.
4. **Generalization suite** (proposal 15, standing): `red_agent=` flag +
   pool support; finite red much harsher than discovery (-139 vs -59);
   lancer_v2 ≈ teacher on every red. Green rates/durations need
   Environment hooks (`docs/coordination/blue-red-variants.md`, draft).

## Standings (8-seed native means; manifests in `docs/proposals/manifests/`)

| policy | regression | held-out | note |
|---|---|---|---|
| round-robin (teacher) | -93.5 ± 27.8 | -85.1 ± 46.1 | the bar |
| hybrid_lancer_v2 | -75.2 ± 33.6 | -91.2 ± 48.5 | best fixed; no transfer edge |
| attn-1 distill (learned) | -85.5 ± 31.1 | -93.9 ± 45.8 | first learned parity, no RL |
| flat BC | -393.4 ± 276.9 | — | learned start point |
| MAPPO fine-tunes | diverge (-500 → -1710) | — | closed |
| risk / risk×recency | -191…-251 | — | closed (coverage collapse) |

## Metric-semantics repair + max-age guard, 2026-10-04

On `blue/mappo-training` at `2900711` (uncommitted at time of writing).
**No training run in this session.** The Phase 2/5 harness
(`blue/analysis/ordering.py`) was measuring requested instead of completed
analyses, kept only the first analysis per host, let failed requests refresh
ages, counted non-defendable `contractor_network` hosts, and reported the
final-tick age as `max_age` (constant 398). Replaced by
`blue/analysis/metrics.py` + `blue/analysis/recorder.py` (30 tests, including
live cross-checks against the wrapper's own tracker fields).

Consequences worth knowing before reading any older number: the defendable
universe is **seed-dependent, 57–96 hosts (mean 76.3)** — per-agent splits
vary by seed (e.g. agent 4 holds 21–39, not a fixed 26); any fixed "66
hosts (5/12/14/9/26)" denominator was one seed's layout (7701);
analysis failures are **0 in 32 seeds**; ~1.97 investigations per episode are
unresolved at the episode end (one pending request per agent). The unguarded
lancer arm re-runs **32/32 returns exactly equal** to the historical manifest,
so the identity claim holds on the repaired path. Design finding: unguarded
lancer coverage was already **1.000** (0 never-investigated in 32/32 seeds),
so the guard bought freshness, not coverage.

`MaxAgeGuard` added with explicit guard modes; threshold **A = 48**
pre-registered from the baseline age profile only, plus 24/96 sensitivity
arms. `A` is an intervention threshold, not a guaranteed maximum — overshoot
is reported, not hidden. Invalidated evidence is listed in
`docs/proposals/manifests/guard-maxage-20261004.json`: the ordering/guard
metric columns and the -83.0 residual pilot (preserved as history, not
evidence).

**Outcome (32 seeds x 8 arms, `guard-maxage-matrix-20261004.json`, corrected
2026-10-05 per `blue-handoff-20261004.md` §3): no detectable guard benefit
at this resolution.** Means of per-episode maxima fall 274.3 -> 133.3–179.1
(global worst 370 -> 186–324) with coverage 1.00 (the mechanism acts), but
return is 1.3–6.3 *worse* than unguarded at every threshold, monotonically
so in how hard the guard binds, and `maxage48` is indistinguishable from
the strict coverage guard (-0.22 CI[-6.78,+6.53]). With n=32 the design
resolves only ~7–16 per contrast, so "stop" means no detectable benefit,
not a proven zero effect. The tested privileged scorer improves unguarded
return (+16.22 CI[+6.66,+25.56]) but is **not an upper bound** on ordering
choices (greedy onset scorer ignoring cost/timing; its own mean-of-maxima
age is worse, 371.0 vs 274.3); its 0.284 coverage is a mean 54.97
never-investigated hosts of the seed-dependent 57–96 universe — an
association with the gain, not a demonstrated cause — and inside a coverage
guard its edge over lancer vanishes (strict -3.12, max-age -1.97, upper
bounds +6.00/+7.22 not excluding +5). The raw 256 cells are committed as
`guard-maxage-matrix-20261004-cells.jsonl`. The bounded **unguarded**
residual/MAPPO pilot therefore proceeds (handoff P4) instead of the
guarded pilot the original gate blocked.

Audit outcome: the residual/PPO path is green (16 torch-gated tests) and the
one real defect it found is now fixed — `residual_pilot cmd_eval` contrasted
learned-vs-`lancer`, which changes the guard and the learner at once. The
**primary deployment comparison is `learned − lancer`** (replacing the
incumbent is a total-system question); the scheduler control
(`learned − sched_control`, identical guard) is the secondary attribution
contrast, not the headline. The legacy `analysis/ordering.py` entry point now
withholds its four known-invalid metric columns by default
(`--emit-invalid-metrics` to reproduce them), so the superseded numbers
cannot be re-published by accident.

Validation: sim venv 178 passed / 11 skipped, train venv 227 passed /
1 skipped (pre-existing rvs checkpoint skip). The seed ledger conflict (`blue.md` said `8201+` reserved
although 8201-8216 are consumed; `blue-session.md` claimed 8200-8420 free
although `8301-8400` belong to finite Red) is reported and deliberately
untouched — `8301-8400`, `8401+` and `8501+` stay clear.

## Open threads / next

- KL-anchored PPO from the attention ckpt is unlocked (stop-rules apply).
  Seed blocks: development seeds 7629-7729 are consumed; `8301-8400`
  (finite Red), `8401+` (ledger conflict) and `8501+` (reserved) are
  untouched. The "8201+ reserved" claim in earlier notes is stale: 8201-8216
  are already consumed.
- Coordination proposal awaits Environment review (green rates, durations).
- Infra: process-pool eval + manifests; 2 vendored-learner patches recorded
  in `environment/patches/` (third_party itself is gitignored).
- Suites: sim 112 passed / 7 skipped; train 22+ passed. No privileged state
  in any actor/mask/scorer (audited); native reward is the only score.
