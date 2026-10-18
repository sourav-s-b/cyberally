# GPT Blue harness handoff

Branch: `gpt-blue`. Isolated checkout:
`/home/sourav/Projects/cyberally/worktrees/gpt-blue`.
Base dependency: `243a6fd97dc29d59c7bfc677209d363412f6cc10`.
Harness published at 15b76448795b9568cc629e824fb7ddee4e5368ab; private Kaggle
pilot completed. No merge, teammate message or external deployment performed.
Latest measured audit: [blue-gpt-pilot-audit.md](blue-gpt-pilot-audit.md).

The shared checkout remains on `blue/metric-repair-maxage`. OpenCode's
`risk_actor.py`, `guard_check.py` and existing `trace_episode.py` were left
untouched. Only GPT-owned harness code was relocated. Ignored runtime and
artifact storage are symlinked into this worktree; no model weights are staged.

The older `docs/current-state.md` and session headers describe historical
branches and missing-policy status. They do not describe this new local
prototype. The implementation, saved artifacts and checks below are current
evidence; older literature diagnoses are not treated as established causes.

## Delivered

- Blue-only feature contract, explicit telemetry and per-value missingness,
  freshness/completed-action ages, pending status, observation-clock validation,
  idempotent deltas, public phase and host-role context.
- Calibrated histogram gradient boosting and benign-reference Isolation Forest
  in a versioned/hash-checked bundle. A separate logistic baseline uses the same
  new features; no claim that any algorithm must improve defender returns.
- Existing urgent/verification priorities plus completed-analysis age protection,
  independently of ML. No risk-based legality masks or V(s)-based action gate.
- New PPO collector/trainer and loader; same harness in training and execution.
  Both ML inputs can be zeroed for the matched control. New actor geometry
  requires retraining; legacy models are preserved for coverage-only diagnosis.
- Diagnostic evaluator with full-episode requests, host targets, busy/pending
  status, completion/result and input snapshots. Privileged session labels and
  per-host native penalty events are stored separately; each event sum is
  checked against native reward. Failed ground-truth access never becomes a
  fictitious zero-compromise result.

See `blue/harness/README.md` for executable commands and
`docs/coordination/gpt-blue-harness.md` for the local contract proposal.

## Measured checks and artifacts

Artifacts are ignored under `blue/results/gpt_harness/`.

Detector dataset: eight complete 399-tick development episodes, all from
previously consumed seed lists. TRAIN 7706–7709; CALIBRATION 7701/7702;
TEST 7703/7704. Final seeds 7809–8200 are explicitly rejected. Existing risk
datasets collected on 7901–7908 were not reused. Labels reside in
`data.labels.npz`, separate from `data.npz` features/episode metadata.

Dataset SHA-256:
`a1d1d340249b08ec545c70724ad045b1ec8b8078ed9f1ee456577348e69a7f5c`.
Final scorer (`scorer_v1.pkl`) SHA-256:
`949048797f2b89a5703189c53fb9062a82f056989da931d6c56b177af5073b2a`.
`scorer_v1.json` contains settings, actual split lists, runtime and source hashes.

| Development detector metric | HGB | Logistic, same features |
|---|---:|---:|
| PR-AUC | 0.74734 | 0.65489 |
| ROC-AUC | 0.91603 | 0.86921 |
| Brier error | 0.10196 | 0.12143 |
| Log loss | 0.34645 | 0.36759 |

These 61,073 evaluated host-ticks are NOT independent experiments: only two
test episodes exist. Per-episode metrics are saved. HGB's log loss on 7704
is worse than logistic's; do not claim uniform calibration superiority.
Novelty PR-AUC is 0.28821 against prevalence 0.25167. The novelty heuristic
bonus defaults to zero; novelty remains an experimental actor input.

Mechanism report `mechanism/report.json` reproduced historical failures:

| Arm | Seed 8241 | Seed 8229 |
|---|---:|---:|
| Lancer, unguarded | -64 | -112 |
| Original real-risk actor, unchanged | -1273 | -1034 |
| Same actor, age threshold 80 | -66 | -113 |
| Coverage-only Lancer heuristic | -59 | -107 |
| ML-assisted heuristic, same coverage | -78 | -128 |

The last two rows are in `ml_diagnostic/report.json`. These mechanism checks
are neither fresh evaluation nor a general performance claim. The ML heuristic
is worse than coverage-only on both episodes: better detection metrics have
not established better defense.

On 8241, original actor agent 3 analysed its operational-zone-B router 162
times and completed no positive Analyse there on any controlled host. Native
penalties comprised -373 Red Impact and -872 failed Green local work, plus
-28 action cost. With the age guard, those totals became -6/-6/-54. The first
Red Impact penalty is at tick 150 (phase 1, -1); the phase-2 weights amplify
later failures. Thus the large late penalties need no invented missing reward.
The guard alters investigation scheduling, leaving forced remediation intact.
This supports the starvation mechanism on these diagnostic episodes.

Live PPO smoke: one update from two 100-step episodes, 341 trainable decisions,
mean return -10. Checkpoint reload/evaluation on 7703 at the same 100-step
horizon completed with return -12. These are plumbing checks, not policy
performance results. Artifacts: `rl_smoke/` and `rl_eval_smoke/`.

Focused initial suite: 47 passed (new harness + existing age guard + risk
ablation). After adding a recorder test, updated harness suite: 10 passed.
The new test checks executed-action log probability, sampled-versus-greedy
metrics, and the discounted return/padding contract. Compile checks and
`git diff --check` passed. Final isolation checks are recorded below.

Final isolated-worktree focused suite: **48 passed**, one existing upstream
`pkg_resources` deprecation warning, in 75.27 seconds. Command:
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=blue/results/harness_runtime .venv-train/bin/python -m pytest blue/tests_blue/test_harness.py blue/tests_blue/test_max_age_guard.py blue/tests_blue/test_risk_ablation.py -q`.
Compile and owned-file whitespace checks passed. Repeated isolated mechanism
evaluation uses frozen source/artifact hashes under `isolated_mechanism/`;
guarded legacy returns again matched -66/-113.

## Risks and next concrete task

Keep the optional ML dependency proposal unmerged until Environment review;
Evaluation should review grouped statistics and diagnostic event accounting.
No upstream simulator or shared telemetry schema was changed. The artifacts
are trusted local pickles and are tied to pinned runtime/feature geometry.

No long training or independent multi-seed RL superiority test has been run.
The critic remains the existing Blue-visible pooled V(s), and gamma remains
0.99. Four repeated training environments and stochastic-versus-greedy
execution remain research limitations. Coverage can overshoot its threshold
while agents are busy or executing urgent rules. Additional input features
and scores cannot reveal hidden compromise without observable evidence.

Next: freeze a bounded matched PPO comparison on this harness (`both` versus
`zero` ML inputs), report every training seed, and compare against coverage-only
and Lancer. Review development seed ownership before enlarging suites; no
untouched-seed claim based on the old incomplete seed-audit script. Keep native
mean return primary and tail failures in the estimate. No scoring changes or
best-run selection. The current ML heuristic loss makes this ablation necessary,
not optional evidence to omit.

PR checklist: only GPT-owned files, source/artifact manifests, no tracked
weights/logs/runtime, correct local branch/worktree, preserved OpenCode work,
tests actually run, versioned checkpoint migration, affected-role review pending.

Pilot preparation update (2026-10-06): added risk-only actor control alongside
zero/both; added atomic iteration-boundary checkpoint (actor, critic, optimizer,
Python/NumPy/Torch RNG and history) with strict resume source/config/scorer
validation. All 11 harness tests passed. Interrupted 2-iteration smoke resumed
successfully from iteration 1 to iteration 2; uninterrupted equivalence check
is running. Kaggle authenticated dataset listing succeeded. Separate draft
runner is ops/kaggle/harness/run.py; private source/scorer package is ignored
blue/results/gpt_harness/kaggle_pilot_package. Nothing uploaded or launched.
Remaining before submission: runtime compatibility/golden scorer check,
common evaluation integration and complete package regeneration after final
runner edits. Do not treat this draft as a submitted or validated remote run.

Pilot validation: 12 tests passed, including configuration-keyed baseline caching
and all six matched model rows. Recovery matched uninterrupted actor/critic
weights and history exactly. Golden scorer and simulator preflight passed both
in the worktree and extracted source snapshot. User requested GitHub code
transport: runner now checks out an exact commit; weights stay in private
Kaggle input. Remote runtime preflight remains mandatory before training.

Kaggle submission (2026-10-06): user approved GitHub source transport. Harness
published on gpt-blue at a5c91393497beaccee38ff8e196f5181da1aa736.
Private dataset: souravsreekumar02/gpt-blue-harness-pilot-20261006 contains only
scorer.pkl, golden.npz and pilot.json. Separate private kernel
souravsreekumar02/gpt-blue-harness-pilot version 1 accepted; API reports RUNNING.
Startup preflight checks pinned Python/runtime, ML reference outputs and native
simulator episode before training. RUNNING is not evidence that training or
preflight completed. Six final models: zero/risk/both x RNG seeds 0/1, 8x4x400;
common reused dev episodes 8221–8228, baselines Lancer/coverage-only. All models
reported, no significance claim with two training seeds. Final 7809–8200 unused.
Next: retrieve startup/output logs, verify preflight and all six completion
manifests, then inspect pilot/report.json before authorizing a larger experiment.

Runtime repair: Kaggle version 1 failed before training because the isolated
runtime omitted gymnasium, imported by upstream RandomAgent. Added pinned
Gymnasium 0.28.1 and moved runtime dependencies into a single requirements.txt.
Fresh /tmp/gpt-blue-clean-runtime (no inherited site packages) passed golden
scorer + native simulator preflight and evaluation imports. Experiment/model/
seed/budget/reward settings unchanged; relaunch uses a new source commit only.

Audit update (2026-10-07): completed pilot manifests/weights/cells verified;
all 12 frozen-model diagnostic replays reproduced evaluation returns. Optional
request hashes and same-state ML sensitivity added. Eight-episode detector
validation preserves training/calibration arrays exactly; HGB PR-AUC wins 8/8
with episode macro 0.72794 vs 0.63107. Older NumPy/Torch candidate validated
after separately rebuilding the scorer; stochastic training differs between
stacks, so no runtime equivalence claim. Actual local pilot stack already
matched Kaggle. See the audit for all tables, tests and limitations. Larger
training remains unlaunched; next is a frozen, sharded configuration review.

Main32 preparation (2026-10-07): user authorized the next frozen/sharded run.
Design and measured submission state in [blue-gpt-main32.md](blue-gpt-main32.md).
Five independent training RNG seeds, three arms each, 32x4x400, common 32 reused
development episodes, four baselines. All runs new under one older frozen stack.
Primary BOTH must clear Lancer AND guard; other comparisons descriptive. New
cohort aggregation rejects incomplete or mixed shards and uses n=5, df=4, never
160 model replicas. Clean full-lock runtime preflight and 18 tests passed.

Main32 cohort submitted: five private kernel version 1 jobs, actual IDs
`gpt-blue-harness-main-seed-0` through `-4`, all checked RUNNING.
Experiment source pin `6942d584f15231a7f19add0a510a04f7ff3ad229`;
see blue-gpt-main32.md for cohort hash, controls and next retrieval steps.

## 2026-10-07: design audit and opt-in v2

Branch gpt-blue, isolated worktree. Research and component audit:
`docs/research/blue-harness-redesign-20261007.md`. Draft interface review:
`docs/coordination/gpt-blue-harness-v2.md`. No shared contract, simulator,
reward or existing checkpoint migration changed.

Implemented persistent Blue-visible event memory, masked candidate-set
actor context, all-host centralized critic summaries, separate optimizers,
decision-time GAE/MC, exact teacher-mixture behavior, explicit MAPPO and
A2C paths, nested stochastic evaluation and an untrained-mixture control.
Weighted HGB/sigmoid detector candidate retains per-episode validation;
its defensive value remains unproven. Novelty defaults off.

Validation: both learners completed two full-horizon training iterations
with two episodes each. MAPPO full-horizon evaluation completed on reused
8241. Short-horizon resume produced exactly matching actor weights against
uninterrupted training; all evaluation/control paths completed. Focused
harness tests run under Python3.11.16/NumPy1.26.4/Torch2.2.0 CPU lock.
These are smoke/mechanism checks, not performance claims.

Blocker for scaling: downloaded main32 risk_s1 reproduces -97 locally on
8236 versus remote -225; guard matches -81. Plain runner agrees with trace,
so logging alone is not the explanation. risk_s2 also differs on 8233.
Trace audit now marks remote reproduction failures explicitly. Next task:
resolve full policy/request replay, then isolated optimization ablation and
matched MAPPO/A2C screen. Draft screen is not launchable; no Kaggle run
started. Reserved 7809–8200 preserved. Origin main advanced to 01140fe,
inspected but not merged; no external Red LLM experiment introduced.

Follow-up: established local numerical ranking sensitivity. Same risk_s1
on8236 gives -136/thread1 versus -97/thread2. At tick2, identical inputs
have a score difference7.45e-9 that changes the host tie-break. Explicit
1e-6 resolution yields -152 and identical request hashes for threads1/2:
a stability mechanism check, not a performance fix. V2 pins one thread;
frozen v1 untouched. Preparing a bounded private remote replay with frozen
weights, no training, to resolve cross-machine behavior.

Private replay submitted after explicit user approval of checkpoint/scorer
upload: Kaggle `souravsreekumar02/gpt-blue-frozen-numerical-replay-audit`,
source cdd17647fc6f72cd374801ccf3a05ea379e63749, private dataset
`souravsreekumar02/gpt-blue-replay-audit-20261007` version2. Eight
frozen-checkpoint episodes (8233/8236, thread1/2, exact/1e-6 near-tie
ranking), no training, no reserved block. Dataset uploader initially
skipped nested checkpoint folders; fixed by flat filenames and bootstrap
reconstruction before launch. WSL shutdown interrupted local references;
completed artifacts preserved and missing local episodes resumed.

Evaluation counting now uses a read-only hook recorder so learned choices
are not mislabeled as forced by the legacy runner. Focused tests:28passed
under locked legacy runtime. Source remains isolated on gpt-blue.

Numerical replay completed:all eight remote jobs retrieved and validated.
Near-tie policy matches local return and full request digest across both
thread counts on8233/8236. Original exact ranking remains platform-sensitive;
Kaggle thread2 reproduces old -225 on8236. Stability intervention preserves
-205 on8233 and gives -152 on8236, not a defense improvement.30focused
tests passed; resume exact and real hook counts verified after restart.
Preparing four-model full-horizon MAPPO/A2C pipeline check, two iterations,
RNG0/1, original approved scorer, reused8233. No algorithm ranking claim.

User steering:run small training locally, Kaggle reserved for long runs.
The four-model pipeline is moved to `harness_redesign_smoke` on the local
locked runtime. No v2 Kaggle training kernel was launched; the remote
smoke bootstrap is a future template only.

## Local check completed after user steering

Four models completed locally (MAPPO/A2C, training RNG0/1, two iterations,
two episodes per iteration,400-step configuration, original risk scorer).
Exact experiment source:8a5da51b2c7649283f60a4269540671072c27748.
Artifacts:`blue/results/gpt_harness/v2_local_smoke/cohort/analysis.json`.
All four sampled policies return -49,-68 on reused8233, exactly matching
the untrained-mixture control. No learning benefit or algorithm ranking
established. Lancer -110, guard -97, Sleep -2901, random -203.
Greedy diagnostic results are reported separately, never selected as
primary. Seed0 request hashes distinguish MAPPO's changed first sequence
with equal return from A2C's unchanged sequences.

Validation:30focused tests passed; full-horizon local training/evaluation
completed; same-source/runtime/model/scorer hashes verified; resume actor
weights matched exactly. Numerical audit on Kaggle completed, with full
local/remote hashes matching under declared near-tie resolution on both
diagnostic episodes. No long training started. Original final seeds
7809–8200 untouched.

Fair ML comparison on eight reused episodes:original HGB macro PR-AUC
0.722771, weighted candidate0.741667; difference0.018896, conditional
episode interval[0.01043,0.02736], one fit per variant. Prediction gain
does not establish defense gain.

Next concrete task:local one-factor discount/exploration sensitivity
check with more development episodes and untrained controls, before
freezing a longer multi-seed Kaggle algorithm comparison.

PR checklist:Blue-only additive implementation; existing checkpoints and
simulator unchanged; tests and fixed artifact manifests recorded; no
weights/logs/runtime committed; shared evaluation protocol proposal
remains draft pending affected-role review before merge. Remote replay
dependency is cdd17647fc6f72cd374801ccf3a05ea379e63749; origin main
01140fe inspected but not merged. Known risks:reused suites, few calibration
episodes, bundled design changes, discount/native objective mismatch and
stochastic exploration ceiling; no deployment or performance claim.

## Local sensitivity study started

Started additive `blue.training.harness_sensitivity` on the locked local
runtime, branch gpt-blue, base commit bda8da97b434397295ec6cf58fb04b970f69aabd.
Runner bytes and runtime are frozen in
`blue/results/gpt_harness/v2_sensitivity/plan.json`; runner is currently
uncommitted. Six MAPPO models: reference gamma .99/exploration .2,
gamma 1/exploration .2, gamma .99/exploration .5; training RNG 0/1,
four iterations with two episodes each. Original ML scorer unchanged.
Evaluation uses reused episodes 8233/8236/8241/8229, two nested policy RNG
replicas and matching untrained-mixture controls, recording request hashes.
No result yet; no longer Kaggle training launched. Final seeds protected.
Compilation passed. No core learner changes; prior tests are not claimed
as a new run. Do not change HEAD or frozen source files during this study.
Resume runner from the same worktree/runtime; completed models and eval
cells are retained and incomplete training resumes from checkpoint.pt.

## Sensitivity study completed

All six local MAPPO models and 64 evaluation cells completed successfully.
Four iterations, two training episodes per iteration; four reused development
episodes and two nested policy RNG replicas. Frozen plan and full cells:
`blue/results/gpt_harness/v2_sensitivity/`.
Mean native return gain against each matching untrained mixture, training
RNG 0 / 1: reference +0.250 / +0.125; undiscounted +0.375 / -0.625;
higher exploration -1.625 / -5.125. Across the two training seeds these
average +0.1875, -0.125 and -3.375 respectively. No superiority or convergence
claim. Discounting and exploration changes do not establish a learning gain
at this budget. Action-request hashes changed in 8/8 and 3/8 reference
replicas, 8/8 and 4/8 undiscounted replicas, 8/8 and 6/8 higher-exploration
replicas: equal returns must not be reported as unchanged trajectories.
Next: inspect policy update magnitude and decision changes before increasing
training budget; retain original scorer and native metric. Two training
seeds remain the independent replication unit, not 64 evaluation cells.

## Read-only update and single-request audit completed

See docs/research/blue-v2-update-audit-20261007.md. Six checkpoint parameter
comparisons confirm weights changed; clipping/KL stop were inactive. Two
instrumented frozen evaluations reproduce return and all 1,995 request hashes.
Average total variation from zero residual is .001779 / .001620; direct ML
risk+entropy zeroing changes distributions by only .00004334 / .00002676.
These are same-state diagnostic shadows, not action-value estimates.

For higher-exploration s1, episode 8233, policy RNG 8208302: original -122,
untrained -61. Replacing only tick 48 blue_agent_4's requested investigation
from admin router to office user host 1 returns -64. Same fixed checkpoint,
normal RNG draw consumed, later decisions allowed to diverge. One diagnostic
intervention, no superiority claim. All checkpoint hashes unchanged.
Next concrete task: trace completion/Blue observations/reward events around
that tick-48 host choice to explain why its downstream consequence is large;
do not deploy a host-specific rule or increase training based on this episode.
Scripts compile and diff check passed; core tests not rerun (core unchanged).
Branch gpt-blue; additive scripts/docs currently uncommitted. Final seeds
7809–8200 untouched; no new Kaggle training or external upload.

## Full decision timeline completed

Read docs/research/blue-v2-decision-timeline-20261007.md. All three full
replays reproduce frozen returns/digests (-122/-64/-61), with per-tick native
penalty accounting. The office swap is not an early compromise discovery:
both targets lack Red-session labels and visible files at tick 48. Office
is already the trained policy's 53% favorite; router is a ~3% sampled pick.
Modified sequence repairs a different admin host two ticks later.

Shared controller/state/Red/Green RNG identities verified. Two 80-tick
prefixes match saved requests; policy does not consume simulator RNG, but
simulator RNG states split after tick 63 and NPC actions differ at 64.
Thus the 58-point result cannot be attributed solely to better prioritization.
Original -122 = -70 action costs and -52 attack/service events; modified
-64 is action costs only. One scored operational user host has a 98-tick
completed investigation gap (38 to 136), with failures starting at request
112. This is a stronger next target than an office-host-specific rule.

Next: inspect visible evidence and busy/urgent capacity on operational A
user host 0 in that gap, and native penalty controllability. No algorithm
change or long training until that mechanism is established. Model/scorer
unchanged; privileged labels/logs separated from actor inputs; final seeds
protected. Two focused logging tests pass; native reward/request/RNG checks
complete; compile/diff checks pass. Additive scripts/docs remain uncommitted
on gpt-blue. No external upload or simulator changes.

## Operational A host gap follow-up

`harness_host_gap_audit.py` records every tick's Blue view, observation
freshness, tracker state, pending/busy status, requests/completions and native
penalties; privileged Red-session and event traces are separate. Full replay
matches frozen -122 return and all 1,995 requests. For operational A user 0,
Blue completed an empty Analyse at38, then no new Files observation until136.
The Process view went from2 to12 entries at the first diagnostic Red-session
tick; a new SSH connection targets neighboring operational A user 1.

At decision38 the approved risk model scores the host1.0, but actor selection
probability is only .07434 versus .07295 with actor residual zero, while the
teacher router has .606. A single frozen intervention immediately analyses
user0: completion at40 exposes cmd.sh and CONFIRMED; the same actor requests
Remove at40. Original next analyzes it at134, completing136 with cmd.sh and
escalate.sh. This demonstrates potentially actionable early evidence and
underuse of risk at this one state. It does not prove episode return benefit,
clean removal, or generalization. Partial prefix rewards are not interpreted.

Next: reuse diagnostic episodes to tabulate high-risk signal -> detection ->
Remove completion/verification and per-episode native event outcomes before
proposing a protected-priority rule or retraining. Do not infer that red
session labels always imply detectable compromise. No training, Kaggle or
simulator changes. Frozen checkpoint hash preserved. Scripts compile by
execution; host-gap trace and 42-tick prefix request checks pass. Final seeds
7809-8200 untouched.

## High-risk sweep audit on reused development episodes

On branch gpt-blue, two frozen reference MAPPO checkpoints were replayed
locally on development episodes 8233, 8236, 8241 and 8229, with sampled-policy
RNG `episode*997`. All eight native returns, 399-step reward ledgers, and full
1,995-request hashes matched the saved cells. Each sweep decision records the
real risk score, teacher and sampled choice, and Blue-visible host state;
completion and tracker changes are in Blue JSONL, while Red-session labels
and penalty events are in separate diagnostic JSONL. Threshold >=0.8 was
frozen before the replays. See
`docs/research/blue-v2-high-risk-audit-20261007.md` and ignored artifacts
under `blue/results/gpt_harness/v2_sensitivity/high_risk_audit/`.

Across the four episodes, high-risk candidates appeared in 112–252 sweep
decisions per cell and were selected in 32–48. In each cell 11–17 distinct
hosts were skipped at their first high-risk opportunity, then Blue-confirmed
after a wait of >=20 ticks. Lancer's teacher selected a high-risk candidate
about as often as the trained sampler. This is evidence of delayed
investigation, not proof that earlier Analyse would detect or improve native
return. Non-cost losses in these episodes occurred on contractor-network
hosts and did not overlap the Blue-owned high-risk candidates; the other
episodes had action costs only. No trained-policy change or new training.

The original audit process was interrupted after two validated cells.
Remaining cells were resumed under the same frozen replay code; partial
JSONL files were retained with `.interrupted` suffixes. The new scripts
compiled and the final diff check passed. Next: one single-request local
mechanism check at a first high-risk opportunity with a later confirmation,
recording detection, action cost, penalties, and simulator RNG divergence.

## Finite-Red native-reward comparison completed

On 2026-10-15 the six native-reward finite-Red models (training RNG seeds
0-5, 32 iterations each) were evaluated against the same finite-Red Lancer
cache on seeds 7665-7688. Sleep and masked-random were also evaluated on that
block. All six manifests were complete; actor hashes, config, attacker,
objective, and source/scorer hashes were checked during the run. All six actor
reports contain the full paired seed list.

The paired mean difference by training seed was -45.42, -13.96, -10.08,
-22.13, -25.17 and -68.96. Their mean is -30.95 native reward; the 95%
training-seed t interval is [-54.37, -7.54], entirely below zero. On this
fixed evaluation block Blue underperforms Lancer. This is a six-seed
exploratory result; do not claim an RL improvement or proceed to LLM Red from
this cohort. Per-model and per-episode returns, paired differences and tail
losses are in ignored `blue/results/finite_red_native/analysis.json`; the
initial three-seed report is preserved as `analysis_3_seed.json`.

The original `gpt-blue` worktree's unrelated research edits were retained.
The temporary training source clone was removed after the run; manifests keep
the source digests, and they were checked against that source tree before its
removal. The ignored result artifacts remain available for audit.
