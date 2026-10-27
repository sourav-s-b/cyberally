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

## Finite-Red native-reward gate, 2026-10-15

Six finite-Red models (training RNG seeds 0-5; 32 iterations x 4 rollouts)
completed under native reward with shaping off. Lancer, Sleep and masked-random
controls and all actors were evaluated on the shared 24-seed block 7665-7688.
All manifests, actor/config/scorer metadata and paired report seed lists were
checked. The source digests were verified against the temporary run source tree
before it was removed.

Mean actor-minus-Lancer native return by training seed: -45.42, -13.96,
-10.08, -22.13, -25.17, -68.96. The six-seed mean is -30.95; the training-seed
95% t interval is [-54.37, -7.54], entirely below zero. This exploratory
cohort underperforms Lancer on this fixed evaluation block. Do not claim an RL
improvement or proceed to LLM Red on this evidence. Full episode returns,
paired differences and tail losses are in ignored
`blue/results/finite_red_native/analysis.json`; the earlier three-seed result
is preserved in `analysis_3_seed.json`.

Implementation is on isolated branch `blue/finite-red-check`, based on
`2f44db1`; 12 focused tests pass, including a real child-process concurrency
bound and resume drift checks. The original `gpt-blue` worktree and the current
checkout's dirty research files were preserved. The exact temporary source
snapshot used for training was removed between sessions; restored implementation
files in this branch are reviewable but have different source hashes, so this
branch is not represented as a byte-for-byte reproduction of the six models.
The initial finite-Red gap diagnosis is recorded below. No commit or push made;
no teammate notified.

Finite-Red gap diagnosis (2026-10-15): added an ignored same-seed guard-only
control on all 24 evaluation seeds. Guard-only mean return was -150.21 versus
Lancer -151.79; the age-80 guard alone does not explain the learned actors'
loss. Actor-minus-guard mean paired difference across six training seeds was
-32.53 (training-seed 95% t interval [-55.95, -9.12]). The main associated
penalty difference was Green Local Work failure, -22.33 native points/episode;
host exposure was similar. The excess LWF was concentrated in mission phase 1
(-16.35 points/episode). This is descriptive, not a causal attribution.
The diagnostic report is
[`finite-red-native-gap-diagnosis-20261015.md`](../research/finite-red-native-gap-diagnosis-20261015.md);
reproduction code is `blue.training.harness_finite_diagnose`. Guard traces and
summary JSON are ignored under `blue/results/finite_red_native_diagnosis/`.
The control used the restored source at base `2f44db1`: wrapper, policy and
ordered-policy hashes match the actor reports, while evaluator/training hashes
do not. The original temporary training source remains unavailable, so the
comparison is not byte-exact. Follow-up joined phase-1 LWF events to subnet and
same-host Blue actions: Operational Zone A accounts for -13.06 points/episode
actor-minus-guard. There were 2.51 such LWF events/episode for actors vs 1.21
for guard-only; 1.75 actor events/episode followed an `Analyse / UNKNOWN` more
than 20 ticks earlier (guard-only: 0.88). This association does not establish
cause. No new training was launched. Next: review the candidate audit below,
then freeze a matched follow-up protocol before another cohort.

## Candidate-choice audit complete, 2026-10-15

Added opt-in Blue-visible candidate instrumentation to `HarnessPolicy` and the
actor hook, plus `blue.training.harness_finite_candidate_audit` for a separate
post-hoc join to privileged event traces. Replayed the six frozen actors on
seeds 7665-7688. All 144 returns match their stored reports exactly; all
Blue-agent action/host decisions match across 57,456 ticks. Mean Lancer rank of
the actor's selected candidate was 6.93 across actor-ranked decisions, and the
actor disagreed with Lancer 86.2% of the time. For phase-1 Operational Zone A
LWF events, the actor's latest candidate pick was the event host 23/362 times,
while Lancer's same-state candidate pick was that host 55/362 times. The
direction varies by training seed, and divergent trajectories prevent causal
attribution. Full counts and limits are in
[`finite-red-native-gap-diagnosis-20261015.md`](../research/finite-red-native-gap-diagnosis-20261015.md);
ignored candidate reports/traces and `summary.json` are under
`blue/results/finite_red_candidate_audit/`.

Validation: one-seed smoke retained s0 seed-7665 return (-101); `py_compile`
and `git diff --check` passed; complete action-trace comparison found zero
mismatches. No training or commit/push. Current branch: `blue/finite-red-check`
at base `2f44db1`. Next concrete task: review the candidate audit and decide
whether to preregister a finite-Red follow-up with a specific residual-control
change; the existing learner still fails the Lancer gate, so LLM Red remains
deferred.

## Bounded anchored PPO pilot, 2026-10-15

Completed the approved three-seed, eight-iteration finite-Red development pilot
on native reward, comparing current residual PPO with a Lancer-anchored
categorical PPO actor. Both arms trained from fresh rollouts on seeds 7706–7709;
all six manifests are complete. Fresh Lancer and guard-only controls and all
12 model/mode reports completed on the matched 24-seed evaluation block
7665–7688. The exact source snapshot, model/scorer hashes, configs, source
hashes, report modes and seed lists passed read-only validation. The pilot
implementation and ignored artifacts are on this isolated branch; no original
`gpt-blue` or unrelated dirty files were overwritten.

Anchored greedy matched guard-only in all 72 paired episodes and at every Blue
action/host decision across 28,728 ticks. Its mean native return was −150.21,
versus Lancer −151.79 and current residual greedy −175.71, −169.04, −211.12
by training seed. The anchored arm improves on the current residual arm's poor
greedy outcome but contributes **zero** improvement over guard-only. Anchored
sampled means were −166.29, −159.08, −157.79, all below Lancer. The frozen
analysis marks `recommend_confirmation: false`; no RL superiority claim or
LLM-Red progression is warranted. See
[`finite-red-anchor-pilot-20261015.md`](../research/finite-red-anchor-pilot-20261015.md)
for protocol, paired differences, tails and limitations. The evaluation block
was used before, so these are development results rather than held-out proof.

Validation run: 14 focused tests passed (anchor, resume, worker pool and
privilege-access behavior), zero-head real-simulator preflight passed 20
episodes, six train runs and 14 evaluation/control reports completed, frozen
analysis and an independent
artifact audit passed, and all 72 greedy anchored action traces matched guard.
No commit, push, remote experiment, or teammate notification. Current branch:
`blue/finite-red-check` at `2f44db1`. Next concrete task: audit saved candidate
logit margins and actor corrections, then preregister one controlled-deviation
policy change and test it on fresh finite-Red evaluation seeds; keep LLM Red
deferred.

## Non-PPO method review, 2026-10-15

Reviewed the local A2C/JSRL redesign notes, algorithm shortlist and prior
experiments, then checked primary sources. Recommended screening synchronous
A2C with a JSRL-style Lancer guide-prefix curriculum: this changes the update
rule and how the learner reaches states, while avoiding the fixed per-action
0.9 anchor that froze greedy behavior in the latest pilot. The existing A2C
path is opt-in, but its earlier two-iteration/two-episode check matched the
untrained control and is not evidence of improvement. Discrete SAC remains a
later candidate because the decision-hook replay needs a verified next
candidate/mask and pending-action reward contract. No training launched. See
[`non-ppo-jsrl-a2c-option-20261015.md`](../research/non-ppo-jsrl-a2c-option-20261015.md)
for sources and a bounded screen protocol. Current branch `blue/finite-red-check`
at `2f44db1`; next task, if approved by the work plan, is a frozen A2C versus
guide-A2C development screen on verified fresh finite-Red seeds.

## Finite-Red A2C / guide-prefix screen, 2026-10-15

Completed the v2 frozen screen: three training RNG seeds each for plain A2C
and guide-prefix A2C, finite Red, native reward, eight iterations, and a shared
fresh evaluation block of 32 seeds. Shared Lancer, guard-only, Sleep and
masked-random controls are complete. The paired training-seed 95% intervals
overlap zero for every model-versus-Lancer/guard comparison and for guide-prefix
versus plain A2C. Guide-prefix A2C is near Lancer on greedy mean (-0.30 paired
difference), but the interval is [-25.36, 24.76]; this does not establish a
win. `recommend_confirmation` is false. Do not advance to LLM Red on this
evidence; add finite-Red training seeds before any superiority claim.

All six manifests/configs and source hashes, model artifact hashes and matched
32-seed report lists passed audit. The analyzer also validated baseline/model
seed alignment. Full estimates and tails are in
[`non-ppo-jsrl-a2c-option-20261015.md`](../research/non-ppo-jsrl-a2c-option-20261015.md)
and ignored run artifacts under `blue/results/finite_red_jsrl_a2c_screen_v2/`.
The earlier partial v1 output remains preserved and excluded. No further
training was started. Current branch `blue/finite-red-check` at `2f44db1`;
next task is a fresh finite-Red cohort with additional training RNG seeds,
then reevaluate the same decision gate before considering LLM Red.

## Finite-Red one-decision intervention diagnostic, 2026-10-15

Added `blue.training.harness_finite_intervention` to mine a frozen set of 20
actor/Lancer disagreements from the existing Blue-visible candidate audits and
compare (a) Lancer with one legal actor-selected alternative and (b) the actor
with one Lancer-choice substitution. Paired controls are cached by policy,
model and episode seed. The runner rejects used or declared-reserved evaluation
seeds, checks that paired action histories match through the target decision,
and reports reach/change status, native-return differences, tails and coverage.
Privileged event traces remain evaluator-only and are not read by the miner or
intervention policy.

Focused tests passed (5/5); the real-simulator one-target smoke used the fresh
development seeds 7689-7693. Both intervention types changed the selected
choice on all five seeds and every paired trace matched through the target.
For this single target, Lancer-side mean return difference was 0.0; actor-side
Lancer substitution averaged +33.6 with a wide 95% interval [-35.7, 102.9].
Treat this as runner validation only. The 20-target panel was not run; the
smoke block is now consumed. The intervention, ordered-policy parity and
harness checks passed 20 tests total. The 20-target panel is now running on
the audited fresh block 7694-7698 under
`blue/results/finite_red_intervention_panel_v1/`; the first target's
Lancer-side pair completed, and its actor-correction pair is next. Work is
uncommitted on `blue/finite-red-check` at `2f44db1`; unrelated dirty files were
preserved.

## Guide-prefix A2C tick-0 behavior replay, 2026-10-18

Replayed only the reset-state decision for `blue_agent_1` from all three
completed guide-prefix A2C checkpoints on the already-used screen block
9001-9032. Greedy actor choice disagreed with the Lancer-ranked choice on all
96 model/seed rows: each actor chose the operational-zone router 32/32 times;
Lancer chose a user host on every row. This confirms that the tick-0
router-vs-user gap seen in the native residual candidate audit is still
present in guide-prefix A2C. No environment step, reward, or privileged state
was read; this is a behavior-only replay on an existing evaluation block, not
a new performance result. The report, source/model/scorer hashes and per-row
choices are in the ignored artifact
`blue/results/finite_red_guide_prefix_tick0_replay_20261018/report.json`.
The initial replay seed passed the frozen model checks after setting Torch
threads to the recorded value (1); all three actor manifests and hashes
validated. No model training or full-episode evaluation was run. DAgger is
therefore still a plausible next experiment only for improving Lancer-like
behavior/parity; this trace gives no evidence it would beat Lancer.
