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
