# GPT Blue pilot audit, 2026-10-07

Branch `gpt-blue`; completed remote source
`15b76448795b9568cc629e824fb7ddee4e5368ab`. Frozen model SHA-256
`949048797f2b89a5703189c53fb9062a82f056989da931d6c56b177af5073b2a`.
Native team return remains the primary outcome. All final models are retained;
no post-hoc checkpoint selection or reward change. Evaluation episodes
8221–8228 were previously used development episodes, not fresh tests.
Final block 7809–8200 remains unused by this experiment.

The downloaded complete Kaggle log records all 48 model evaluation cells,
16 baseline cells, and `PILOT COMPLETE: all six models evaluated` at 5075 seconds
(about 85 minutes). Reconstructed cells are stored in ignored
`blue/results/gpt_harness/kaggle_status/log_evaluation_cells.json`.
`blue.training.harness_audit` verifies complete matched seed sets and reports
every model against BOTH controls. No row/episode pooling confidence interval
or chance-based ordering significance test is used.

| Policy | RNG seed 0 mean | RNG seed 1 mean | Average | vs Lancer | vs guard |
|---|---:|---:|---:|---:|---:|
| Lancer | — | — | -54.75 | 0.00 | +0.25 |
| Coverage-only guard | — | — | -55.00 | -0.25 | 0.00 |
| RL zero | -67.25 | -71.75 | -69.50 | -14.75 | -14.50 |
| RL risk | -60.125 | -63.875 | -62.00 | -7.25 | -7.00 |
| RL risk + novelty | -54.50 | -56.50 | -55.50 | -0.75 | -0.50 |

Guard is 0.25 WORSE than Lancer, not 7.25 better. Risk-minus-zero matched
training-seed contrasts are +7.125/+7.875; both-minus-risk +5.625/+7.375.
These are descriptive estimates from two independent training seeds, not
established superiority. The best arm averages below both controls.

## Episode differences against coverage-only

| Model | 8221 | 8222 | 8223 | 8224 | 8225 | 8226 | 8227 | 8228 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_s0 | -30 | -9 | -23 | -15 | +8 | -63 | -3 | +37 |
| risk_s0 | -23 | -4 | -19 | -9 | +18 | -2 | -2 | 0 |
| both_s0 | -17 | +3 | -10 | -7 | 0 | -18 | +10 | +43 |
| zero_s1 | -12 | -3 | -26 | -27 | -8 | -39 | -4 | -15 |
| risk_s1 | -49 | -22 | -18 | -5 | +5 | +7 | -3 | +14 |
| both_s1 | -13 | +9 | -8 | +5 | +9 | -28 | +12 | +2 |

Ordering both >= risk >= zero holds on 5/8 episodes for seed 0 and **4/8**
for seed 1, correcting the review's 5/8 claim for both. Heterogeneous effects
do not invalidate an average. Episodes share models and environments, so a
simple 1/6 random-order calculation is not a justified significance test.
Worst observed model return is -133 (zero_s1); the absence of a -1200 collapse
on eight episodes cannot establish protection on unseen environments.

## What is and is not known

Equal episode returns and decision counts in early iterations do not prove
request-sequence identity. A zero-initialized head guarantees identical initial
actor outputs, but does not establish how long trajectories remain identical.
New optional diagnostics fingerprint full tick/agent/action/host requests,
including busy Sleep ticks, and measure fixed-state changes in actor logits
when ML columns are zeroed. They do not estimate action values or reward gains.
Historical trajectories cannot be recovered from counts; exact replay would
need the original runtime/RNG/config and unchanged training implementation.

Novelty's standalone PR-AUC is weak, but that does not establish that its
conditional actor input is useless, nor that its apparent return contribution
is general. No claim of regularization, credit-assignment repair, or other
mechanism is established by these means.

## Runtime evidence

The ACTUAL local `.venv-train` reports NumPy 2.5.3, Torch 2.14.1+cpu,
SciPy 1.18.1, Gym 0.26.2, Gymnasium 0.28.1, matching the remote pilot pins.
The older Environment handoff refers to historical targets; no current shared
NumPy 1.26.4/Torch 2.2.0 lockfile exists in this checkout. Thus the assertion
that Kaggle differs from this local validated harness stack is incorrect.
Equivalence with older experiments is still unestablished. An isolated older
runtime candidate is being tested rather than silently changing the pilot.

## Next gates

Audit saved manifests/weights and fixed-state ML sensitivity. Validate an older
NumPy runtime candidate with golden predictions, native simulator behavior,
PPO and focused checks. Expand detector testing to eight disjoint development
episodes while preserving training/calibration data and model settings.
Any later 32-iteration/five-training-seed comparison must be frozen, sharded,
resumable and clear BOTH controls. No larger run has been launched.

## Completed artifact and input audit

Official `pilot/report.json` now downloaded. All six manifests declare complete,
contain eight iterations, and match their final actor SHA-256. Official evaluation
cells equal the log reconstruction exactly. On reused diagnostic episodes 8226
and 8228, all 12 model replays reproduce original returns, despite added read-only
sensitivity instrumentation. Each request hash covers all 5 agents on every
399-tick episode, including busy waits. Checks did not modify saved checkpoints.

Fixed-state candidate top-index changes when zeroing ML inputs:

| Model | 8226 ML changes / sweeps | 8228 ML changes / sweeps | Novelty-only changes, 8226 / 8228 |
|---|---:|---:|---:|
| zero_s0 | 0 / 423 | 0 / 463 | 0 / 0 |
| risk_s0 | 17 / 583 | 16 / 491 | 0 / 0 |
| both_s0 | 42 / 487 | 28 / 543 | 31 / 30 |
| zero_s1 | 0 / 531 | 0 / 469 | 0 / 0 |
| risk_s1 | 4 / 597 | 15 / 457 | 0 / 0 |
| both_s1 | 49 / 521 | 45 / 424 | 45 / 28 |

Thus ML columns measurably affect final investigation rankings on these states.
This is not evidence that every affected choice is beneficial. It also cannot
identify when the original training trajectories first separated. Future training
can record full request hashes without dumping all observations.

## Older runtime compatibility candidate

Isolated Python 3.11.16, NumPy 1.26.4, SciPy 1.12.0, Torch 2.2.0+cpu,
scikit-learn 1.7.2, with remaining pins recorded in
`ops/kaggle/harness/requirements-legacy-candidate.txt`.
The original NumPy-2 detector fails deserialization on its BitGenerator state.
A separately rebuilt scorer from the ORIGINAL unchanged dataset passes the
original 128-vector risk/novelty golden check and 40-step native preflight.
Four full baseline replays (Lancer/guard x 8226/8228) reproduce pilot returns:
Lancer -40/-96, guard -37/-118.

An initial one-update PPO smoke succeeds in both stacks but is NOT identical:
newer-stack return -10 with 169 decisions versus older-stack -9 with 178.
Full request hashes differ. This is a stack-level stochastic-training difference,
not an isolated diagnosis of NumPy or Torch. All future arms and controls must
use ONE frozen runtime; do not mix their training results as replicas.
Existing pilot pins and checkpoints remain unchanged. The older runtime candidate
is optional and not yet installed in a new Kaggle submission.

## Expanded detector validation

TRAIN 7706–7709 and CALIBRATION 7701/7702 are unchanged. Test episodes:
7703, 7704, 7629, 7630, 7640, 7705, 7710, 7711. These are previously consumed
development episodes (including expanded script range 7701–7729), disjoint
from training/calibration, not fresh final evaluation.
Recollected training/calibration feature arrays and privileged labels match
original arrays EXACTLY. Model parameters and preprocessing were unchanged;
refitted risk predictions still match original golden outputs.

| Episode | HGB PR-AUC | Logistic PR-AUC | Novelty ROC-AUC |
|---|---:|---:|---:|
| 7703 | 0.8194 | 0.7511 | 0.5634 |
| 7704 | 0.6422 | 0.5140 | 0.5424 |
| 7629 | 0.7718 | 0.6861 | 0.4769 |
| 7630 | 0.7265 | 0.5712 | 0.4733 |
| 7640 | 0.6645 | 0.6006 | 0.6156 |
| 7705 | 0.8086 | 0.7478 | 0.4623 |
| 7710 | 0.6394 | 0.5396 | 0.5489 |
| 7711 | 0.7511 | 0.6382 | 0.5509 |

Equal-episode macro PR-AUC: HGB **0.72794**, logistic **0.63107**.
Paired HGB-minus-logistic difference **+0.09687**, conditional episode t interval
**[+0.06861, +0.12514]**, n=8, df=7. This describes episode variation for ONE
fixed fit on reused development scenarios; it is not training-seed uncertainty,
fresh confirmation or proof of a defensive return gain. Novelty remains weak
standalone (pooled PR-AUC 0.2514 vs prevalence 0.2412); keep its probability
semantics separate and retain the risk-only control.

Expanded dataset SHA-256:
`9fbc7e4fce15892535fbdd7907c6f1ab35f2f726fcbdcb1470d5d221a2d007e2`.
Expanded scorer SHA-256:
`94b37182ff78d17a20f68c77cb03db585d883e2121d385562835d313aed8a98d`.
Files live under ignored `blue/results/gpt_harness/ml_expanded/`.

## Validation and next concrete task

New request-hash, sensitivity/RNG preservation, complete-cell matching and
per-episode weighting tests added. Current-stack harness/diagnostic suite:
16 tests. Older candidate: 47 harness/diagnostic/age-guard tests passed before
adding the episode-weighting test; updated harness/diagnostic suite 16 passed.
Diagnostics on/off PPO smokes yield exactly identical actor and critic weights
within the current runtime. Compiler/whitespace checks passed.

Next: freeze the runtime/artifact/source and scenario suite for five separate
Kaggle shards, one training RNG seed and all three arms per shard, 32 iterations.
Success must be evaluated against BOTH Lancer and guard using all training-seed
contrasts and retaining tail losses. The source runner now exports only artifacts
(runtime/source under /tmp), supports an explicit frozen runtime profile, and
can enable request diagnostics. No main-experiment package is frozen, no larger
run launched, and no claim of learned superiority is made.

Runtime candidate full installed lock is recorded in
`ops/kaggle/harness/requirements-legacy-lock.txt` (includes validation tools).
All detector episodes use Discovery Red and one collection policy; neither
these metrics nor pilot returns establish generalization to other attackers.

Additional recovery check under the older runtime: interrupted/resumed and
uninterrupted two-update smokes produced EXACT actor/critic weights and history,
including request hashes. A deliberately mismatched runtime manifest was rejected
before resume. This validates recovery within one frozen stack, not across stacks.
