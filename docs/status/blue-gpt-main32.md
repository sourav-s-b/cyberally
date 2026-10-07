# GPT Blue 32-iteration comparison

Branch `gpt-blue`. User authorized submission after the completed pilot audit.
Frozen design: `ops/kaggle/harness/main32-plan.json`; submission resolves its
source commit to the published HEAD in private `pilot.json`, then seals that
cohort configuration across all five shards. No final-seed use, scoring change,
new RL algorithm, new risk filter, best-checkpoint selection or metric switch.

Five private CPU Kaggle kernels, one training RNG seed 0/1/2/3/4 each. Each
trains ALL three arms (zero/risk/both) for 32 iterations x four fresh 400-step
rollouts, using unchanged training episode seeds 7706–7709, gamma .99,
bonus .25, temp .5, hidden 64, lr .0003, max_age 80. New models under one
runtime; no pooling old pilot training into these replicas.

Runtime: Python 3.11.16, NumPy 1.26.4, SciPy 1.12.0, Torch 2.2.0+cpu,
scikit-learn 1.7.2; complete installed lock hash is frozen in the package.
Scorer rebuilt on unchanged original training/calibration data under this
runtime, SHA-256 ec4026a3c81c2746253f306fd7add00a07f42fe13533b856fb6b4d01b8402fcd.
Original golden vectors and native 40-step episode must pass remotely before
any training. Actual runtime versions and environment kwargs must match.

Every model evaluated on the SAME 32 previously used development environments
8221–8252, Discovery Red, at 400 steps. Not fresh held-out confirmation or
other-attacker generalization. Final seeds 7809–8200 structurally excluded.
Lancer/guard/Sleep/masked-random baselines run under the identical frozen
configuration. Baselines replicated per shard must match exactly at aggregation.

Primary arm declared before this run: BOTH risk + novelty inputs. It must have
positive lower endpoints of paired training-seed t95 intervals versus BOTH
Lancer and guard (intersection requirement). The replication unit is five
training RNG seeds, df=4. The shared 32 episodes are blocked environments,
not 160 independently trained replicas. Other arm and ML contrasts remain
descriptive; no post-hoc winner among three arms. Retain every episode and tail
loss. Success applies only to this development suite and frozen attacker/horizon.

New diagnostics save full requested-action hashes (including busy ticks and
host targets) and fixed-state ML logit/index sensitivity. These do not assert
completion or beneficial action values. Iteration-boundary checkpoints retain
actor/critic/optimizer/RNG state and fresh-on-policy recovery. Runtime/source/
configuration drift on resume is rejected. Source/runtime in /tmp; only useful
experiment artifacts exported, unlike the pilot's full virtualenv export.

Local validation: main/shard validation, cohort completeness, cross-shard
baseline matching, training-unit CI calculation, existing harness/diagnostics
and baseline-cache tests: 18 passed under older candidate. Exact full-lock fresh
runtime/preflight and submission status recorded below when measured.

Aggregate after all five finish using `python -m blue.training.harness_experiment
--config <frozen-cohort-pilot.json> --folders <each-downloaded-pilot-directory>
--out <aggregate.json>`. Missing shards, altered configs, unmatched episode
sets, foreign/incomplete checkpoints or baseline mismatches block a full-cohort
result. Source/model/runtime references remain exact, not moving branch names.

PR checklist: optional Blue experiment only; shared simulator unchanged; weights,
logs and credentials ignored; source and contract review pending before merge.
This status does not notify another role. No launch is inferred from preparation;
actual accepted kernel IDs/statuses must be verified after submission.

Fresh /tmp/gpt-blue-main-runtime installed FROM the complete lock successfully;
golden ML predictions, native simulator episode and exact runtime/version
preflight passed. Same 18 tests passed in this clean full-lock environment.
Package generation uses harness_prepare_main, rejects a dirty tree and checks
all ML/requirements hashes before inserting the clean published source commit.
