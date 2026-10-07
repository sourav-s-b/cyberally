# Blue investigation harness v1

Opt-in prototype on `gpt-blue`. Native reward, simulator behavior, action
validity masks, legacy training paths and existing checkpoints are unchanged.
The feature contract and new actor are incompatible with legacy checkpoints.

## Components

- `features.py`: Blue-visible snapshots, availability and freshness, completed
  action history, belief, idempotent per-tick deltas, public phase schedule and
  host-role context. Tree inputs preserve NaN; neural inputs receive TRAIN-only
  normalization plus an explicit availability bit for every input column.
- `scoring.py`: histogram gradient boosting for supervised compromise risk,
  calibrated on separate episodes; Isolation Forest fitted to benign TRAIN
  rows for a separate novelty percentile. Novelty is not a probability.
- `policy.py`: busy/validity handling and existing urgent/verification rules,
  then the existing completed-analysis age intervention, then learned ranking.
  ML scores never disable Remove/Restore or remove legal investigation targets.
  `max_age=80` is an intervention threshold, not a guaranteed bound.
- `harness_rl.py`: fresh PPO rollouts using the same preprocessing and harness
  during training and evaluation. All actor decisions are retained; guard
  decisions are deterministic parts of the policy, not sampled actor rows.
  Greedy agreement and sampled override frequency are logged separately.
- `harness_eval.py`: every requested action, target, pending/completion status
  and selected-host input; privileged compromise labels and penalty events
  are written to a separate diagnostic file. Event sums must match native reward.

The central critic remains the existing Blue-visible pooled-context V(s).
It is not used as an action-value gate. Discounted Monte Carlo credit assignment
(`gamma=0.99`) is retained; the harness does not claim to fix that independently.

## Runtime

The optional pinned ML packages live in ignored storage; Environment's shared
runtime has not been changed. From this worktree:

```bash
uv pip install --python .venv-train/bin/python \
  --target blue/results/harness_runtime --no-deps \
  -r blue/harness/requirements.txt
export PYTHONPATH="$PWD/blue/results/harness_runtime${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
```

This uses the existing training environment's NumPy/SciPy/Torch/CybORG.
Recorded ML runtime: NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.7.2.
Pickles must be trusted local artifacts, not downloaded untrusted models.

## Reproduce the bounded build checks

```bash
.venv-train/bin/python -m pytest blue/tests_blue/test_harness.py \
  blue/tests_blue/test_max_age_guard.py blue/tests_blue/test_risk_ablation.py -q

.venv-train/bin/python -m blue.training.harness_ml collect
.venv-train/bin/python -m blue.training.harness_ml fit \
  --model blue/results/gpt_harness/scorer_v1.pkl

.venv-train/bin/python -m blue.training.harness_rl \
  --scorer blue/results/gpt_harness/scorer_v1.pkl \
  --out blue/results/gpt_harness/new_smoke --iters 1 --eps-per-iter 2 --steps 100

.venv-train/bin/python -m blue.training.harness_eval \
  --scorer blue/results/gpt_harness/scorer_v1.pkl \
  --out blue/results/gpt_harness/new_mechanism
```

Training/evaluation run directories must be new. Evaluation of a new actor
uses `--arms actor --model-dir <checkpoint directory>` with its frozen horizon
and max-age setting. The checkpoint loader rejects mismatched scorer bytes,
feature contracts, incomplete training and modified actor artifacts.
The `--ml-inputs zero` control zeros both ML columns during train and eval.

Default ML splits: TRAIN 7706–7709, CALIBRATION 7701/7702, TEST 7703/7704.
These are previously consumed DEVELOPMENT episodes. Rows are not independent
experiments; there are only two held-out detector episodes. Collection and
training reject final seeds 7809–8200; old risk datasets containing 7901–7908
were not reused. Default mechanism episodes 8241/8229 are already consumed.

## Measured limits

The grouped development comparison improved HGB PR-AUC to 0.747 from 0.655
and Brier error to 0.102 from 0.121 for a freshly fitted logistic baseline
using the SAME new features. This is not a comparison against the old frozen
logistic artifact, and does not establish better RL returns. On test episode
7704 the HGB log loss was worse (0.362 versus 0.334), despite better PR-AUC
and Brier error. Do not claim uniform calibration superiority.

Isolation Forest novelty PR-AUC was 0.288, versus prevalence 0.252. Its
heuristic bonus defaults to zero; it remains an experimental RL input.

The ML-assisted heuristic scored -78/-128 on 8241/8229, versus -59/-107 for
the coverage-only heuristic. Thus better detector metrics have NOT yet shown
a better defender. No long RL run or multi-seed RL improvement is claimed.

Next experiment: frozen harness with matched independent RL seeds, `both`
versus `zero` ML inputs, plus coverage-only and Lancer baselines. Freeze all
settings, report all runs and analyze training-seed replication separately
from shared environment seeds. Keep native mean return primary; report tail
failures, detection delays and unresolved incidents as additional diagnostics.

Optional diagnostics for new training runs: pass `--diagnostics` to
`blue.training.harness_rl`. Each episode prints a full requested-action hash
(tick, agent, action name and target, including busy Sleep ticks); iteration
files store fixed-state logit/ranking sensitivity to zeroed ML columns. This
does not recover old action sequences or measure action-value advantage.

Audit downloaded pilot cells with `python -m blue.training.harness_audit
--cells <cells.json> --out <audit.json>`. It requires all six models on identical
episodes and reports both baseline contrasts without pooling training replicas.
Inspect saved models without training with `python -m
blue.training.harness_model_audit --model-dir <run> --scorer <scorer.pkl>
--seeds 8226 8228 --out <diagnostic.json>`. Use the artifact's validated runtime.
These reused episodes are mechanism diagnostics, not new performance evidence.

## Opt-in v2 investigation scheduler

See `docs/research/blue-harness-redesign-20261007.md` for the design audit,
limitations and experiment gates. `blue.training.harness_redesign` exposes
`--algorithm mappo|a2c`, uses persistent observable evidence and candidate-set
context, independent actor/critic optimization, and an exact teacher-mixture
sampling policy. Existing checkpoints and runners are unchanged.

```bash
python -m blue.training.harness_redesign train --out blue/results/v2_smoke --algorithm a2c --scorer PATH_TO_TRUSTED_SCORER
python -m blue.training.harness_redesign eval --model-dir blue/results/v2_smoke --out blue/results/v2_eval --scorer PATH_TO_TRUSTED_SCORER --eval-seeds 8241 8229
```

Default training is a short two-iteration smoke check, not long training.
Use the pinned runtime and the exact scorer used to train. Resume with
`--resume` and an unchanged configuration. Source/artifact/runtime drift
is rejected. Primary evaluation samples the trained marginal distribution;
`greedy-diagnostic` is a separate policy. The untrained-mixture control
uses the same sampler without learned residuals. Policy RNG repetitions
are nested within episodes and training seeds, never extra trained models.

The older remote checkpoint replay mismatch must be resolved before a new
Kaggle comparison. No defense improvement is claimed for v2.

V2 pins one Torch thread by default (`--threads 1`) and records the setting.
Its greedy diagnostic declares a 1e-6 near-tie resolution; sampled primary
execution remains the exact trained marginal. Numerical ranking intervention
in v1 is available only through `harness_numerical_audit`, preserving frozen
checkpoint and original evaluation semantics.
