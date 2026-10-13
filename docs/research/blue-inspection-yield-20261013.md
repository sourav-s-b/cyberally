# Inspection-yield input feasibility (2026-10-13)

## Question

The frozen compromise scorer often ranks currently compromised hosts well,
yet directly prioritizing its score worsened Blue return. Can Blue-visible
features available **before** an `Analyse` request predict whether that
specific inspection will produce a new actionable confirmation?

This is narrower than estimating the true value of selecting a host. Only
chosen actions have observed completion results, so the dataset is selected
by the guard-only policy. An outcome predictor fit here may fail when used to
rank previously unchosen hosts. Nothing here is a counterfactual action-value
estimate or a claim of improved Blue return.

## Data contract

`blue/training/inspection_yield_data.py` runs the frozen finite-Red
`HarnessPolicy(max_age=80)` without any ML ranking or RL hook. It advances
Blue-visible preprocessing and evidence memory on **every tick**, including
busy ticks. When the policy requests `Analyse`, it saves only the requested
host's pre-action features, ML scores, heuristic score, agent, and branch.
The wrapper's pending identity joins that request to a later completion,
failure, or unresolved horizon end. A completed `Analyse` is labelled
`new_confirmation=true` only when Blue's own tracker enters `CONFIRMED`.
Failure is not a completed negative label; horizon-censored requests are
excluded from model fitting. **Forced verification and max-age-guard scans
are also excluded from model fitting**, because the learned hook operates
only at ordinary sweep decisions. Action success is not presumed to prove
host cleanliness.

For diagnosis only, the simulator's Red-session presence at request time is
sampled **after all Blue actions are fixed** and saved in a separate JSONL
file. It is never read by the policy, scorer, or fitter. Missing telemetry is
stored as JSON `null` and remains NaN for the tree model; no zero-imputation
is confused with a clean observation.

The source plan, scorer, collector and each episode's files have SHA-256
records. Collection rejects final seeds 7809–8200, any seed outside the
frozen plan's training/development cohorts, a changed scorer, or a non-finite
Red configuration. `inspection_yield_audit.py` checks one-to-one joins,
outcome semantics, file hashes, and whole-episode partitions. No host rows
from one episode cross a partition.

## Frozen predictive feasibility design

- Training episodes: 7730–7741 (12 independent simulator resets).
- Validation episodes: 7742–7745 (four resets; reserved for the initial
  model check, not for row-wise hyperparameter search).
- Diagnostic audit episodes: 8221–8228 (eight **reused development**
  episodes; these are not untouched final evaluation).
- Fitter: two histogram-gradient-boosted trees with **identical fixed
  settings** and equal total sample weight per episode. Both receive the
  pre-request Blue-visible
  vector, its availability flags and event memory, current risk and novelty
  scores, the heuristic base score, and agent identity. One also receives
  three explicit consequence-context inputs. This is a pre-specified input
  ablation, with no hyperparameter search. Neither sees the post-request
  tracker result or privileged truth as an input.
- Consequence context: the public maximum penalty weight for that zone and
  mission phase, time since the host's last completed inspection (or never
  inspected), and age-weighted unreviewed Blue-visible telemetry changes.
  These are **potential exposure proxies**, not measured penalties or
  causal blame for ignoring a host. The native reward is shared across all
  Blue agents and may depend on hidden state, so it stays an evaluation/
  learning outcome rather than a privileged actor input. A lagged reward or
  return-to-go may inform a training critic, but it cannot identify the
  specific ignored host without an action-conditional credit model or
  controlled counterfactual rollout.
- Reference: frozen compromise-risk score on the same selected actions.
- Evaluation: PR-AUC, ROC-AUC and Brier **per episode**, with prevalence
  reported per episode. Row-pooled metrics are not independent evidence.

The frozen compromise model was trained on Discovery Red episodes 7706–7709,
calibrated on 7701–7702, and tested on 7703–7704; this collection uses finite
Red. Any difference in predictor behavior can include attacker-distribution
shift. Report model prediction quality and Blue defensive return separately.

## Results

All 24 episodes completed locally with 20,490 `Analyse` requests: 20,425
observed successes, zero failures and 65 censored at episode end. The
collector's eight development-episode returns matched the prior frozen
guard-only results exactly, and the 8221/8222 request files matched an
earlier independent collector run byte-for-byte. The episode validator
accepted all file hashes, joins and disjoint partitions.

The first fitter included forced verification actions and therefore answered
the wrong question. Verification is never selected by the learned sweep hook.
That first readout was discarded before interpretation; the corrected fitter
uses **only completed `sweep_scorer` requests**. It excludes guard-forced
sweeps too. The denominator is selected actions, not all legal candidates.

| Whole-episode partition | Episodes | Completed selectable sweeps | New confirmations | Mean episode prevalence | Frozen risk score PR-AUC | Yield model PR-AUC | Yield model + consequence PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 12 | 7,634 | 443 | 0.059 | 0.757 | 0.947 | 0.947 |
| Validation | 4 | 2,501 | 144 | 0.058 | 0.768 | 0.896 | 0.892 |
| Reused development audit | 8 | 5,125 | 274 | 0.054 | 0.735 | **0.866** | **0.867** |

Metrics are **episode-macro means**, not row-pooled confidence estimates.
On the eight audit episodes the standard yield model improved PR-AUC over
the frozen compromise score on all eight, by +0.029 to +0.240 (mean +0.131).
The consequence features changed PR-AUC by only +0.0008 on average and won
on four of eight episodes; validation declined by 0.004. Their predictive
contribution is unresolved and too small to justify putting them into the
Blue actor. The standard yield model's audit mean Brier score is 0.013;
the frozen compromise score is calibrated for a **different target**, so
its Brier score is not a fair calibration comparator here.

The largest simple pre-request differences between positive and negative
sweeps were Blue-visible process and connection counts. On audit positives
averaged 8.14 processes and 7.29 connections, versus 1.41 and 1.06 on
negatives. This is an association, not proof those features cause defensive
gain. No requested sweep had a currently visible unknown-file count above
zero, so this result was not a shortcut through the post-`Analyse` file
result.

Frozen artifact hashes (all large files are ignored by Git):

| Artifact under `blue/results/inspection_yield_24/` | SHA-256 |
|---|---|
| `summary.json` | `8799414abdd5aa998b5c41504c925839e3478049b25872bbf26bcfeb0da60628` |
| `episode_partitions.json` | `5f097eff90ff8fe3065ea41c6d796f8d0a076dbe0d6259b84e87950a1e6762a8` |
| `models/report.json` | `40bc33b30e4e868efe8f7de45e9fa265453977e8239964a6a60b9d3802379d9c` |
| `models/inspection_yield_standard_inputs.joblib` | `ada64fe05872cbae2b7967d7f40c7128bcf0c488f2436429661466814b8cdb55` |
| `models/inspection_yield_with_consequence_context.joblib` | `ee81ce69624e028de8ef33a70892cdf6d90983a69b6513b1c21b5089027aeb21` |

## Implementation and limits

Code: `blue/training/inspection_yield_data.py`,
`blue/training/inspection_yield_audit.py`,
`blue/training/inspection_yield_model.py`,
`blue/harness/consequence_context.py`, and focused tests. Large JSONL
records and model weights live under ignored `blue/results/`, not in Git.
Fourteen focused tests passed. The frozen local runtime used Python 3.12,
NumPy 2.5.3 and scikit-learn 1.7.2; exact library versions and source
hashes are saved in the ignored model report. No full RL training suite was
run because no actor/trainer path changed.

This feasibility run uses selected actions from a single guard-only policy.
The predictor **does** look useful for observed selected-action yield, but
the current data do not validate its ranking over *unselected* candidate
hosts. The next decision experiment should obtain broader action coverage
(for example, a pre-registered randomized inspection mixture in training
episodes), then compare a yield-assisted ranking against the guard and
Lancer on paired episodes. If moved into RL, it requires fresh on-policy
rollouts and independent training seeds. The eight audit episodes have
already been examined during this project; no final-seed or generalization
claim follows from them. Final seeds remain untouched.
