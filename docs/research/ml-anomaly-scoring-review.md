# Review: `ml-anomaly-scoring`

Reviewed at commit `dc96f96` ("blue: add anomaly scorer fit contract").

## Short answer

This branch does **not** do ML scoring yet. It is a plumbing contract: a
model class, one extra column in the observation, and three tests. The
model is never trained in the run path, the column it adds is a constant,
and the model it uses was already measured here at chance level.

Recommendation: **do not merge yet.**

## What is actually in the branch

Three commits ahead of the fork point (`a7a174a`):

| File | Change |
|---|---|
| `blue/blue_iforest_anomaly.py` | new, 279 lines: IsolationForest wrapper |
| `blue/cc4_epymarl_wrapper.py` | appends 1 anomaly-score column per host row |
| `blue/tests_blue/test_foundation.py` | 3 new tests, geometry 510 -> 561 |

Everything else in the diff came from the merged `blue/mappo-training`
history, not from this line of work.

## Three reasons it is not scoring yet

### 1. The runtime scorer is never fitted, so the column is dead

`cc4_epymarl_wrapper.py` creates `IsolationForestAnomalyScorer()` in
`__init__` and nothing in the repository ever calls
`fit_anomaly_scorer(...)`. An unfitted scorer returns `0.5` for every
input, so every host row gets the same useless `0.5` appended.

### 2. Isolation Forest already failed on this project

`docs/archive/handoff.md` and `docs/archive/blue-agent-plan.md` both record
Isolation Forest AUC at about **0.494**. That is chance. The branch re-adds
the same detector without addressing that result.

For comparison, the logistic risk model reached AUC 0.68. That is the
detector that actually produced signal.

### 3. `sklearn` is not in the pinned requirements

`cage-challenge-4/Requirements.txt` does not list `scikit-learn`, but
`cc4_epymarl_wrapper.py` imports `blue_iforest_anomaly` at module level,
and that module imports `sklearn` at module level. The Blue wrapper
therefore cannot be imported on an environment built from the pinned
requirements. This project checkout has no `sklearn`, which reproduces it.

This is a blocker for merging, independent of the modelling questions.

## Bugs confirmed by running the code

Each row below was reproduced, not inferred.

| Issue | Evidence |
|---|---|
| `fit()` discards constructor settings | requested `n_estimators=500, contamination=0.05, random_state=7`; actual `100, 'auto', 0` |
| Unfitted scorer returns a constant | `score_batch` on 6 extreme inputs returned `[0.5]` |
| Scaling uses the training set's own min/max | ~23% of held-out **clean** hosts scored above 0.5; the number is not a meaningful threshold |
| Extreme anomalies clip to 1.0 | 420 of 600 inputs tied at the top, so the agent cannot rank the worst hosts |
| Training data does not match run data | `collect_clean_reset_matrix` reads only `reset()`; its `steps=300` argument is never used, and the default `fit_clean_reset_collection` uses exactly that collector |

In fairness to the code, the min-max scaling is monotonic, so it does not
hurt host *ranking* inside the normal range: AUC matched the raw
`decision_function` exactly. The damage is to threshold semantics and to
ordering at the extreme end, which is where host selection actually
happens.

There is also an inconsistency between the two collectors: the "clean"
one runs `DiscoveryFSRed` as the red agent, while the benign one uses
`SleepAgent`. Red behaviour should be off in both, or the "clean" labels
carry noise.

## Why the tests do not catch this

The new tests check that scores are finite and inside `[0, 1]`, plus one
obvious outlier outranking one normal row. A constant `0.5` satisfies all
of them. `test_anomaly_score_is_appended_per_host` asserts
`0 <= row[-1] <= 1`, which the dead column passes. The suite is green
while the feature does nothing.

## Smaller items

- `save()` pickles the object with no record of `sklearn` version, feature
  names, or feature order. A reordered feature vector would score silently
  wrong. AGENTS.md asks for artifact hashes and a preprocessing version.
- Untrained models return `0.5` silently. `0.5` is also the decision
  threshold used elsewhere in the pipeline, so a missing model looks like a
  borderline host. Missing model should be a loud error.
- `collect_benign_matrix` reaches into
  `cyborg.environment_controller.state` and mutates `fp_detection_rate`.
  That is private simulator coupling. Per AGENTS.md, Environment owns
  simulator hooks, so this should be proposed as a hook instead.
- `collect_benign_matrix` defaults to five Blue agents and does not check
  that the scenario actually produced five.
- Nothing records which seeds, host rows, or preprocessing produced a fit
  model, so a fitted artifact cannot be traced back to its data.

## If this work continues

1. Do not wire the score into the observation until it has a **held-out
   AUC** number on seeds that were never used for fitting. Report the
   number even if it is worse than 0.494.
2. Train on stepped, monitored data (`collect_benign_matrix`), not on
   reset frames.
3. Fix `fit()` so it honours `n_estimators`, `contamination`, and
   `random_state`.
4. Fail loudly when the scorer is unfitted instead of returning `0.5`.
5. Return the raw `decision_function` (or a percentile against a stored
   clean reference set) instead of min-max over the training data, so the
   value survives out-of-range inputs without clipping ties.
6. Add `scikit-learn` to the pinned requirements with a version.
7. Add a test that fails when the appended column is constant, and a test
   that a fitted model separates held-out anomalies. Without those two, the
   feature can silently stay dead.

## Merge impact on other work

This branch changes the default observation from 510 to 561 by adding one
column per host row. Work on `blue/metric-repair-maxage` builds 15-wide
host rows, which would become 16-wide. That breaks every saved checkpoint,
config, and recorded result that assumes the current geometry.

Per AGENTS.md this counts as a breaking schema change: it needs a version
number and migration notes, not a quiet merge. An unrelated long training
run is already pinned to a commit on `blue/metric-repair-maxage` and is not
affected by this branch.

## How this was checked

- Read the full diff of `dc96f96` against the fork point.
- Ran the scorer class directly in a clean virtualenv with
  `scikit-learn`: hyperparameter handling, unfitted behaviour, clean vs
  anomalous score distribution, clipping ties, and AUC of the shipped
  scaling vs the raw decision function.
- Confirmed `scikit-learn` is absent from this checkout and from
  `cage-challenge-4/Requirements.txt`.
- No simulator episodes were run; the collectors were reviewed by reading
  them, and that reading is the basis for the train/serve mismatch claim.