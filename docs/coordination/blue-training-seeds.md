# Proposal: blue-training-seeds

- Status: implemented on `blue/mappo-training` (`cc9a5c2`), unreviewed
- Author role and branch: Blue, `blue/mappo-training`
- Producer and affected consumers: Producer Blue (`CC4MARLEnv`, training
  driver manifest); consumers Evaluation (retraining loop seed bookkeeping,
  held-out suite separation) and Environment (training config pinning).
  Red: none.
- Related issue/PR and exact dependency commits: extends BLUE-03 conformance
  (`3b414b4`); implemented in `cc9a5c2` on top of `a241c77`. EPyMARL pinned
  at `uoe-agents/epymarl@cbc38c09` (see `blue-epymarl-contract.md`).

## Problem and current behavior

The pinned `EpisodeRunner` calls bare `env.reset()` every episode
(`episode_runner.py:65`). The wrapper honored the constructor seed only on
the first reset; every later episode continued the RNG stream with no
recorded seed, so training episodes were not reproducible as a seed list
and the run manifest could not say which seeds were trained on.

## Proposed contract change

Additive; default (`seed_cycle=None`) preserves foundation-v2 behavior, so
`WRAPPER_VERSION` stays `foundation-v2`. No mask, reward, pending-action,
or observation change.

Before:

```python
CC4MARLEnv(seed=7629, ...)   # bare reset(): 7629 once, then RNG continuation
env.reset_seeds              # AttributeError (did not exist)
```

After:

```python
CC4MARLEnv(seed=7629, ..., seed_cycle=[7629, 7630, 7640])
env.reset()      # consumes next cycle entry, wrapping: 7629, 7630, 7640, 7629, ...
env.reset(99)    # explicit seed wins once, cycle pointer untouched
env.seed(77); env.reset()  # setter wins once, cycle pointer untouched
env.reset_seeds  # [7629, 7630, 7640, ...] — every applied seed, None for
                 # RNG-continuation resets when no cycle is given
```

- `seed_cycle` must be a non-empty list/tuple of int seeds (bools rejected);
  otherwise `ValueError`.
- `CC4BlueWrapper` passes `seed_cycle` through and exposes `reset_seeds`.
- Driver (`blue_train_mappo.py`): `build_config(..., train_seeds=(7629, 7630,
  7640))` with `--train-seeds` CLI override; manifest `seeding` block records
  `seed_cycle`, chronological `reset_seeds` (train and greedy test episodes
  interleaved — test episodes also consume cycle slots), and `n_episodes`.
- Train/eval separation: the cycle above is the *training* seed source.
  Held-out evaluation suites (≥30 seeds per the Blue plan) remain
  Evaluation's build; `blue_eval_mappo.py` keeps explicit per-episode seeds.

## Required branch changes

| Role | Files/behavior to update | Dependency | Acceptance test |
|---|---|---|---|
| Blue | `cc4_epymarl_wrapper.py` cycle/record; driver manifest; `tests_blue/test_seed_cycle.py` (6 tests) | none | cycle order/wrap, explicit-override, setter precedence, validation, default unchanged, cross-env reproducibility |
| Evaluation | retraining loop reads `manifest.json Seeding.reset_seeds` for the trained seed list; keep held-out suite disjoint from any published cycle | this lands | fixture: manifest seed list round-trips; eval seeds disjoint from train cycle |
| Environment | pin `train_seeds` alongside dependency profile in training configs | this lands | clean-install run reproduces the same `reset_seeds` prefix |
| Red | none | — | confirm none exist |

## Merge and migration order

1. This lands with default `seed_cycle=None` — main stays runnable, no
   consumer forced to move. No checkpoint exists with cycle semantics yet,
   so no compatibility burden; manifests without a `seeding.seed_cycle`
   field predate this change.
2. Evaluation adopts `reset_seeds` for loop bookkeeping; Environment pins
   the cycle in the reproducible training profile.
3. No removal step: the no-cycle path stays as the documented default.

## Validation and decision

- `tests_blue/test_seed_cycle.py`: 6 passed (sim venv). Full `tests_blue`:
  47 passed, 1 skipped (sim); 42 passed (train venv).
- Driver smoke (train venv, 2x50, seed 7): 4 resets
  `[7629, 7630, 7640, 7629]` in manifest (3 train + 1 greedy test).
- 8x100 short run (~3.5 min): finite pg/critic losses, periodic + first
  checkpoints saved, manifest seed list complete.
- Affected-role review: pending (Evaluation for manifest consumption,
  Environment for config pinning). Do not mark accepted just because this
  file exists.
