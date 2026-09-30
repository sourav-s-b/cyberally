# Proposal: blue-epymarl-contract

- Status: draft
- Author role and branch: Blue, `blue/epymarl-conformance` (stacked on `blue/baseline-policies`)
- Producer and affected consumers: Producer Blue (`CC4MARLEnv`); consumers
  Evaluation (step-output reward shape, `episode_limit` bootstrapping flag) and
  Blue training configs. Red: none.
- Related issue/PR and exact dependency commits: BASE `d05d043`; stacked on
  BLUE-02 commit `4471c27` (independent files, no dependency). EPyMARL pinned
  at `uoe-agents/epymarl@cbc38c09` (shallow clone, machine-temp, not in Git;
  EPyMARL itself is NOT installed).

## Problem and current behavior

The pinned `EpisodeRunner` constructs envs as
`REGISTRY[env](**env_args, common_reward=..., reward_scalarisation=...)`,
accumulates `episode_return += reward` (scalar), steps with a torch batch
(`env.step(actions[0])`), and reads `env_info.get("episode_limit", False)` to
tell horizon cut-offs (critic bootstraps) from true terminals. The foundation
wrapper accepted none of the constructor kwargs, always returned a 5-list
reward, always returned `info == {}`, and had no `close`/`seed`/`render`/
`save_replay`/`get_stats` lifecycle. It could not be constructed by the runner.

## Proposed contract change

All additive; defaults preserve foundation-v1 behavior, so `WRAPPER_VERSION`
stays `foundation-v1`. No mask, pending-action, or observation change.

Before:

```python
CC4MARLEnv(seed, max_hosts, steps, mask_mode)   # TypeError on runner kwargs
step(...) -> (obs_list, [r]*5, terminated, truncated, {})
```

After:

```python
CC4MARLEnv(seed, max_hosts, steps, mask_mode,
           common_reward=False, reward_scalarisation="sum")  # sum|mean validated
step(...) -> (obs_list, reward, terminated, truncated, {"episode_limit": bool})
```

- `common_reward=True` returns `float(rewards[0])`. The native Blue reward is
  already one shared team signal duplicated per agent, so the scalar is that
  signal normalized once — never a 5x sum (contracts invariant). `sum`/`mean`
  coincide on identical values; the flag is still validated.
- `info["episode_limit"]` is True only at the horizon end (`truncated`, or
  `terminated` with `tick >= episode_limit - 1`, which is where the native
  scenario always ends). An early natural terminal, should one ever occur,
  keeps `False` and is treated as a true terminal.
- Lifecycle: `close()`/`render()` no-ops (no simulator resource / visual);
  `save_replay()` raises `NotImplementedError` (runner only calls it when its
  own `save_replay` flag is set); `get_stats()` returns `{}`; `seed()` setter
  applies on the next reset via a one-shot pending flag. This also fixes a bug
  found in testing: a `seed()`-set value was silently ignored by the next bare
  `reset()` (RNG advanced instead of applying the seed).
- Internal rename `self.seed` -> `self._seed` (attribute/method collision with
  the required `seed()` method). No in-repo consumer read `env.seed`.

## Required branch changes

| Role | Files/behavior to update | Dependency | Acceptance test |
|---|---|---|---|
| Blue | `cc4_epymarl_wrapper.py` ctor/reward/info/lifecycle; `CC4BlueWrapper` gains `seed()`/`close()` | none | `tests_blue/test_epymarl_conformance.py` (6 tests) + 16/16 ABC methods present against the pinned tree |
| Evaluation | reward readers must handle scalar when `common_reward=True`; use `info["episode_limit"]` for timeout-vs-terminal bootstrapping | this lands | fixture: scalar reward equals per-agent entry; flag False mid-episode, True at native end |
| Red | none | — | confirm none exist |
| Environment | none (no shape/dependency change) | — | — |

## Merge and migration order

1. This lands with defaults (`common_reward=False`, list reward, `{}` -> flag
   key added) — main stays runnable, no consumer forced to move.
2. Evaluation adopts the scalar + flag; training configs set
   `common_reward=True`.
3. No checkpoint exists yet, so there is no compatibility burden. Any
   checkpoint saved later must record `common_reward` alongside feature/action
   versions; a loader must reject a scalar-trained checkpoint under list mode.

## Validation and decision

- `tests_blue/test_epymarl_conformance.py`: 6 passed (sim venv; torch-tensor
  test runs in train venv, skips in sim).
- Full `tests_blue`: 34 passed, 1 skipped (sim venv); 6/6 conformance in train
  venv.
- Duck-type check against the pinned `multiagentenv.py`: 16/16 methods
  present; runner-style construction + `get_env_info()` verified.
- Train smoke (`blue_train_smoke.py`, train venv, seed 7629, 30 live steps):
  29 ticks, return -14.0, one finite masked update (loss -0.1556), temp
  checkpoint reloads with mask-legal actions. Plumbing only, not a policy.
- Affected-role review: pending (Evaluation for reward/flag semantics).
  Do not mark accepted just because this file exists.
