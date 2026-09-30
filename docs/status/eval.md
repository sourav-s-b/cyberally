# Evaluation and loop handoff

> Shared blockers and two measured facts recorded by Blue on 2026-09-30; owner to
> confirm or correct. See [`../current-state.md`](../current-state.md).
>
> - **Do not branch from `main` yet.** The BLUE-01 wrapper fixes
>   (`7b76fc2`, `58b6cc7` on `blue/foundation`) are unmerged. `main` still has a
>   wrapper that truncates host observations and cannot reach `Restore`.
> - No pinned runtime exists (Python 3.10 agreed but not installed; current venv
>   is 3.12 with no torch). You can draft metric tests against fixtures now.
> - **`TrueStateWrapper` is broken against this CAGE4 snapshot.**
>   `SimulationController.get_true_state(info)` returns an `Observation` object,
>   not a dict, so `TrueStateWrapper.py:62`'s `.pop("success")` raises
>   `AttributeError`; and `import CybORG.Agents.Wrappers` pulls in Ray. Build the
>   privileged-label extractor on
>   `env.environment_controller.get_true_state(info).data`, which returns
>   `{host: {...}}` after dropping the `success` key. Under masked-random Blue
>   at seed 7629, true state reached 34 Red sessions by step 100 while
>   `host_ground_truth()` over `blue_agent_4`'s own observation view returned 0
>   for every host at every one of 399 steps — the two must never be merged into
>   one channel, and neither figure is a like-for-like ratio of the other.
> - Blue observation and action widths are **per agent** (17/51 host capacity),
>   so records must carry per-agent widths rather than one shared shape.
> - Measured reference points, seed 7629 masked-random, 400 steps: 44 ms per
>   joint step; 162/399 steps carry reward and the first ~135 do not; mission
>   phases `(134, 133, 133)`; episode ends natively at `steps-1` with
>   `terminated=True`, so `truncated` never fires on the default scenario.

- Owner: teammate to assign
- First branch: `eval/baseline-harness`
- Status: not started
- Completed: proposed metrics and logging contract documented.
- Validation: no unified evaluator or baseline suite executed.
- Blockers: ENV-01 runtime; coordinate Blue policy/reset interface for integration.
- Dependency commits/PRs: none yet.
- Contract changes: metric denominators, horizon, event schema and artifact manifest.
- Next: EVAL-01 fixture-based metric tests + Sleep evaluator; then paired baselines.
- Artifacts: no new evaluation report.
