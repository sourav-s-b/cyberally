# Environment handoff

> Shared blockers recorded by Blue on 2026-09-30; owner to confirm or correct.
> See [`../current-state.md`](../current-state.md).
>
> - **`main` is unblocked.** BLUE-01 wrapper fixes are merged; `main` and
>   `blue/foundation` are both at `f8a9deb`. Branch from `main` and verify with
>   `cage-challenge-4/tests_blue/` before starting.
> - The agreed target is **Python 3.10**, not installed. The local venv is 3.12
>   with no `torch`; `torch==2.2.0` has no cp312 wheel. The historical Linux venv
>   is behind the current Windows work and should not be treated as the
>   reference.
> - Blue host capacity is **per agent**: 17 for `blue_agent_0..3`, 51 for
>   `blue_agent_4` (agent 4 owns public_access_zone + admin_network +
>   office_network; the others own one subnet each). Do not pin a shared shape
>   constant such as 155/510/2550. See
>   [`../coordination/blue-action-space.md`](../coordination/blue-action-space.md).
> - Split the dependency profile: a simulator-only set for Red/Eval, and a
>   separate training set that adds torch. `CybORG/Tests/test_cc4/test_blue_actions.py`
>   cannot collect without Ray; that is a missing optional dependency, not a
>   code failure.

- Owner: teammate to assign
- First branch: `environment/reproducible-runtime`
- Status: not started
- Completed: inherited CAGE4 source exists; exact upstream revision unknown.
- Validation: no fresh environment install validated in this setup.
- Blockers: agree minimal runtime profile and Python/platform target.
- Dependency commits/PRs: none yet.
- Contract changes: coordinate scenario factory, reset/step semantics and Red hooks.
- Next: ENV-01 isolated install + seeded reset/joint-step smoke test; record commands.
- Artifacts: none.
