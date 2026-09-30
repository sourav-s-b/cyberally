# Red handoff

> Shared blockers recorded by Blue on 2026-09-30; owner to confirm or correct.
> See [`../current-state.md`](../current-state.md).
>
> - **`main` is unblocked.** BLUE-01 wrapper fixes are merged; `main` and
>   `blue/foundation` are both at `f8a9deb`. Branch from `main` normally.
> - No pinned runtime exists (Python 3.10 agreed but not installed; current venv
>   is 3.12 with no torch). Draft the adapter and fixtures meanwhile.
> - Measured baseline for calibrating any strategy: at seed 7629 under
>   masked-random Blue, true-state Red reached 13 user-level and 21 root-level
>   sessions by step 100, and 17/27 by step 200. A `strategy_id` entry must be a
>   replayable policy/config with seed, version and preconditions — a stored
>   `action_sequence` is a diagnostic trace and may be invalid once defenses or
>   topology change.

- Owner: teammate to assign
- First branch: `red/strategy-adapter`
- Status: not started
- Completed: built-in scripted Red is available in CybORG; no team adapter yet.
- Validation: no new Red experiments run.
- Blockers: ENV-01 runtime and agreed scenario injection seam.
- Dependency commits/PRs: none yet.
- Contract changes: propose strategy factory/config and replayable pool entries.
- Next: RED-01 scripted adapter/config; then pinned multi-attacker variants.
- Artifacts: none; no LLM integration or external attack execution configured.
