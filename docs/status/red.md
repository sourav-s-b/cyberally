# Red handoff

## RED-01 continuation, 2026-10-01

- Branch: `red/strategy-adapter`, isolated worktree
  `C:/Projects/cyberally/runs/red-strategy-adapter`.
- Exact dependency: main/telemetry
  `ad9e5c0423f42fc8970e4ffe1051c1cb47a9530c`.
- Delivered: strict versioned native DiscoveryFSRed configuration, policy-class
  adapter preserving native RNG/constructor, standalone simulator smoke factory,
  source/config provenance and Red-only diagnostic CLI. No Blue or simulator
  files modified. The shared injection/manifest contract is proposed, not accepted.
- Runtime: existing local Python 3.12.14 environment runs CybORG successfully;
  older no-runtime statements below are superseded for this machine. Environment
  dependency/runtime review remains open, not a blocker on local smoke tests.
- Tests: combined suite **75 passed in 46.56 s** (15 Red + 60 Blue), including
  native parity at seeds 7629/7630, fresh instances/repeated reset, invalid config
  rejection, manifest hashes, episode boundaries and refusal to overwrite outputs.
- CLI smoke: `python -m red.smoke --output runs/red-smoke-20261001 --seeds 7629
  7630 --steps 40` with this worktree's simulator on PYTHONPATH. Both episodes
  executed 39 native joint ticks, native Blue team return 0.0 and zero invalid
  executed Red actions. This short result does not establish attack effectiveness.
  Ignored artifacts: `runs/red-smoke-20261001/manifest.json`, `red-actions.jsonl`.
- Skipped: long runs/training, full effectiveness metrics, alternate attackers,
  shared-factory consumer integration and upstream optional-framework suite;
  these exceed the native scripted baseline scope. No new dependencies installed.
- Limitations: only the built-in baseline, Sleep Blue, native Green/full topology;
  no training integration, attacker-strength claim, privileged labels or real attacks.
- Next: affected-role review of `docs/coordination/red-strategy-factory.md`, then
  Environment's shared factory and Blue/Evaluation opt-in integration. Native
  parity remains the default regression gate before adding variants/pool sampling.
- PR checklist: preserve upstream attribution; policy/config replayability rather
  than action-sequence replay; explicit seed/horizon/source hashes; diagnostics
  isolated from Blue inputs; no schema accepted or teammates notified by local docs.
- Publication: Red changes committed/pushed only to `red/strategy-adapter`;
  main contains telemetry `ad9e5c0`, no Red implementation. Original dirty Blue
  checkout and its staged training work were preserved. No PR/review messages sent.

Earlier status below is historical and does not describe the current task.

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
