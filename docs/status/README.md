# Role handoffs

Each role owns its file. Update with every meaningful PR/session handoff; use
issues/PRs for notification. Other roles inspect unmerged versions via `git show`
after fetch. Do not use a single shared diary that everyone edits concurrently.

Include status, branch, completed changes, tests actually run, blockers, exact
dependency SHAs/PRs, contract changes, next tasks, and external artifact locations.
Avoid a self-referential "this commit" hash; identify dependencies and let Git
history identify the status update's own commit.

Read [`../current-state.md`](../current-state.md) before any role file here. It
records the measured behaviour of the simulator, which branch currently holds a
working wrapper, and the traps that will otherwise cost an assistant an afternoon.

| Role | File | First branch |
|---|---|---|
| Blue | [blue.md](blue.md), detailed in [blue-session.md](blue-session.md) | `blue/foundation` |
| Environment | [environment.md](environment.md) | `environment/reproducible-runtime` |
| Red | [red.md](red.md) | `red/strategy-adapter` |
| Evaluation and retraining | [eval.md](eval.md) | `eval/baseline-harness` |

Environment, Red and Evaluation are all recorded as **blocked** on two shared
dependencies: the BLUE-01 wrapper fixes are not yet merged into `main`, and no
pinned runtime exists. See `docs/current-state.md` sections 0 and 5.
