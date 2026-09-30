# Role handoffs

Each role owns its file. Update with every meaningful PR/session handoff; use
issues/PRs for notification. Other roles inspect unmerged versions via `git show`
after fetch. Do not use a single shared diary that everyone edits concurrently.

Include status, branch, completed changes, tests actually run, blockers, exact
dependency SHAs/PRs, contract changes, next tasks, and external artifact locations.
Avoid a self-referential "this commit" hash; identify dependencies and let Git
history identify the status update's own commit.
