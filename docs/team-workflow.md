# Shared Git and coordination workflow

## One repository, separate workspaces

Create one private GitHub repository named `cyberally` under the project owner or
a team organization. Invite all teammates as collaborators. Everyone clones it
and works on a task branch. Separate repositories/forks are optional, not required.
This is GitHub's branch/PR model: https://docs.github.com/en/get-started/using-github/github-flow

The current local project uses the workspace root as the Git root, including the
existing `cage-challenge-4/` folder. Do not initialize another repository inside it.
The bootstrap source import has no prior upstream history. Do not add the upstream
CAGE repository as origin or pull its root into this differently structured repo.

## Owner: first remote setup

On GitHub, create an **empty private repository**. Do not initialize it with a
README, license or .gitignore because the local repository already contains them.
Copy the HTTPS URL and substitute it below. Run from the Cyberally root:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/cyberally.git
git remote -v
git push -u origin main
git push -u origin blue/foundation
```

If `origin` already exists, inspect it first; do not blindly replace a teammate's
remote. Git Credential Manager is available on the owner's machine and may open
a login prompt on the first push. Never place a token in the URL or a tracked file.
Creating this document does not create a GitHub repository or grant access.

After the first push:

1. Invite collaborators through repository Settings -> Collaborators/access.
2. Keep `main` as the default branch.
3. Configure main protection/rules where the account plan supports it: PR required,
   one reviewer, resolved conversations, no force pushes, required `source-check`
   after that workflow has run. If unavailable, follow the same rules manually.
4. Do not populate CODEOWNERS with guessed usernames; assign real role owners first.
5. Enable Issues; use an issue per task/dependency and a draft PR for early visibility.

## Teammates: clone and start

Use your own identity, not Sourav's. Replace the URL and identity placeholders:

```powershell
git clone https://github.com/YOUR_USERNAME/cyberally.git
cd cyberally
git config user.name "Your Name"
git config user.email "YOUR_VERIFIED_OR_GITHUB_NOREPLY_EMAIL"
git switch -c red/strategy-adapter
```

Choose the appropriate branch:

| Role | First branch | Handoff |
|---|---|---|
| Blue (Sourav) | `blue/foundation` | `docs/status/blue.md` |
| Environment | `environment/reproducible-runtime` | `docs/status/environment.md` |
| Red | `red/strategy-adapter` | `docs/status/red.md` |
| Evaluation/loop | `eval/baseline-harness` | `docs/status/eval.md` |

Create short-lived branches for subsequent tasks from updated main. A role prefix
is ownership metadata, not a reason to keep a months-long unmerged branch.

## Daily start and sync

```powershell
git status --short --branch
git fetch origin --prune
git log --oneline --max-count=12 origin/main
git diff --stat HEAD...origin/main
```

Read contracts and other roles' latest merged status. Fetch only downloads refs;
it does not change your files. If your working tree is clean, merge main into
your task branch before starting integration work:

```powershell
git merge origin/main
```

If dirty, first review and commit your work or use another worktree; do not discard
it. Merge conflicts require a semantic resolution and focused checks. Use
`git merge --abort` to cancel an unresolved merge when needed. Avoid force pushes.

## Inspect a teammate's unmerged update

```powershell
git fetch origin --prune
git show origin/red/strategy-adapter:docs/status/red.md
git diff --stat origin/main...origin/red/strategy-adapter
git log --oneline origin/main..origin/red/strategy-adapter
```

These refs exist only after the teammate pushes. For a combined test, create a
temporary integration branch in a separate worktree, starting at origin/main:

```powershell
git worktree add ../cyberally-integration -b integration/blue-red origin/main
cd ../cyberally-integration
git merge origin/blue/foundation
git merge origin/red/strategy-adapter
```

Record the exact merged SHAs in the test report. The integration branch is for
testing; it does not replace reviewed PRs from each source branch. Avoid merging
large unrelated role branches into your feature branch or cherry-picking entire
features, which obscures dependency ownership.

## Save and submit a change

Review `git diff` and use explicit paths with `git add` (or interactive `git add -p`).
Update your role status. Then:

```powershell
git diff --cached --stat
git diff --cached --check
git commit -m "blue: preserve pending action identity"
git push -u origin HEAD
```

Open a PR targeting main in GitHub and complete the template. Include task/issue,
contract versions, exact dependent PRs/commits, tests, and consumer migration.
An affected teammate reviews shared-interface changes; an available teammate
reviews isolated changes. Merge when checks/review pass, then teammates fetch
and merge main. Use `git pull --ff-only` when updating a clean local main.

## Communication protocol

Branches hold code, not a message bus. Status files become visible remotely only
after a push; issues/PRs create the discussion and notification trail.

For cross-role changes:

1. Producer drafts `docs/coordination/<role>-<topic>.md` from the template and opens
   an issue/PR describing the current behavior and proposed replacement.
2. List affected roles, exact fields/types/semantics, compatibility period and tests.
3. Consumers identify required changes in their own role status/linked tasks.
4. Merge additive producer support first, then consumers, then remove old support
   in an explicitly breaking change. Keep main runnable between merges.
5. At session end, record completed work, test results, blockers and next tasks.
   If an agent cannot access GitHub, its human posts the prepared update.

Example: adding `observation_age` is proposed by Blue; Environment confirms its
source, Evaluation records the feature version, and checkpoint loading rejects
incompatible dimensions. A mask/reward/done change needs an explicit semantics
review even if the Python signature is unchanged.

Urgent blockers belong in a linked issue and role handoff, not only private chat.
Use short team syncs to resolve decisions; record outcomes in the contract/PR so
other coding sessions can recover context. No automatic cross-branch messaging
service or bot has been installed by this setup.

## Agent sessions and simulation deployment

Open one assistant in your clone/worktree; give it a prompt from
`docs/agent-prompts.md`. A Markdown instruction file supplies context; it does not
launch a coding agent, train a model or deploy a policy by itself.

For policy handoff, provide a checkpoint outside Git plus a committed artifact
manifest: source SHA, config, contract/feature/action versions, hash, seed suite,
evaluation results and load command. A recipient first runs the fixed smoke suite
in the isolated simulator. Actual loader/CLI implementation is in the roadmap.
