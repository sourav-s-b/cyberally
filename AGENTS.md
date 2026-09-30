# Instructions for human contributors and coding agents

## Purpose and source of truth

Build and evaluate a simulation-based autonomous defense system using CAGE4:
Environment + Red + Blue + retraining/evaluation. Work within your assigned role.
Do not treat the review slides' performance targets as achieved results.

Read at session start:

1. This file and any AGENTS.md in the area you will edit.
2. `docs/team-workflow.md` and `docs/contracts.md`.
3. `docs/implementation-plan.md` and `docs/status/<your-role>.md`.
4. Blue work: `cage-challenge-4/blue-agent-plan.md` and the correction at the
   beginning of `cage-challenge-4/handoff.md`.

Current instructions and accepted contracts take priority over old handoff text.
User instructions take priority over this file. Contracts marked proposed are
not implemented APIs; do not silently treat them as existing functionality.

## Branch and workspace discipline

- Inspect `git status --short --branch`, branch name, and recent commits first.
  Preserve unrelated/uncommitted user work. Do not auto-stash or reset it.
- Use `<role>/<task>`: `blue/foundation`, `red/strategy-adapter`,
  `environment/reproducible-runtime`, `eval/baseline-harness`.
- The initial bootstrap commit on main is intentional. Subsequent development
  belongs on task branches; create a branch before editing if necessary.
- Each concurrent coding session needs its own clone or Git worktree. Switching
  branches in a shared folder changes the files for every agent using it.
- Never force-push shared history or merge another person's work without review.
  Normal integration is task branch -> reviewed PR -> main -> teammates' branches.
- Do not automatically spawn other agents. Team roles here describe separate
  contributors; parallel agent delegation requires a user request.

## Ownership and interfaces

- Blue owns existing root-level `blue_*.py`, `cc4_epymarl_wrapper.py`, Blue tests,
  training configs and Blue model code under `cage-challenge-4/`.
- Environment owns `environment/`, simulator changes, dependency/runtime setup.
- Red owns `red/` and Red adapters. Propose required simulator hooks jointly with
  Environment instead of independently modifying core simulator behavior.
- Evaluation owns `loop_eval/`, metric definitions, scenario suites and retraining.
- Shared contracts, schemas, rewards, topology and dependency changes need a
  concrete proposal plus review by affected roles before merging. Prepare the
  proposed diff and tests on your branch; this rule does not block local drafting.
- Do not rewrite shared schemas or rename existing simulator agents/hosts just to
  match a diagram. Version a breaking change and supply migration instructions.

## Communicating across branches

Branches do not exchange messages automatically. Use the workflow in
`docs/team-workflow.md`: GitHub issues for dependencies/decisions, PRs for concrete
changes, and committed role status files for session handoffs.

- At start, fetch origin if configured and inspect main changes. If unavailable,
  continue independent local work and report that remote status is unknown.
- Read dependencies at their exact commit, using `git show origin/<branch>:<path>`
  or another worktree. Do not checkout another role's branch over ongoing work.
- Put contract proposals in `docs/coordination/` using the template. Every proposal
  names producer, consumers, changed fields, defaults, tests and rollout order.
- Update only your own role status by default. Include your branch, deliverable,
  tests actually run, blockers and exact dependency commits. A branch name alone
  is a moving reference and is not sufficient to identify a tested dependency.
- Agents may draft local issue/PR text. Sending comments/messages or requesting
  reviews on teammates' behalf requires the user's authorization for that action.
  Do not imply that a local status file has notified a teammate.

## Engineering and research rules

- Fix simulator/wrapper correctness before expensive training. Test the behavior
  that matters: seeding, shared reset, host coverage, pending-action attribution,
  passive observations, masks, remediation, rewards and episode boundaries.
- Only Blue-visible observations enter actors, masks and anomaly detectors.
  Privileged simulator state is restricted to clearly separated evaluation labels.
- A successful Remove execution does not prove the host is clean. Record action
  completion, observed verification and privileged evaluation as distinct facts.
- Keep simulator-validity masks separate from evidence-based policy restrictions.
- Include Sleep, random and a strong heuristic before asserting an RL improvement.
  Reproduce historical baseline numbers rather than reporting them as fresh tests.
- Training/evaluation seeds and attacker suites must be separated. Save config,
  source commit, preprocessing version, artifact hash and actual seed lists.
- For PPO/MAPPO retraining, collect fresh rollouts against old/new scenarios;
  do not naively replay old transitions into an on-policy update.
- Model weights, runtime logs, credentials, virtualenvs and downloaded tools stay
  outside Git. Commit small config files and artifact manifests instead.
- Retain upstream license/attribution. Record pinned revisions for external code.
- No external deployment, remote experiment launch or live-network action is
  implied by the phrase "deploy an agent" here. Initial deployment means loading
  and evaluating a policy in the isolated project simulator.

## Completing a task

Run checks appropriate to the change. State skipped checks and why; do not invent
passed tests or make long training a default CI step. Inspect the final diff and
stage only intended files. When asked to commit/push, use a specific change title.
Include the role handoff and PR checklist, with risks and dependent branch changes.
End with what changed, validation, current branch and the next concrete task.
