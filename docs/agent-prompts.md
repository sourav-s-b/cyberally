# Kickoff prompts for each teammate's coding assistant

Open the assistant in your own clone or worktree. These prompts do not authorize
external deployment or messaging. Replace placeholders with the real task/branch.

**Every prompt below starts the same way, for a reason:** `origin/main` and
`blue/foundation` currently differ in behaviour, not just in content. The `main`
copy of `blue/cc4_epymarl_wrapper.py` silently truncates host
observations to 16 hosts, resets the shared simulator five times per episode and
cannot reach `Restore`. Verify with
`git show origin/main:blue/cc4_epymarl_wrapper.py | Select-String max_hosts`
before you write code, and record which branch you based your work on.

---

## Blue

> Work on Blue in this repository, branch `blue/foundation`. Read root
> `AGENTS.md`, then `docs/current-state.md`, `docs/contracts.md`,
> `docs/implementation-plan.md`, `docs/status/blue.md`,
> `docs/coordination/blue-action-space.md`, and the correction at the top of
> `docs/archive/handoff.md`. BLUE-01 is implemented with 20 passing tests and
> **is merged into `main`** (both refs at `f8a9deb`), so branch from `main`
> normally and treat the earlier "main is broken" warnings as obsolete. Then
> implement BLUE-02 heuristic baselines (Sleep, built-in random,
> masked random, round-robin Analyse/Restore) and reproduce the historical
> step-200 seed-7629 numbers before any RL claim. Derive all shapes from
> `get_env_info()`; never hardcode 155/510/2550 or the superseded 160/800/50.
> Report actual tests, outstanding blockers and the next PR-sized task.

## Environment

> Work as Environment owner on `environment/reproducible-runtime`. Read
> `AGENTS.md`, `environment/AGENTS.md`, `docs/current-state.md`, contracts,
> roadmap and your role status. Implement ENV-01: install Python 3.10 (the
> documented target; the current local venv is 3.12 and no `torch==2.2.0` wheel
> exists for it), rebuild an isolated venv, and **split the dependency profile**
> into a simulator-only set and a separate training set that adds torch. Do not
> install ray, torch_geometric, sb3 or pygame-gui into the base profile.
> Document exact install commands, run the Blue regression suite on the fresh
> interpreter, and record upstream CAGE4 provenance. Blue host capacity is
> per agent — 17 for agents 0-3, 51 for agent 4 — so do not pin shared shape
> constants. Propose scenario factory hooks needed by Red and Blue with a
> fixture and a migration note. Preserve Blue's file paths and observation
> semantics. Record tested behavior and compatibility, not intent.

## Red

> Work as Red owner on `red/strategy-adapter`. Read `AGENTS.md`, `red/AGENTS.md`,
> `docs/current-state.md`, shared contracts, roadmap and role status. Your work
> is blocked until the pinned runtime exists; the `main` merge is done, so draft
> the design and fixtures meanwhile. Implement RED-01 using the built-in
> scripted attacker first. Create a replayable configuration/factory and
> deterministic smoke scenario, with **no privileged attacker state exposed to
> Blue actor inputs** — a stored action sequence is a diagnostic trace, not a
> replayable strategy, so version the policy/config with seed and preconditions.
> Draft required injection hooks with Environment/Blue. Keep experiments inside
> the supplied simulator. Defer LLM integration until the scripted policy and
> the evaluator agree on action and log contracts.

## Evaluation and retraining

> Work as Evaluation/Loop owner on `eval/baseline-harness`. Read `AGENTS.md`,
> `loop_eval/AGENTS.md`, `docs/current-state.md`, contracts, roadmap and your
> role status. Implement EVAL-01: define and test episode/step metrics, seed
> suites and a JSONL evaluator beginning with Sleep. Two concrete notes: the
> upstream `TrueStateWrapper` is **broken** against this CAGE4 snapshot
> (`get_true_state(info)` returns an `Observation`, not a dict, so its `.pop()`
> raises, and importing `CybORG.Agents.Wrappers` pulls in Ray) — build your
> privileged-label extractor on
> `env.environment_controller.get_true_state(info).data` instead. And Blue's
> observation and action widths are **per agent**, so records must carry
> per-agent widths rather than assuming one shared shape. Keep privileged
> labels separate from actor observations. Specify timing units and unresolved
> incidents. Prepare adapters with Blue/Environment, not assumed APIs. Later
> compare frozen vs new-only vs mixed-strategy retraining using fresh on-policy
> rollouts. Record only measured results.

## Resume any role

> Resume `<role>` on `<branch>`. Read `AGENTS.md`, `docs/current-state.md` and
> `docs/status/<role>.md`. Confirm which branch holds the current wrapper before
> assuming anything; `main` and `blue/foundation` differ behaviourally. Inspect
> local changes and fetch origin if configured. Summarize new main changes and
> exact dependency commits relevant to this task. Complete `<task ID>` within
> your ownership, update the role handoff and draft PR notes. Do not overwrite
> other contributors' work or claim that a local draft has notified them.
