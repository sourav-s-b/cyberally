# Kickoff prompts for each teammate's coding assistant

Open the assistant in your own clone or worktree. These prompts do not authorize
external deployment or messaging. Replace placeholders with the real task/branch.

## Blue

> Work on Blue in this repository, branch `blue/foundation`. Read root AGENTS.md,
> docs/contracts.md, docs/implementation-plan.md, docs/status/blue.md, and the Blue
> plan/handoff correction. First inspect Git status and preserve existing work.
> Implement BLUE-01 as small tested fixes, starting with single-reset/reseeding and
> host coverage. Then repair pending actions, passive observations and remediation.
> Coordinate required Environment/Eval changes through a local proposal and role
> handoff. Do not start long training until the correctness acceptance checks pass.
> Report actual tests, outstanding blockers, and the next PR-sized task.

## Environment

> Work as Environment owner on `environment/reproducible-runtime`. Read AGENTS.md,
> environment/AGENTS.md, contracts, roadmap and your role status. Implement ENV-01:
> create a reproducible minimal isolated runtime for the existing CAGE4 copy, test
> reset/joint step, document exact install commands and provenance. Preserve Blue's
> file paths and observation semantics. Propose scenario factory hooks needed by
> Red and Blue with a fixture and migration note. Do not change topology/reward or
> upgrade every dependency together. Record tested behavior and compatibility.

## Red

> Work as Red owner on `red/strategy-adapter`. Read AGENTS.md, red/AGENTS.md, shared
> contracts, roadmap and role status. Implement RED-01 using the built-in scripted
> attacker first. Create a replayable configuration/factory and deterministic smoke
> scenario, with no privileged data exposed to Blue. Draft required injection hooks
> with Environment/Blue. Keep experiments inside the supplied simulator. Defer LLM
> integration until the scripted policy and evaluator agree on action/log contracts.

## Evaluation and retraining

> Work as Evaluation/Loop owner on `eval/baseline-harness`. Read AGENTS.md,
> loop_eval/AGENTS.md, contracts, roadmap and your role status. Implement EVAL-01:
> define and test episode/step metrics, seed suites and a JSONL evaluator beginning
> with Sleep. Keep privileged labels separate from actor observations. Specify
> timing units and unresolved incidents. Prepare adapters with Blue/Environment,
> not assumed APIs. Later compare frozen vs new-only vs mixed-strategy retraining
> using fresh on-policy rollouts. Record only measured results.

## Resume any role

> Resume <role> on <branch>. Read AGENTS.md and docs/status/<role>.md. Inspect local
> changes and fetch origin if configured. Summarize new main changes and exact
> dependency commits relevant to this task. Complete <task ID> within your ownership,
> update the role handoff and draft PR notes. Do not overwrite other contributors'
> work or claim that a local draft has notified them.
