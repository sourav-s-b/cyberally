# Cyberally: self-evolving cyber defense research

Shared student project for a CybORG/CAGE4 cyber-range, Red agents, Blue MARL
defenders, and a measured adversarial retraining loop.

**Status:** research prototype. Blue observation/masking/wrapper code exists but
has known correctness blockers. No trained Blue policy or real-world deployment
is claimed. The first milestone is a reliable simulation and evaluator.

## Start here

1. Human or coding assistant: read [AGENTS.md](AGENTS.md).
2. Set up your clone and branch: [team workflow](docs/team-workflow.md).
3. Find your deliverables: [implementation plan](docs/implementation-plan.md).
4. Read [integration contracts](docs/contracts.md) before changing interfaces.
5. Copy the appropriate [agent kickoff prompt](docs/agent-prompts.md).
6. Update your role's file in [team status](docs/status/README.md) with each PR.

To resume Sourav's active Blue task in a new assistant conversation, read
[the current Blue session update](docs/status/blue-session.md) and paste its
resume prompt after the standard project instructions.

## Repository layout

| Path | Purpose / primary owner |
|---|---|
| `cage-challenge-4/CybORG/` | Existing upstream simulator; Environment owns team modifications |
| `cage-challenge-4/blue_*.py`, `cc4_epymarl_wrapper.py` | Existing Blue experiments and wrappers; Blue |
| `cage-challenge-4/blue-agent-plan.md` | Detailed Blue design, audit and experiments |
| `cage-challenge-4/handoff.md` | Historical Blue handoff; read correction at its top |
| `red/`, `environment/`, `loop_eval/` | New team-owned integration work; scoped instructions in each |
| `docs/` | Shared contracts, roadmap, handoffs and references |
| `Zeroth_main.pptx` | Original review deck; aspirational claims are not measured results |

Keep the current simulator and Blue file locations to avoid breaking imports.
Use one shared remote repository, separate clones/worktrees, and task branches.
The local initial setup starts Blue on `blue/foundation`; other teammates create
their task branches from the shared `main` after cloning.

## Execution environment

Git/documentation work can run on Windows. Python 3.10 in a dedicated virtual
environment is the initial compatibility target, based on the existing 2024-era
dependency pins. It is not yet a validated installation. The historical Linux
venv and the current global Python installation are not a reproducible setup.

Environment's first task is a tested minimal dependency set and lock/install
instructions. Do not install all optional Ray, SB3, GNN and EPyMARL stacks into a
global interpreter or upgrade pins independently on each branch. Training scripts
and a unified run command are pending; see milestone M0 in the plan.

## Source and licensing

`cage-challenge-4/` contains a local copy of
[CAGE Challenge 4](https://github.com/cage-challenge/cage-challenge-4).
Its original [license](cage-challenge-4/LICENSE.txt) and notices are preserved.
The source snapshot's exact upstream commit has not been established; do not
claim that it tracks upstream HEAD. No upstream Git history was present locally.

Local third-party paper PDFs and generated extracts are excluded from Git;
[references](docs/references.md) link to the sources. The team's review deck is
included. Project-created code licensing remains a team decision before any
public release; the upstream license continues to apply to upstream code.
