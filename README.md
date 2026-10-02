# Cyberally: self-evolving cyber defense research

Shared student project for a CybORG/CAGE4 cyber-range, Red agents, Blue MARL
defenders, and a measured adversarial retraining loop.

**Status:** research prototype. Blue observation/masking/wrapper code exists and
20 regression tests pass, but it lives only on `blue/foundation`; no trained Blue
policy or real-world deployment is claimed. The first milestone is a reliable
simulation and evaluator.

## Start here

1. Human or coding assistant: read [AGENTS.md](AGENTS.md).
2. Read [docs/current-state.md](docs/current-state.md). It is the newest document
   here and holds the measured behaviour, the branch status and the traps.
3. Set up your clone and branch: [team workflow](docs/team-workflow.md).
4. Find your deliverables: [implementation plan](docs/implementation-plan.md).
5. Read [integration contracts](docs/contracts.md) before changing interfaces.
6. Copy the appropriate [agent kickoff prompt](docs/agent-prompts.md).
7. Update your role's file in [team status](docs/status/README.md) with each PR.

To resume Sourav's active Blue task in a new assistant conversation, read
[the current Blue session update](docs/status/blue-session.md) and paste its
resume prompt after the standard project instructions.

### Branch status

`main` and `blue/foundation` are both at `f8a9deb` and hold identical content.
Branch from `main` normally. Up to `05999fb`, `main` held a broken
`cage-challenge-4/cc4_epymarl_wrapper.py` that truncated host observations to 16
hosts, reset the shared simulator once per Blue agent and left `Restore`
unreachable; `blue/foundation` has since been fast-forwarded into `main`, so
those warnings are obsolete. Verify your clone with the commands in
[docs/current-state.md](docs/current-state.md) section 0.

## Repository layout

| Path | Purpose / primary owner |
|---|---|
| `cage-challenge-4/CybORG/` | Existing upstream simulator, unmodified; Environment owns team changes |
| `cage-challenge-4/blue_*.py`, `cc4_epymarl_wrapper.py` | Blue experiments and wrappers; Blue |
| `cage-challenge-4/tests_blue/` | Blue live regression suite; Blue |
| `cage-challenge-4/blue-agent-plan.md` | Detailed Blue design, audit and experiments |
| `cage-challenge-4/handoff.md` | Historical Blue handoff; read correction at its top |
| `cage-challenge-4/team-guide.md` | **Superseded** original planning doc; read the banner before using |
| `red/`, `environment/`, `loop_eval/` | New team-owned integration work; scope files only so far |
| `docs/` | Shared contracts, roadmap, handoffs and references |
| `assets/Zeroth_main.pptx` | Original review deck; aspirational claims are not measured results |

Layout rule: `cage-challenge-4/` is the pristine simulator (do not add
role code there); Blue code lives in `blue/` (run it with cwd=`blue/`);
only `blue/cc4_epymarl_wrapper.py` bridges to `CybORG` via a
`__file__`-derived path insert. Keep these locations to avoid breaking
imports.
Use one shared remote repository, separate clones/worktrees, and task branches.
The local initial setup starts Blue on `blue/foundation`; other teammates create
their task branches from the shared `main` after cloning.

## Execution environment

Git/documentation work can run on Windows. **Python 3.10 is the agreed
compatibility target** and is not yet installed; the existing local venv is
Python 3.12 with no `torch`, and `torch==2.2.0` has no cp312 wheel. The historical
Linux venv and the current global Python installation are not a reproducible
setup.

Environment's first task is a tested minimal dependency set and lock/install
instructions, split into a simulator-only profile and a separate training
profile that adds torch. Do not install all optional Ray, SB3, GNN and EPyMARL
stacks into a global interpreter or upgrade pins independently on each branch.
Training scripts and a unified run command are pending; see milestone M0 in the
plan.

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
