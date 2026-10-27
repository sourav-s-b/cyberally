# Cyberally: self-evolving cyber defense research

Shared student project for a CybORG/CAGE4 cyber-range, Red agents, Blue
defenders, and a measured adversarial retraining loop.

**Status:** research prototype. The CAGE4 wrapper, baseline policies, and a
local simulator-backed dashboard are available. No trained Blue policy,
real-world telemetry, or real-world deployment is claimed.

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

### Local CAGE4 dashboard

The dashboard UI is a React + Tailwind app in `frontend/`, built with Vite.
Build it once after checkout (Node.js 20+), then start the local-only API using
the training virtual environment (needed for PyTorch checkpoint
discovery/loading):

```sh
cd frontend
npm ci
npm run build
cd ..
./.venv-train/bin/python -m blue.dashboard_server
```

`npm run dev` serves the UI at <http://127.0.0.1:5173> with hot reload and
proxies `/api` to the dashboard server on port 8765. `frontend/node_modules/`
and `frontend/dist/` stay out of Git; commit source and `package-lock.json`.

Dashboard actor weights and verified dashboard models are bundled in
`blue/assets/dashboard-models.zip` (about 43 MB). The server unpacks missing
files into ignored `blue/results/` on first start, so clones need no separate
model download. Existing local model files are left untouched. PyTorch is
required to inspect and run neural checkpoints; simulator baselines remain
available without it.

On Windows PowerShell, use `.\.venv-train\Scripts\python.exe -m blue.dashboard_server`.
If `.venv-train` does not exist, set it up with the project's training and
simulator dependencies before launching; do not install them into the system Python.
For bundled learned models, install the scorer's pinned dependencies into that
same environment before starting the server:

```sh
./.venv-train/bin/python -m pip install -r blue/harness/requirements.txt
```

On Windows, use `.\.venv-train\Scripts\python.exe -m pip install -r blue\harness\requirements.txt`.
The bundled scorer requires `scikit-learn==1.7.2`. A missing `sklearn._loss`
module usually means these dependencies were installed into a different Python
environment or the scikit-learn install is incomplete. Simulator baselines do
not require this optional scorer dependency.
If the dashboard starts with a different interpreter, saved checkpoints are
reported as unavailable (a neutral notice) while the baselines remain usable.

Then open <http://127.0.0.1:8765>. The dashboard (React sources in `frontend/src/`,
built into `frontend/dist/`) launches
the existing CAGE4 Enterprise wrapper with a selected native Red variant and a
Blue baseline or locally saved `agent.th` actor chosen from a fixed, friendly
drop-down. It shows a zone-aware host graph with coloured pending/recent
Remove and Restore connectors, compact Red/Blue traces, a reward-over-steps
chart, speed control, a filterable Logs page and an interactive analyst page
backed only by Blue-visible decisions and observations. It records joint
simulator steps, Blue actions and native team reward. Privileged Red-session
counts are marked as evaluation-only
and are not policy inputs. The episode audit trail can be downloaded as SIEM
NDJSON (Elasticsearch `_bulk`); this is a dashboard audit export, not real host
telemetry. The server binds to loopback and serves only the built dashboard
assets. This is a CybORG simulator view, not a container range, packet monitor, or
real-host control panel.

Branch from the current `main` for new work. Historical branch and wrapper
notes below this document's current-state links may be stale; read
[docs/current-state.md](docs/current-state.md) and the role handoffs first.

## Repository layout

| Path | Purpose / primary owner |
|---|---|
| `cage-challenge-4/CybORG/` | Existing upstream simulator, unmodified; Environment owns team changes |
| `blue/` | Blue core wrapper, baseline policies, experiments and dashboard server; Blue |
| `frontend/` | React + Tailwind dashboard (Vite sources in `src/`, build in `dist/`); Blue |
| `blue/tests_blue/` | Blue live regression suite; Blue |
| `docs/archive/blue-agent-plan.md` | Detailed Blue design, audit and experiments |
| `docs/archive/handoff.md` | Historical Blue handoff; read correction at its top |
| `cage-challenge-4/team-guide.md` | **Superseded** original planning doc; read the banner before using |
| `red/`, `environment/`, `loop_eval/` | New team-owned integration work; scope files only so far |
| `docs/` | Shared contracts, roadmap, handoffs and references |
| `assets/Zeroth_main.pptx` | Original review deck; aspirational claims are not measured results |

Layout rule: `cage-challenge-4/` is the pristine simulator (do not add
role code there); Blue code lives in `blue/` (run it with cwd=`blue/`);
the Blue wrapper bridges to `CybORG` via a `__file__`-derived path insert.
Keep simulator source pristine and use that existing bridge.
Use one shared remote repository, separate clones/worktrees, and task branches.
Create role/task branches from the current `main`; do not switch another
contributor's branch in a shared worktree.

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
