# Blue session update

## Telemetry producer implementation (2026-10-01)

Branch `blue/telemetry-schema`, base/main dependency
`2f0cfb67ae1d3b2da6a1b1ccff28eff9e1a5c915`. Created this task branch from
main after its schema-only fast-forward. Existing staged/unstaged training work
and unrelated untracked files are preserved in the original checkout.
Publication uses a separate worktree to exclude that work. User explicitly
requested committing and pushing telemetry directly to main; role review
remains pending, so this does not mark the schema accepted.
Older sections' claimed current branch is historical, not this checkout's branch.

Delivered `blue_telemetry.py`: frozen Proc/Conn/File/Auth records, strict JSON
serialization/validation, source-field simulator ticks, simulated provenance,
and schema-only parity with all ten existing features. Optional wrapper
`get_telemetry(agent_id)` exports only that agent's local Blue view. Actor inputs,
masks and native simulator behavior stay unchanged. `blue_collect_telemetry.py`
writes JSONL plus reproducibility manifest, using schema records for features.
Native Red is active; data is unlabeled and not a verified benign training set.
The proposed contract now documents nullable fields, owner/name retention,
external availability and retained-row semantics. Repeated rows are not new events.

Validation: 18 focused telemetry tests passed; isolated main-based publication
suite 60 passed in 32.59 s. Earlier local full Blue suite: 67 passed, with
8 existing upstream warnings in 54.76 s (includes uncommitted MAPPO tests). Real tests cover all agents, live passive
events, Analyse, reset, feature/mask parity, artifact hashes and no-overwrite.
No long training, real-sensor experiment, SIEM integration or detector fitting.
Environment/Evaluation review remains pending; shared contract not accepted.

After timestamp validation/runtime-manifest refinements, focused tests again
passed (18). CLI smoke: seeds 8123/8124, 30 native steps, round-robin policy,
300 agent snapshots including reset/final views. Artifacts are ignored under
`runs/telemetry-smoke-20261001/` (`telemetry.jsonl`, `manifest.json`).
Final `git diff --check` passed. Existing Git staging was preserved.

PR checklist: producer additive, old consumers unchanged, schema versions recorded,
no privileged labels in collector, no upstream edits or tracked runtime artifacts.
Future consumers migrate after affected-role review; old checkpoints remain
compatible because existing policy features/actions did not change. No teammate
messages or review requests sent. Existing training files/index remain untouched.

Resume: inspect telemetry diff and test results; obtain Environment field-shape
and Evaluation JSONL/time-unit review before merge. Then implement a separate
anomaly experiment with train/validation episode splits and clearly separated
evaluation labels. Example collection command lives in the schema proposal.

**Last updated:** 2026-09-30

**Owner:** Sourav

**Working branch:** `blue` (singular Blue branch; task branches consolidated
and deleted local+remote, `origin/blue` tracks it). Head carries BLUE-02,
BLUE-03 conformance, the docs handoff, and foundation-v2 below.
**Base commit:** `e605ef3` (`chore: import CAGE4 prototype and establish team workflow`)
**Merged in:** `origin/main` at `f8a9deb` — `blue/foundation` was fast-forwarded
into `main`, so both refs are identical and Environment, Red and Evaluation can
branch from `main`.
**Current state:** BLUE-01 foundation is complete as `7b76fc2` plus `58b6cc7`,
merged into `main` at `f8a9deb`/`d05d043`. Landed since, both pushed, both
unmerged: BLUE-02 baselines as `4471c27` on `blue/baseline-policies`
(`blue_baselines.py` + `tests_blue/test_baselines.py`, 9 tests) and BLUE-03
runner conformance as `3b414b4` on `blue/epymarl-conformance` (stacked).
Full suite: 34 passed, 1 skipped (torch-only test, runs in train venv).
Seed-7629 step-200 numbers recomputed under the fixed wrapper (see below);
the historical broken-wrapper targets are superseded.

Read `docs/current-state.md` before this file. It carries the measured facts and
the trap list that everything below depends on.

## Goal

Build a reliable CAGE4 Blue defender, establish strong baselines, then train and
evaluate MAPPO and attacker adaptation. Read root `AGENTS.md`,
`docs/implementation-plan.md`, `docs/contracts.md`,
`cage-challenge-4/blue-agent-plan.md`, and the correction at the top of
`cage-challenge-4/handoff.md` before changing code.

## Current Blue changes

Committed in `7b76fc2` and `58b6cc7` on `blue/foundation`:

- Reworked `cage-challenge-4/cc4_epymarl_wrapper.py`: reset the shared simulator
  once, apply explicit seeds and advance the RNG when none is supplied, reject
  host overflow instead of truncating, track exactly one pending action per
  agent, merge observations for every visible host every tick, keep simulator
  validity masks separate from opt-in evidence masks, and distinguish native
  termination from horizon truncation.
- Capacity raised to the scenario-derived 51 hosts. At seed 7629 agent 4 sees 38
  hosts, so the current dimensions are 510 local observation features, 155
  discrete actions, 2550 concatenated state.
- `CC4BlueWrapper` is a single-agent facade over the same joint-step
  implementation rather than a parallel copy of the bookkeeping.
- Reworked `cage-challenge-4/blue_action_masking.py`: pending actions no longer
  overwrite belief, `Restore` stays reachable in validity mode, and successful
  remediation transitions to `VERIFY` instead of declaring the host clean.
- Added `cage-challenge-4/tests_blue/test_foundation.py`.
- Added live tests for lost target sessions (Analyse/Remove/Restore returning
  `FALSE` when the child session disappears mid-action, with the original target
  still attributed and no clean label inferred), overdue-action failure, reset
  clearing unresolved pending state, and passive Monitor event replacement with
  timestamp advancement.

Drafted this session, not committed to any contract yet:

- `docs/current-state.md` — measured facts, branch topology, trap list.
- `docs/coordination/blue-action-space.md` — per-agent host bounds proposal.

## Verification already completed

- From `cage-challenge-4`: `..\.venv\Scripts\python.exe -m pytest -q tests_blue`
  — **20 passed** (12.7 s).
- Three masked-random episodes (seeds 42, 7629, 7630), 75 steps each —
  completed; per-agent common rewards matched and action masks held.
- `.venv\Scripts\python.exe scripts\check_source.py` from repo root — 163
  tracked Python files parsed.
- Measured this session, masked-random policy, seed 7629 unless noted:
  - 44 ms per joint step, 0.19 s reset, ~18 s per 400-step episode.
  - True-state Red sessions via `get_true_state(info).data`, host-wide: 1 at t0,
    13 user + 21 root at step 100, 17 + 27 at step 200, 18 + 15 at step 300.
  - `host_ground_truth()` over `blue_agent_4`'s own observation view: 0 for every
    host at every one of 399 steps, across seeds 7629, 7630 and 7640. Note the
    true-state count is host-wide and the Blue-view count covers agent 4's hosts
    only, so they are not a like-for-like ratio.
  - 162 of 399 steps carry nonzero reward; the first ~135 do not. Mission phases
    are `(134, 133, 133)`; phase 0 is pre-planning.
  - 1374 of 1995 agent-ticks had exactly one legal action (Sleep) — 69 %.
  - Live nonzero features within a single agent's observation: 22–39 %.
  - Per-agent host counts across seeds 7629–7660: agents 0–3 hold 7–14 hosts
    with exactly 1 router each; agent 4 holds 29–38 with 3 routers. Scenario
    bound is 17 per subnet, 51 for agent 4.
  - `CybORG/Agents/Wrappers/TrueStateWrapper.py:62` raises `AttributeError`
    against this snapshot because `get_true_state(info)` returns an
    `Observation`, not a dict; `CybORG.Agents.Wrappers` also imports Ray.
- Not done: no MAPPO training, no EPyMARL integration, no heuristic benchmark,
  no historical baseline reproduction, no upstream test-suite run.

## Next steps

1. ~~Torch blocker (resolved 2026-09-30, pending Environment review).~~
   `.venv-train/` rebuilt with uv 0.12.5 against the system Python 3.12.10:
   `torch==2.14.1+cpu` (PyTorch CPU index) + sim pins identical to `.venv/`
   (numpy 1.26.4, gym 0.26.2). Verified: torch import/matmul, Linear+Adam
   optimizer step, CybORG import in the same process, all 29 Blue tests green
   under `.venv-train`. uv verdict: yes for 3.12 env management (venv+install
   in seconds, `uv pip` resolves the draft pins cleanly), but uv cannot
   deliver 3.10 here — `uv python install 3.10` downloads yet never executes
   (WinError 4551, Application Control; only system 3.12 runs, and torch 2.2.0
   has no cp312 wheel). So the documented 3.10 target needs an IT-approved
   interpreter install, not a tooling change. Drafted pins:
   `environment/requirements-{sim,train}.txt` (uv-native rebuild commands in
   the train file's header); `.gitignore` covers `.venv-train/`. Note:
   uv venvs ship without pip — manage `.venv-train` via
   `uv pip --python .venv-train\Scripts\python.exe`. `environment/setup_uv.ps1`
   now rebuilds + verifies both venvs from scratch (29/29 suites each).

## BLUE-03 progress (`3b414b4` on `blue/epymarl-conformance`, pushed)

- EPyMARL pinned at `cbc38c09` (`uoe-agents/epymarl` HEAD 2026-09-30, shallow
  clone in machine-temp, not in Git). EPyMARL itself is NOT installed; no
  install was needed for conformance.
- `CC4MARLEnv` now accepts the runner's `common_reward`/`reward_scalarisation`
  kwargs (native reward is already one shared signal, so the scalar is
  `rewards[0]`, never a 5x sum), returns `info={"episode_limit": ...}` (True
  only at the horizon end, for critic bootstrapping), and adds `close()`,
  `render()`, `save_replay()` (raises), `get_stats()` plus a `seed()` setter
  that applies on the next reset (found + fixed: bare `reset()` was ignoring a
  `seed()`-set value instead of applying it once). All additive; defaults
  preserve foundation-v1 behavior. `self.seed` attr renamed `self._seed`.
- Verified 16/16 `MultiAgentEnv` ABC methods present (duck-typed, checked
  against the pinned tree) and `get_env_info()` matches the base shape.
- Train smoke (`blue_train_smoke.py`, train venv, seed 7629, 30 steps, live
  Red): 29 ticks, return -14.0, one finite masked-REINFORCE update
  (loss -0.1556), checkpoint to temp reloaded with mask-legal actions over a
  fresh rollout. Smoke plumbing only, not a trained policy.
- Still open: EPyMARL install (sacred/torchvision stack, Windows risk) and the
  first real MAPPO rollout-plus-update inside their runner.
2. Ask Environment to confirm the per-agent host bounds of 17 and 51 before
   they write any config, and get Evaluation's agreement on the `VERIFY`
   resolution rule using Blue-visible evidence only.
3. Whole-agent primary-session loss: new evidence weakens this. A 200-step
   forced-Sleep breakdown (seed 7629, native Red) found 676/676 forced ticks
   were pending-lockout with zero session losses, so session loss is at most
   rare under these conditions — keep the Environment question open but stop
   treating it as a likely blocker.
4. Get the three stacked PRs reviewed and merged branch-by-branch
   (`blue/baseline-policies`, then `blue/epymarl-conformance` with its
   contract proposal, then this docs handoff). Extend the seed suite beyond
   the 3-seed snapshot below (≥30-eval-seed suite per the Blue plan) before
   any RL comparison.
5. BLUE-03 remainder: install EPyMARL (sacred/torchvision stack, Windows risk)
   into `.venv-train` and run the first real MAPPO rollout-plus-update inside
   their runner, on top of the conformance already landed. No long run before
   that smoke passes.

## BLUE-02 recomputed numbers (this session, fixed wrapper)

Seed 7629, `CC4MARLEnv(steps=400)`, counts are privileged hosts-with-Red-session
(total / root) via `env.env.get_true_state(env.env.INFO_DICT['True']).data`:

| Policy | @100 | @200 | @300 | final | return |
|---|---|---|---|---|---|
| Sleep | 44 / 26 (0.0) | **64 / 54 (-60)** | 69 / 58 (-564) | 71 / 59 | -1821 |
| Masked-random (policy seed 0) | 37 / 22 (-55) | **33 / 25 (-102)** | 38 / 23 (-155) | 36 / 17 | -205 |
| Round-robin heuristic | 26 / 11 (-8) | **38 / 11 (-65)** | 46 / 8 (-74) | 50 / 9 | -85 |

Historical broken-wrapper targets (Sleep 60/43, random 48/35) were produced
before the reset/host-coverage fixes and must not be compared against these.
Masked-random ran under one policy seed only. Correction found while writing
the evaluator: the privileged path on `CC4MARLEnv` is `env.env`, not
`env.environment_controller`; `docs/current-state.md` trap 3 now says so and
`count_compromised()` is the reference.

## foundation-v2 per-agent bounds (this session, on `blue`)

`CC4MARLEnv(per_agent_bounds=True)`: caps `[17, 17, 17, 17, 51]` derived from
the scenario's own subnet counts, giving actions `[53, 53, 53, 53, 155]`,
obs `[170, 170, 170, 170, 510]`, critic state 1190. Host order and indices
are untouched, so masked-random trajectories are identical across modes
(tested). Out-of-agent-range indices coerce to Sleep instead of raising;
`get_env_info()` gains `*_per_agent` keys plus version. Default mode is
byte-identical v1 behavior and stock EPyMARL still requires it (one shared
head cannot span 53- and 155-wide agents). `CC4BlueWrapper` reports its own
agent's widths; `run_episode`/`evaluate_policies` take env kwargs;
`WRAPPER_VERSION` is now `foundation-v2`. Proposal step 1 of 3 — Environment
and Evaluation have not reviewed; no consumer migrated yet.

## Multi-seed suite (this session, `evaluate_policies`, seeds 7629/7630/7640)

Same setup as above; masked-random policy seed fixed at 0, fresh policy per
episode. @200 columns are total / root / return; final column adds full return:

| Policy | 7629 @200 | 7630 @200 | 7640 @200 | finals (total/root/return) |
|---|---|---|---|---|
| Sleep | 64 / 54 / -60 | 55 / 49 / -242 | 58 / 52 / -553 | 71/59/-1821, 64/63/-1537, 63/59/-4138 |
| Masked-random | 33 / 25 / -102 | 35 / 19 / -126 | 26 / 15 / -131 | 36/17/-205, 20/13/-231, 28/15/-233 |
| Round-robin | 38 / 11 / -65 | 25 / 13 / -65 | 30 / 9 / -99 | 50/9/-85, 32/11/-105, 36/10/-123 |

Paired reading: round-robin halves root compromise vs masked-random on every
seed (25→11, 19→13, 15→9) with equal-or-better return at @200 and at final.
Sleep collapses late (returns in the thousands negative). Still only 3 seeds —
the ≥30-seed held-out suite from the Blue plan is Evaluation's build, not this.

## Resume prompt for a new conversation

> Continue the Blue defender work in `G:\Projects\cyberally`. Read `AGENTS.md`,
> `docs/current-state.md`, `docs/status/blue-session.md`, `docs/status/blue.md`,
> `docs/contracts.md`, `docs/implementation-plan.md`,
> `docs/coordination/blue-action-space.md`, `cage-challenge-4/blue-agent-plan.md`,
> and the correction at the top of `cage-challenge-4/handoff.md`. Preserve my
> work: inspect `git status --short --branch` and fetch origin first. Blue
> work lives on singular branch `blue` (tracks `origin/blue`); the
> `environment/` uv drafts stay uncommitted for Environment. Do not reset or
> rewrite shared history. Next tasks: open the PR from `blue`, get the
> action-space proposal reviewed (steps 2–3), then EPyMARL install + first
> real MAPPO rollout in their runner. Do not start long training before that
> smoke passes, and never report an unreproduced number as a result.

## Update rule

At each meaningful Blue handoff, replace "Current Blue changes," "Verification"
and "Next steps" with the latest state. Include branch and base commit, whether
changes are staged/committed/pushed, exact commands and results, open contracts
and the next task. Update `docs/status/blue.md` as the short index. Never report
a run that did not actually complete.
