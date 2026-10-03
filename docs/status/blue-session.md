# Blue session update

## Research audit (2026-10-03)

Branch `blue/mappo-training` at `a6eab7e`; exact main dependency
`6f1d1ca`. Primary-source review in
[`blue-rl-feasibility-audit-20261003.md`](../research/blue-rl-feasibility-audit-20261003.md).
This research-only handoff records no new training or simulator result.
Code inspection found that `blue_fqe_select.py` fits the candidate's action
to the logged action's reward, making its FQE ranking invalid; the earlier
claim that six checkpoints have equal offline value is retracted. Other
findings: `blue_iql.py` updates but never reads target `qt`; offline logs
drop the last reward/mark the prior tick terminal; extraction CE omits
validity masks; linear shared recurrent context cancels in same-command
host comparisons. The report ranks correctness checks, episode validation,
contextual host scoring, DAgger, coverage-preserving extraction, and later
duration/attacker work. These are proposed experiments, not implemented
changes. `git diff --check` is the documentation check; no simulator or
training checks were run. This handoff needs no schema/reward change or
dependent branch migration. PR checklist: review primary citations,
verify the local algebra, preserve the untouched final seeds. Next concrete
task: repair offline transition/FQE correctness, with a tiny known-MDP
calibration before checkpoint selection.

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
`docs/archive/blue-agent-plan.md`, and the correction at the top of
`docs/archive/handoff.md` before changing code.

## Current Blue changes

Committed in `7b76fc2` and `58b6cc7` on `blue/foundation`:

- Reworked `blue/cc4_epymarl_wrapper.py`: reset the shared simulator
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
- Reworked `blue/blue_action_masking.py`: pending actions no longer
  overwrite belief, `Restore` stays reachable in validity mode, and successful
  remediation transitions to `VERIFY` instead of declaring the host clean.
- Added `blue/tests_blue/test_foundation.py`.
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

## First real MAPPO training (2026-09-30, `blue/mappo-training`, base `377ab29`)

PR #1 (`blue` → `main`) is merged. EPyMARL `cbc38c09` is vendored at
`third_party/epymarl/` (gitignored via new `third_party/` line; LICENSE +
NOTICE + PIN.txt retained, src unmodified) and linked into `.venv-train` by
`.venv-train/Lib/site-packages/cyberally_epymarl.pth`. `smaclite` exists only
as from-source GitHub (not PyPI), so the SMAC-only dead import is stubbed in
our glue with a loud error on real use. No sacred/torchvision/pettingzoo in
the training path — driver calls EPyMARL's `run_sequential` directly with
their EpisodeRunner/buffer/BasicMAC/PPOLearner and a plain stdlib logger.

- `blue_train_mappo.py`: config mirrors their mappo.yaml (episode runner,
  batch 1, common reward, RNN-64 actor + central-V critic, shapes from
  `get_env_info()`), torch threads capped at 2, CLI for steps/t_max/seed,
  per-run manifest.json + stats.json under gitignored `results/`.
- Smoke (2×50 steps) then short run (8×100 steps, seed 7, ~76 s): finite
  pg/critic losses, nonzero grad norms, critic loss 2.94→2.53, checkpoints
  saved every run. Torch 2.14 emits upstream deprecation warnings (non-tuple
  indexing in their buffer) — noise, behavior unchanged; do not patch src/.
- `blue_eval_mappo.py` loads `agent.th` greedily through `evaluate_policies`
  (400 steps, seeds 7629/7630/7640): the 8-episode checkpoint scores like
  Sleep (returns -1651…-4139, roots 41–52 @200) vs round-robin (-85…-123,
  roots 9–13) and masked-random (-190…-276). **Loop proven, not a defender.**
  Masked-random beating Sleep 10× on return says the reward is dense enough
  to learn from — the gap is training volume, not signal.
- Caveats: first reset fixed (7629), later bare resets CybORG-random and
  unrecorded — seed cycling needed before any serious run. Seed 7640 is much
  harsher than 7629/7630 (Sleep -4138 vs ~-1600): train multi-seed. One
  unexplained full-Windows freeze hit the very first attempt; two full runs
  since completed cleanly — cause undetermined, reruns are cheap, watch it.

## Fresh-machine (WSL) rebuild — snapshot 2026-09-30

Provenance: package lists below are `uv pip freeze` snapshots of the working
Windows venvs, inlined here because Environment's canonical
`environment/requirements-*.txt` drafts are uncommitted and do not transfer.
Canonical pins remain Environment's call; this snapshot only unblocks a fresh
clone. EPyMARL pin `cbc38c09` is also recorded in
`docs/coordination/blue-epymarl-contract.md` (on main).

```bash
git clone git@github.com:sourav-s-b/cyberally.git && cd cyberally
git checkout blue/mappo-training
uv venv --python 3.12 .venv            # 3.10 may work here; 3.12 is proven
uv venv --python 3.12 .venv-train
uv pip install --python .venv-train/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python <sim pins below>
uv pip install --python .venv-train/bin/python <train pins below>
git clone https://github.com/uoe-agents/epymarl.git /tmp/epymarl
git -C /tmp/epymarl checkout cbc38c09588064eab978501d0f12c2cf58fa7fc2
mkdir -p third_party/epymarl
cp -r /tmp/epymarl/src third_party/epymarl/src
cp /tmp/epymarl/LICENSE /tmp/epymarl/NOTICE third_party/epymarl/
echo "$PWD/third_party/epymarl/src" \
  > .venv-train/lib/python3.12/site-packages/cyberally_epymarl.pth
cd cage-challenge-4
../.venv/bin/python -m pytest -q tests_blue                     # expect 41+1
../.venv-train/bin/python -m pytest -q tests_blue/test_epymarl_conformance.py \
  tests_blue/test_action_space_v2.py                            # expect 13
```

Sim pins (`.venv`): cloudpickle==3.1.2 colorama==0.4.6
farama-notifications==0.0.6 gym==0.26.2 gym-notices==0.1.0
gymnasium==0.28.1 iniconfig==2.3.0 jax-jumpy==1.0.0 networkx==3.2.1
numpy==1.26.4 packaging==26.3 pluggy==1.6.0 prettytable==3.9.0 pygame==2.5.2
pytest==8.0.0 pytest-mock==3.12.0 pyyaml==6.0.1 typing-extensions==4.16.0
wcwidth==0.9.1. Train pins: same list plus filelock==3.32.3 fsspec==2026.7.0
jinja2==3.1.6 markupsafe==3.0.3 mpmath==1.3.0 setuptools==78.1.0 sympy==1.14.0
and torch==2.14.1+cpu (CUDA build optional later: WSL2 sees the RTX 3050).

Notes: the `smaclite` stub lives in committed `blue_train_mappo.py`, nothing
to install. Torch 2.14 prints upstream non-tuple-indexing deprecation
warnings from EPyMARL's buffer — noise, do not patch `third_party/`. The one
unexplained Windows freeze has no Linux relevance until observed there.

## Seed cycling + Linux rebuild (2026-09-30, `blue/mappo-training`, uncommitted)

Fresh Linux checkout (no venvs) rebuilt from this file's recipe with
uv 0.12.15 + CPython 3.12.14: `.venv/` sim pins, `.venv-train/` train pins +
`torch==2.14.1+cpu`, EPyMARL `cbc38c09` vendored at `third_party/epymarl/`
(+ `.pth` link). Suites: `.venv` 41 passed + 1 skipped; train-venv
conformance + action-space-v2 13 passed; `.venv-train` full 42 passed (the
torch-only skip runs there). Upstream torch deprecation warnings only.

Seed cycling landed (the "bare resets are CybORG-random" follow-up):

- `CC4MARLEnv(..., seed_cycle=None)`: per bare `reset()` consumes the next
  entry (wraps), explicit `reset(seed=)`/`seed()` wins once without advancing
  the pointer, every applied seed appended to `env.reset_seeds`. Default
  `None` preserves v1/v2 behavior (ctor seed once, then RNG continuation —
  confirmed `CybORG/env.py:218` keeps `np_random` when `seed=None`).
- `CC4BlueWrapper` passes `seed_cycle` through, exposes `reset_seeds`.
- `tests_blue/test_seed_cycle.py`: 6 tests (order/wrap, explicit override,
  setter precedence, validation, default unchanged, cross-env reproducibility).
  Full sim suite now 47 passed + 1 skipped.
- `blue_train_mappo.py`: `build_config(..., train_seeds=(7629,7630,7640))`
  (CLI `--train-seeds`), factory stashes the runner-built env, manifest gains
  `seed_cycle` + chronological `reset_seeds` (+ train/test interleave note)
  and `n_episodes`.
- Verified end-to-end: 2x50 smoke (4 resets `[7629,7630,7640,7629]` — 3 train
  + 1 greedy test; the test episode also consumes a slot, `while t_env <=
  t_max` runs a 3rd train episode at t_env=98) and 8x100 short run (~3.5 min,
  finite pg/critic losses, return_mean -46.9). Checkpoint eval on
  7629/7630/7640 replicates the Windows finding: ckpt ~ Sleep
  (-1651...-4139, roots 41-52 @200) vs round-robin (-85...-123, roots 9-13)
  vs masked-random(seed 11) (-190...-276). Loop proven multi-seed; policy
  still not a defender — needs volume.

Changed, uncommitted, unpushed: `cc4_epymarl_wrapper.py`,
`blue_train_mappo.py`, new `tests_blue/test_seed_cycle.py`.
(`results/` checkpoints gitignored.) Next: longer multi-seed MAPPO volume
until the policy beats round-robin; then update `blue.md` + push.

## 12k-step multi-seed run (2026-09-30, `blue/mappo-training`, `cc9a5c2`)

`blue_train_mappo.py --steps 400 --t-max 12000 --seed 7` (~31 episodes,
t_env 12369, ~15 min on Linux CPU). Manifest `reset_seeds` cycles cleanly
through `[7629, 7630, 7640]` x10+ (train + greedy test interleaved);
periodic checkpoints (399 ... 12369) all saved. Run dir
`results/mappo_cc4_seed7_20260930T163901Z` (gitignored).

Eval (400 steps, seeds 7629/7630/7640; baselines identical to Windows table):

| Policy | 7629 ret / root@200 | 7630 ret / root@200 | 7640 ret / root@200 |
|---|---|---|---|
| mappo final (12369) | -407 / 38 | -2536 / 54 | -2824 / 49 |
| mappo mid (7581) | -1821 / 54 (= Sleep) | -1857 / 53 | -3198 / 55 |
| round-robin | -85 / 11 | -105 / 13 | -123 / 9 |
| masked-random (seed 11) | -239 / 17 | -190 / 29 | -276 / 21 |
| Sleep | -1821 / 54 | -1537 / 49 | -4138 / 52 |

Training curve unstable: train return oscillates (-95...-2708), critic loss
3.8 -> 61.8, final episodes collapse. Mid checkpoint is pure-Sleep-like;
final improves 7629 only (likely seed memorization) and is worse than Sleep
on 7630. Verdict: volume alone of this setup will not beat round-robin —
the actor lacks the belief/age inputs the heuristic uses (see BLUE-04
inventory below). Next: BLUE-04 explicit temporal features first, plus
training stabilization (lower lr, real `test_interval` for validation
curve) before the next long run. Do not launch a longer same-config run.

## Phase A verdict: training stabilized at lr=5e-5 (2026-09-30, PASS)

Per the lr-only plan (strict feature freeze): probe at 5e-5 showed a flat
critic band (3.9->5.8 over 6 episodes), so the 12k confirmation ran at
`--lr 0.00005` (31 episodes, t_env 12369, ~15 min; run dir
`results/mappo_cc4_seed7_20260930T172801Z`, gitignored; driver `--lr` knob
committed as `e3e8859`).

- Critic loss: 3.9 -> 1.5, bounded throughout; one mid-run wobble (31.3 at
  t_env 8778) self-corrected back to ~2 by t_env 10374. Baseline at 3e-4
  exploded 3.8 -> 61.8 with terminal collapse. Blowup signature gone.
- Train return: first-3 mean -208, last-3 mean -198. No collapse — gate
  passes (no learning either, which is Phase B's problem, not this gate's).
- Final checkpoint eval: -1956 / -1538 / -1937 (roots 49/49/53 @200) —
  pure Sleep-level. Plumbing fixed, policy parked at do-nothing, exactly
  the expected Phase A outcome. Features are now the binding constraint.
- Next: Phase B (BLUE-04 temporal/belief inputs, wider train seeds,
  Evaluation-designated held-out suite). No same-config volume needed;
  no sweep expansion needed unless BLUE-04 reintroduces instability.

## BLUE-04 signal inventory (read-only, 2026-09-30)

Actor's 10 host features are snapshot counts only
(`blue_obs_features.py:35`); none of the already-tracked Blue-visible
temporal state reaches the policy: `tracker.state` (5-way),
`last_analysis`/`last_remediation` ages, `empty_strikes`, `last_result`,
per-field `observed_at` freshness, pending flag, normalized tick as
mission-phase proxy (134/133/133 is a measured scenario constant).
Proposal: ~+10 features/host, all Blue-visible belief (never privileged
truth), feature-version bump, ablated ages -> belief -> mission context.
Caution: existing `has_root_session` risks the plan's warned misreading
(Blue root session != attacker root); consider ablating it in BLUE-04.

## BLUE-04 implementation (2026-10-01, `blue/mappo-training`, committed)

Merged `origin/main` first (spec-safe). Implemented the flagged bundle per
the agreed decisions (bundle behind flags, ages capped at horizon,
`has_root_session` ablation):

- `blue_obs_features.py`: `host_to_temporal()` pure builder (ages 2,
  belief one-hot 5, freshness 2 incl. pending-busy, mission tick 1;
  sentinel 1.0 = never-observed, distinct from fresh 0.0) + `temporal_len`
  + `ROOT_SESSION_INDEX`. `host_to_vector` untouched (10-wide).
- `cc4_epymarl_wrapper.py`: `temporal_features=()` default (v2-identical
  observations) + `include_root_session=True`; dims from flags;
  `WRAPPER_VERSION` -> `foundation-v3`; env_info gains temporal keys;
  facade passthrough. Full bundle: host 20-wide, obs 1020, critic 5100.
- Driver defaults to the full bundle (`--temporal-groups`,
  `--drop-root-session` CLI); eval derives geometry from env_info
  (`policy_dims` helper) with matching rollout-env flags, and EPyMARL/torch
  imports are now lazy so the module imports in the sim venv.
- `tests_blue/test_temporal_features.py`: 11 tests (v2-identity default,
  bundle/per-agent/drop dims, sentinel layout, live Analyse age decay,
  belief wiring, root-index-only removal, masked-random trace invariance
  on/off, geometry helper). Suites: 58+1 (sim), 59 (train).
- Smoke (train venv, 2x50, lr=5e-5, full bundle): run completes, manifest
  records flags + cycled seeds, obs/state 1020/5100 flow automatically.
- Next: ablation runs at frozen lr=5e-5 — ages-only vs +belief vs +mission
  vs drop-root, bar = beat masked-random everywhere, then approach
  round-robin. Old (510-dim) checkpoints fail loudly on load, as designed.

## BLUE-04 ablation 1/4 verdict: ages-only insufficient (2026-10-01)

12k run, `--temporal-groups ages` (host 12-wide, obs 612), lr=5e-5 frozen,
run dir `results/mappo_cc4_seed7_20261001T043619Z` (gitignored):

- Stability: critic 5.1 -> 22.7 (max 26.1) — bounded, no blowup, but
  plateaus an order above the no-temporal run's 1.5. Watch item: wider obs
  with hidden_dim still 64 may be straining the critic.
- Returns flat (first-3 -226, last-3 -280): no collapse, no learning.
- Final eval: -1369 / -2309 / -2055 (roots 38/55/50 @200) vs
  masked-random -239/-190/-276 and round-robin -85/-105/-123.
  Better than Sleep on two seeds, nowhere near either bar.
- Verdict: staleness awareness alone is not the gap. Proceed to ablation
  2/4 (+belief) per plan — the belief one-hot is the round-robin
  notebook's core, untested until now.
- Eval tooling bugs found and fixed while scoring (both failed loudly,
  neither corrupted results): `policy_dims` double-added the agent-id
  one-hot at net construction, and the lazy-torch refactor left
  `reset()`/`select()` without the module ref. Regression test added
  (`test_greedy_policy_loads_matching_checkpoint_and_selects`,
   train-venv-only). Suites now 58+2 (sim) / 60 (train).

## BLUE-04 ablation 2/4 verdict: +belief worse than Sleep (2026-10-01)

12k run, `--temporal-groups ages belief` (host 17-wide), lr=5e-5 frozen,
run dir `results/mappo_cc4_seed7_20261001T045747Z` (gitignored):

- Stability: critic 5.1 -> 14.0 (max 18.2) — bounded, calmer than
  ages-only's 26-plateau. Returns flat (-242 -> -276). No collapse.
- Final eval: -2551 / -2786 / -2412 (roots 42-50 @200) — worse than Sleep
  on 7629/7630, better only on 7640. Belief inputs made behavior actively
  harmful, not merely useless.
- Autopsy (action histograms, native Red, seeds 7629/7630) overturns the
  Sleep-collapse hypothesis for both temporal checkpoints: ages-only acts
  41-43% of ticks (Analyse 23-29%, Remove 14-18%), +belief 35-44%
  (Analyse 21-36%, Remove 5-11%, Restore 2-3%). Exploration is healthy;
  actions are misdirected — remediation without skill, likely mistimed
  Restore disruption and unverified Removes.
- Revised diagnosis: not collapse (entropy fine) but unshaped behavior —
  the scan->confirm->remediate->verify chain never forms. Next diagnostics:
  scan precision (fraction of Analyses targeting compromised hosts,
  privileged eval-only) to split sensing vs remediation failure; then
  curriculum (phase-dense slices) and/or BC warm-start from round-robin
  before any more flat runs. Flat 12k volume is retired as a strategy.

## Scan-precision split + evidence-mode scoring (2026-10-01)

Privileged eval-only analysis (`/tmp/scan_precision.py`, not committed):
fraction of each action type targeting actually-compromised hosts at
decision time, seeds 7629/7630:

| Policy | Analyse precision | Remove precision | Restore precision |
|---|---|---|---|
| round-robin | 21-29% (blind sweep) | 100% | 86-96% |
| ages-only ckpt | 21-55% | 60-72% | never used |
| ages+belief ckpt | 79-85% | 25-34% | 2-4% (wipes clean hosts) |

Reading: belief features solved sensing (85% vs heuristic's 29%) while
remediation stayed blind — round-robin wins entirely on its
CONFIRMED-gated remediation (100%/86-96%), which no checkpoint learned.
`blue_eval_mappo.py` gains `--mask-mode` (validity|evidence) for ablations.

Evidence-gated scoring of the ages+belief ckpt: -1821/-1963/-3196
(Sleep-level) — gating removed the vandalism but nothing useful replaced
it; the policy's edge came from unconstrained remediation and it never
learned correct remediation. Side finding: evidence mode also improves
masked-random (-239->-134 on 7629): the mask helps any policy, but the
trained net still can't beat a gated coin flip. Remediation skill is zero.

Pivot decision: retire flat full-episode runs. Next, in order: (1)
evidence-gated TRAINING (config-only 12k, gate = Remove precision >80%
+ beats masked-random); (2) park mission/drop-root ablations (sensing
refinements for a solved problem); (3) BC warm-start from round-robin
traces; (4) curriculum feasibility probe. Held-out seeds still await
Evaluation; Red pool (started on `red/strategy-adapter`) is the eventual
robustness track, not the current one.

## Evidence-gated TRAINING verdict (2026-10-01, Step 1)

`blue_train_mappo.py` gains `--mask-mode` (was hardcoded `validity`);
probe reuses `env_args` so the flag flows automatically. Ran 12k
(`results/mappo_cc4_seed7_20261001T115017Z`, ages+belief, lr 5e-5,
evidence masks in training): critic 5.9->1.4, healthiest curve yet.

Paired eval, validity mode: ckpt -701/-1084/-3163 vs ungated
-2551/-2786/-2412 and Sleep -1821/-1537/-4138 — beats Sleep everywhere,
3.6x/2.6x better on 7629/7630. Evidence-mode scoring: -2094/-1969/-2532,
no better than the ungated ckpt under the same mask.

Precision autopsy (validity scoring): Analyse 65-74% (was 79-85%),
Remove 36-51% (was 25-34%), Restore ~0-8% (vandalism cured, Sleep 55-58%).

Gate FAILS both criteria: Remove precision 36-51% (< 80%), returns far
from masked-random (-134/-155/-138). Reading: the constraint taught
restraint and some targeting, but not skill — the failure is now confirmed
deeper than exploration. Next: BC warm-start from round-robin traces
(`run_episode` already logs action traces; need BC loss glue + PPO
fine-tune), per plan Step 3.

## BC warm-start arc (2026-10-01, Step 3)

New: `blue_collect_bc.py` (10 round-robin teacher eps, ages+belief,
~20k samples, teacher 55% Sleep), `blue_bc_pretrain.py` (GRU full-episode
CE -> `results/models/bc_rr_<stamp>/0/` resume quad incl. fresh critic +
Adam states), `blue_train_mappo.py --init-ckpt` (EPyMARL checkpoint_path
resume; needs full quad — critic.th + both opts, first attempt crashed
without them), `tests_blue/test_bc_warmstart.py`.

1. Unweighted BC -> Sleep-mode collapse (acc 54.65% = majority). Cause:
   [EDIT 2026-10-01: "hidden RR cursor unobservable" is FALSE — cursor is
   recoverable from obs (proposal 12). Cause is class imbalance + the flat
   head's argmax bottleneck.] Fix: inverse-sqrt class
   weights + zero loss on busy ticks (mask sum<=1).
2. Weighted BC: nonsleep-acc 0.3%->12% (exact-id ceiling is the hidden
   cursor). Zero-shot semantics transfer: Analyse 565-781, Restore
   86-174 at 36-37% compromised-precision (vs any PPO ckpt ~2-3%).
3. BC zero-shot scores -256/-379/-211 — best learned policy of the
   project, beats Sleep everywhere and masked-random on 7640.
4. PPO fine-tune 12k from BC (`mappo_cc4_seed7_20261001T131137Z`):
   catastrophic unlearning — 9975 scores -1960/-2683/-2522, final 12369
   -1762/-2060/-2522. Fresh critic + noisy early advantages wipe the init
   before the value head learns; never recovers.

Standing result: imitation beats all RL here (-211..-379 vs PPO best
-701); PPO as currently configured cannot preserve a good init, let alone
find one. Next candidates: low-lr preserve-fine-tune (config-only),
critic-warmup-then-joint (needs learner support), or accept BC + report.

## CORRECTION + genuine low-lr fine-tune (2026-10-01)

The two "fine-tune" runs above never tested their stated lrs. EPyMARL
`load_models` (`ppo_learner.py:261`) restores optimiser state wholesale
INCLUDING lr (`Adam.state_dict` param_groups), so both runs trained at the
BC quad's stamped lr=1e-3 — 20-100x the intended 5e-5/1e-5. Verified:
`agent_opt.th` param_groups lr 0.001; critic_loss curves byte-identical
across the two "different-lr" runs. From-scratch runs unaffected (fresh
opts). Fix: `blue_bc_pretrain.py --opt-lr` stamps fresh optimiser states
at the intended PPO lr (default 5e-5); regression test
`test_resume_inherits_checkpoint_opt_lr` pins the mechanism.

Genuine 1e-5 fine-tune from a 1e-5-stamped quad
(`mappo_cc4_seed7_20261001T141327Z`): training returns held -150..-190
throughout (never collapsed); final scores -234/-460/-486 vs BC init
-256/-379/-211. Init PRESERVED, no destruction — but also no gain over
the teacher level in 12k steps. Revised ladder: teacher -85 >
masked-random -134 > BC/fine-tuned -211..-486 > PPO-from-scratch -701 >
Sleep. Next optional: warm-critic continuation (resume the 1e-5 ckpt at
5e-5 now the value head is trained — needs another opt-lr-stamped quad).

## Combined 1+2+3 build: teacher v2 + shaped rewards (2026-10-01)

Teacher v2 `SuspicionSweepBaseline` (same remediation core, suspicion-
ordered coverage + cooldown): ties round-robin exactly (-85/-105/-123) on
all three seeds — no gain, so v1 remains the teacher; v2 kept as a future
diversity option, not a claim.

Shaping (path 1) in wrapper: `SHAPING_DEFAULTS` {clear +3, confirm +1,
vandalism -3} + pure rule `shaping_event_bonus()`; `shaping=` kwarg on
CC4MARLEnv/CC4BlueWrapper, `train --shaping`, recorded in get_env_info +
manifest. Reward-channel privileged scan (true state) is legal per
contracts (only OBSERVATIONS are Blue-restricted); failed clears
score 0 by design (correct escalation, not error). Eval stays native.
Tests: `tests_blue/test_shaping.py` (4, incl. zero-failed-clear rule).

Result A — shaping @5e-5 (from BC quad stamped 5e-5,
`mappo_cc4_seed7_20261001T145833Z`): shaped training return -435 -> -68
(bonuses farmed) but NATIVE eval -1013/-736/-1479, worse than the BC init
(-256/-379/-211) and worse than unshaped 1e-5 (-234/-460/-486). Verdict:
5e-5 destroys the init regardless of reward; rising shaped return was
bonus farming, not skill.

Result B — shaping @1e-5 (`mappo_cc4_seed7_20261001T152955Z`), shaping the
ONLY variable vs the unshaped 1e-5 run: NATIVE -723/-337/-597 (mean -552)
vs unshaped -234/-460/-486 (mean -393). Shaping also degrades.

## Combined verdict: PPO fine-tuning is net-destructive here (2026-10-01)
**[RETRACTED 2026-10-01: the 8-seed eval (±313 band) showed these 3-seed
deltas are inside seed noise. Only lr=1e-3 divergence and Sleep-level
collapse stand. Table preserved for the record.]**

Four independent fine-tunes, all starting from the SAME BC init, all
scored on native returns:

| Run | lr | shaped | 7629 | 7630 | 7640 | mean |
|---|---|---|---|---|---|---|
| BC init (no PPO) | - | - | -256 | -379 | **-211** | **-282** |
| fine-tune | 1e-5 | no | **-234** | -460 | -486 | -393 |
| fine-tune | 1e-5 | yes | -723 | -337 | -597 | -552 |
| fine-tune | 5e-5 | yes | -1013 | -736 | -1479 | -1076 |
| (earlier, lr actually 1e-3) | 1e-3 | no | -1960 | -2683 | -2522 | -2388 |

Monotone in lr, and shaping hurts at BOTH lrs. Every PPO update moves the
policy AWAY from the BC optimum; the optimum is the init itself. The
destructive force is structural (sparse delayed credit + fresh-critic
advantage noise), not a tuning artifact — 4/4 fine-tunes degrade.

Standing deliverable: BC-distilled policy (-211..-379), 3x better than
PPO-from-scratch and beating Sleep on all seeds. Beating the teacher
(-85) requires the neuro-symbolic route (path 2): hard-code
CONFIRMED-gated remediation + verify + escalation as a guaranteed floor,
learn only host-prioritisation/timing on top. PPO must not touch the
weights (or may only be used to generate BC data via the shaped env).

## Vendored-learner patch must be reapplied on a fresh clone (2026-10-01)

`third_party/` is gitignored, so the critic-only warmup change to
`src/learners/ppo_learner.py` is NOT in git. It is recorded as
`environment/patches/ppo-warmup.patch` (+ `.README.md`), verified to apply
cleanly against pristine EPyMARL cbc38c09. A fresh clone that skips this
step will accept `--warmup-steps` and silently do nothing. Add it to the
rebuild checklist below.

## CRITICAL: the 3-seed eval cannot resolve these deltas (2026-10-01)

The table above is a five-row comparison whose every adjacent pair is
smaller than the measurement noise. Re-running the SAME BC init over 8
seeds:

| seed | 7629 | 7630 | 7640 | 7701 | 7702 | 7703 | 7704 | 7705 |
|---|---|---|---|---|---|---|---|---|
| BC init | -256 | -379 | -211 | -323 | -134 | -324 | **-1023** | -497 |

mean **-393.4**, std **276.9**. The 3-seed subset [-256,-379,-211] reads
-282; the 8-seed mean is -393. So:

- Seed 7704 alone scores -1023, worse than the lr=1e-3 runaway.
- 95% CI on a 3-seed mean is about **+/-313** — wider than the entire
  "BC init vs low-lr fine-tune" gap (-282 vs -393) and wider than the
  teacher gap we have been chasing.
- Therefore the "4/4 fine-tunes degrade" conclusion above is **not
  established**. It is consistent with the data but the effect is inside
  the noise band. Only the lr=1e-3 collapse is clearly outside it.

Correction to the previous section: the ladder and the "structurally
destructive" claim are overstated. What is actually measured is that no
fine-tune has been shown to *exceed* its init beyond noise, and that
Sleep-level and lr=1e-3 divergence are real.

Consequence for method: any future A/B needs >=8 seeds and must report
mean +/- std, not a 3-seed mean. Cheap seeds (7629/7630/7640) are fine for
regression tests and training-seed cycling, not for ranking policies.

## Warmup attempt (WSRL-style) — implemented, measured, no help (2026-10-01)

The literature review (`docs/research/blue-sparse-reward-literature.md`)
identifies value divergence at the onset of on-policy fine-tuning as the
mechanism (WSRL, ICLR 2025). Implemented the published fix: a critic-only
warmup that freezes the actor for the first `warmup_steps` while fitting the
critic to on-policy data.

- `blue_train_mappo.py --warmup-steps N` / `--no-warmup-critic-only`
- learner freeze in `third_party/epymarl/src/learners/ppo_learner.py`
  (`_warmup_active`, guarded by `getattr` so upstream configs still load)
- `tests_blue/test_ppo_warmup.py`: 5 tests, incl. actor bit-identical during
  warmup while the critic still moves, and the opt-out ablation arm.

Run `mappo_cc4_seed7_20261001T162853Z`, lr=1e-5, warmup=4000, ages+belief:

| run | 7629 | 7630 | 7640 | 3-seed mean |
|---|---|---|---|---|
| BC init | -256 | -379 | -211 | -282 |
| no warmup | -234 | -460 | -486 | -393 |
| warmup 4000 | -315 | -406 | -643 | **-455** |

No improvement. Two diagnostics explain why it could not have worked here:

1. **Parameter drift is nearly identical** between the arms (relative L2
   0.0047 warmup vs 0.0048 no-warmup). Freezing the actor early did not
   reduce total movement, so this was never the binding constraint.
2. **Behavioural KL(pi_bc || pi_finetuned) is ~0.001 nats** measured on
   4000 teacher-visited states. The fine-tuned policies are almost
   *identical* to the init. Tiny KL, ~170-point return gap => the gap is
   mostly seed noise, per the section above, not policy drift.

The freeze is correct, well-tested and off by default, so keep it as a tool,
but the experiment does not support it and it is not the fix.

## Corrected picture and the real blocker

With KL ~0.001 nats the policies barely differ, so "PPO destroys the BC
policy" was the wrong diagnosis. The actual gap is capability: BC is at
~-393 (8 seeds) and round-robin is -85/-105/-123, and **no amount of
fine-tuning closes that**, because the sweep cursor round-robin uses is not
in the observation. BC cannot learn an ordering the features do not contain.

## CORRECTION: the sweep cursor IS in the observation (2026-10-01)

The two sections above blame a *representation gap*: "round-robin's sweep
cursor is not in the observation, so no fine-tune can recover it". **That is
false, and it is checkable in one experiment.** I tested it.

Implemented `StalestFirstBaseline`: the teacher's exact remediation core
(CONFIRMED -> Remove, re-detected -> Restore, VERIFY -> Analyse) but with
the cursor replaced by the observation-derived rule "Analyse the legal host
with the largest time-since-last-analysis, never-analysed first". That
quantity is literally the `ages` feature, `min(1, (tick - last_analysis) /
cap)` with 1.0 for never-analysed, plus `tick` from `mission`.

| policy | 7629 | 7630 | 7640 | 7701 | 7702 | 7703 | 7704 | 7705 | mean | std |
|---|---|---|---|---|---|---|---|---|---|---|
| round_robin (cursor) | -85 | -105 | -123 | -85 | -135 | -64 | -53 | -98 | -93.5 | 27.8 |
| stalest_first (from obs) | -85 | -105 | -123 | -85 | -135 | -64 | -53 | -98 | -93.5 | 27.8 |

Not just equal returns: **the action traces are byte-identical** (1995
actions per episode, 0 mismatches, verified on 7629/7640/7704). The cursor
carries no information the observation lacks. The teacher's advantage is
fully expressible in Blue-visible features.

So the blocker is not the representation. Re-reading the geometry makes the
real obstacle concrete:

- Observation is a FLAT, POSITIONAL, fixed-slot tensor: 51 host slots x 17
  features = 867 dims (+5 one-hot agent id = 872, matching `fc1: [64, 872]`).
- `17 = 10` base `+ 2` ages `+ 5` belief one-hots.
- Action space is `155 = 2 + 3 x 51`: one index per (host, action) pair.

Choosing a host is therefore an **argmax over 51 slots**, emitted as a single
index into a 155-way head. The information is present, but a policy that must
implement a cross-slot comparison by routing it through a shared MLP + GRU
bottleneck is the wrong shape for it: the comparison is global, the encoder
is local. BC reaches only ~12% non-sleep exact-action accuracy, which is what
failing to express a global argmax looks like.

Revised next step: change the *architecture*, not the inputs. Give the actor
a per-host-slot scoring head whose outputs are combined across slots (masked
softmax/argmax over the 51 slots, then expanded into the 155-way index), so
"which host" is computed by the architecture instead of being learned inside
a dense layer. Keep flat 867-dim obs; keep the fixed CONFIRMED/VERIFY
skeleton as a hard floor. This is a smaller and more honest change than
re-designing the feature set, and it is testable against the -93.5 teacher
number on >=8 seeds.

## Architecture review (2026-10-02)

Deep literature review complete; decision doc:
`docs/research/blue-architecture-review.md` (new, committed).

- Official CAGE-4 analysis (Kiely et al., AAAI 2025, PDF extracted locally):
  top-3 teams all sweep-and-remediate heuristics (UC -113 ± 35, lancer
  -118 ± 40, punch -142 ± 44); best MARL was cybermonic's GNN at -193 ± 84.
  Eval is 100 x 500-step episodes; our -93.5 is over 400 steps, so compare
  per-step rates only (-0.234 vs -0.226/step), never raw totals.
- Winning designs map directly onto our hybrid: lancer's per-host priority
  (decay on touch, boost on Monitor) ~= `HybridBluePolicy(priority_fn)`;
  punch's file-density>0.9 host flag -> belongs in our belief features;
  UC's persistent malicious-event flags + present-host encoding (killed
  invalid actions) -> feature work.
- Four subagent reviews (parallel): TERLA (HGT + 5-action collapse + action
  waiting + shaped training reward); SR-DRL/GACD/CyberDreamcatcher
  (host-then-command factorization wins, OT/GAT-REINFORCE rejected);
  Set-Transformer/REFIL/factored-actions (Thm 1 formalizes our MLP-bottleneck
  diagnosis; REFIL mask pattern ports to EPyMARL); HPPO/H-MARL (learned
  gating ~= null result; expert-rule master already equals our fixed rules).
- Decision: (1) harden hybrid first (S); (2) host-then-command factorized
  actor + optional 1-layer entity attention (M); (3) phishing/stealthy/
  aggressive-style red variants as held-out suite. Parked/rejected: full
  GNN, learned PPO master, curiosity, shaped training rewards,
  factored-additive Q. Next concrete task: lancer-style priority + density
  flag + `HybridBluePolicy(None)` == round-robin parity proof, >=8 seeds.

## Hybrid parity proven (2026-10-02)

Proposal 01 step 1 done. `blue/blue_hybrid.py` (now tracked)
fixed so `priority_fn=None` runs a cursor round-robin identical to
`RoundRobinBaseline` (first draft's `cands[0]` would have stuck on one
host). New regression `tests_blue/test_hybrid_parity.py` passes. 8-seed
manifest (400 steps, native): identical returns AND byte-identical
1995-entry traces on all of 7629/7630/7640/7701-7705 (mean -93.5 both;
numbers in `docs/proposals/01-hybrid-priority.md`). Sim suite: 82 passed,
5 skipped. Next: lancer-style priority + density/persistent flags, same
seeds, then risk scorer.

## Parallel eval harness (2026-10-02)

Per approved plan: `blue/blue_policy_registry.py` (picklable
name -> constructor specs, incl. `mappo_ckpt` torch entry),
`blue_eval_parallel.py` (process pool over policy x seed cells; fork for
heuristics, spawn + train-venv requirement for torch),
`blue/blue_compare.py` (table, paired diffs, trace parity,
proposal-ready markdown). Manifests committed under
`docs/proposals/manifests/`. Correctness: `tests_blue/test_eval_parallel.py`
(pool == serial, cross-worker determinism, torch-spec fails loudly) passes.
Live proof: 8-seed parity re-run through the pool reproduces the serial
record exactly (16 cells, 66 s on 6 workers; manifest
`parity-pool-20261002.json`; compare reports +0.0, MATCH). Sim suite: 86
passed, 5 skipped. Subagent layer: one Task subagent per proposal owns its
eval end-to-end (run shard, analyze, update proposal file).

## Lancer v1 beats teacher (2026-10-02, needs replication)

`LancerPriority` in `blue_hybrid.py` + `hybrid_lancer` registry entry;
`tests_blue/test_hybrid_priority.py` (6 tests) passes. Pool
`lancer-v1-20261002`: hybrid_lancer -63.2 ± 26.0 vs round_robin
-93.5 ± 27.8, paired diff +30.2 (6/8 seeds; t ~= 2.6, p ~= 0.04
uncorrected, first variant — encouraging, not victory). Wins by
remediating less (fewer Restore disruptions); fails on 7704/7705 via
sticky-suspicion re-analysis loops + Restore storms. v2: decay suspicion
bonus with fruitless re-analyses (empty_strikes). Sim suite: 92 passed,
5 skipped. Numbers in `docs/proposals/01-hybrid-priority.md`, manifest
`docs/proposals/manifests/lancer-v1-20261002.json`.

## v2 + held-out: fixed constants exhausted (2026-10-02)

v2 (`fruitless_decay` 0.5 on the suspicion bonus; unit-tested incl. a
caught-and-fixed logic bug) regresses to -75.2 on the same 8 seeds —
still beats teacher (-93.5) but worse than v1 (-63.2). Held-out, 8 fresh
seeds 7801-7808: v1 collapses to -110.1 ± 130.4 (seed 7802: -414)
vs teacher -85.1 ± 46.1; v2 tracks teacher at -91.2 ± 48.5. v1's +30.2
was selection bias; v2 regularizes to parity. Verdict in proposal 01:
no improvement claim stands; next is the adaptive risk scorer, not more
constant-tuning. Manifests `lancer-v2-20261002.json`,
`lancer-heldout-20261002.json`. Sim suite: 94 passed, 5 skipped.

## Snapshot risk scorer killed (2026-10-02)

`blue_risk_data.py` (245k rows, fresh seeds 7901-7908) + `blue_train_risk.py`
(numpy logreg; AUC 0.67 undetected) + `RiskPriority` scorer + registry
`hybrid_risk` + 11 tests, all passing. Pool: hybrid_risk -196.6 / -190.8 vs
teacher -93.5 / -85.1 (worse 15/16 seeds). Cause confirmed: 803 analyses on
27 hosts vs 793 on 67 — pure proba fixates, no touch dynamics. AUC without
ordering dynamics is worthless. Path closed in proposal 01; best remains
lancer_v2 parity. Next: proposal 02 (factorized actor). Sim suite: 102
passed, 5 skipped.

## Factorized actor: head validated, fine-tune diverges (2026-10-02)

Proposal 02 built with zero vendor edits (combined-logit trick):
`blue_factorized_agent.py` + `--agent rnn_factorized` in train/BC drivers +
`GreedyCheckpointPolicy(agent_type=...)` + `mappo_ckpt_factorized` pool
entry (spawn path proven: 16 cells, 156 s). BC-distill 21.8% nonsleep-acc
vs flat 12.2% — architecture VALIDATED. MAPPO fine-tune from distill:
-296 -> -261 (ckpt 2793) -> -583 (7581) -> -1709.5 (12369, 8 seeds). Same
divergence disease as flat head, faster. In-training single-episode probe
(-75) exposed as seed-lottery noise. Also repaired a 6-char typo corruption
in vendored `rnn_agent.py` (`fc2(h)openco`, isolated, restored to upstream;
prior torch runs prove prior intactness); pool CLI now forwards env flags
to torch probe construction. Sim suite 102/6-skip (importorskip guard);
train subset 13 passed. Full record in proposal 02. Next: stabilized
fine-tuning (KL-to-teacher, schedules, behavior constraints) — not more
architecture.

## Review fix batch (2026-10-02)

Three-audit review (docs/code/git) + own verification. Findings fixed:
-02 mean -1770 -> -1709.5 (recomputed from manifest cells); v2now AUC
stale-manifest anomaly resolved (buggy-auc era file; regenerated 0.682 —
docs' "0.68 now" stands).
- Retraction banners: literature F1 header, 2 cursor lines, bc_pretrain
comment, session history (net-destructive verdict + cursor cause), 01 stale
lines; 01 status -> tested, verdict closed; README taxonomy extended.
- Safe code: HybridBluePolicy tick-regression auto-reset (+ test), _n
removed, hybrid_risk/RiskPriority killed-markers, --markdown no-op help,
BC denom guard, pool probe mask_mode forwarding, compare seed-set check,
FRAGILE index comment; new tests: spawn parity, ckpt-resume, stalest
determinism. Lancer behavior frozen (measured artifacts).
- claim_check rebuilt as registry `stalest_first`: pool stalest-20261002
byte-identical traces on all 8 seeds (stronger than the lost 3-seed
scratch); proposal 12 updated.
- professor-brief.md deleted (stale, untracked).

## KL-to-teacher slows collapse only; line closed (2026-10-02)

Frozen-demonstrator KL in the vendored learner (`teacher_kl` + teacher MAC,
`--kl-teacher-ckpt/--kl-teacher-coef`, recorded as
`environment/patches/ppo-kl-teacher.patch` + README, diff-generated and
byte-verified; applies after ppo-warmup.patch). Math unit-tested (3 tests;
caught a flipped expectation). KL run (coef 0.1, same config): KL stat
0.0003 -> 0.07, 8-seed means ckpt-5187 -940 (was -1072), final -1265 (was
-1709.5). Slower collapse, same destination; KL 0.07 nats with returns at
-1000s = razor-thin landscape, and the anchor caps at distilled -205
anyway. Closed absent a better distill. Sim 102/7-skip (torch tests skip);
train 16 passed. Full record in proposal 02.

## Generalization suite phase 1 (2026-10-02)

Wrapper `red_agent=` flag (discovery/finite/verbose/random/sleep; default
preserves all prior results) + pool `--red-agent` + `test_red_variants.py`
(3 tests). 5 reds x 3 policies x fresh seeds 8001-8008 (120 cells):
finite much harsher than discovery (-139 vs -59 teacher); verbose bit-
identical to finite (printing-only subclass — expected, good determinism
cross-check); random mild; sleep-red all zeros (flag sanity). lancer_v2 ≈
teacher on every red (±15, noise) — regression win does not transfer.
Proposal 15 + manifests `red-*-20261002.json`. Green rates/durations need
simulator hooks: coordination proposal `docs/coordination/blue-red-
variants.md` (draft, needs Environment review; Blue will not modify core
sim).

## Distill ladder: attention reaches teacher parity (2026-10-02)

Phase 2. Clean confusion (teacher-driven states): distilled errors are
26-32% wrong-host-Analyse, ~1% wrong-command, 0% omission/commission —
commands perfect, host selection broken. More demos: no gain (21.7% vs
21.8%). Aux host-CE (1.0): HURTS (-503 vs -307 same demos). 1-layer
TransformerEncoder over slot embeddings (`--attn-layers`, default 0 =
identical behavior; 101-indivisible-by-4 fixed by attending on 32-dim
emb only): distill 25.7%, pool -85.5 ± 31.1 regr / -93.9 ± 45.8 heldout
vs teacher -93.5/-85.1 — FIRST learned teacher parity, no RL, worst
heldout seed -147 (vs lancer -414, no-attn -1959). Cross-run trace hashes
identical. Cost ~500 s/8 cells. Full record proposal 02 Exp3, proposal 03
tested. Ladder gate PASSED; KL-PPO unlocked with stop-rules. Sim 108/7;
train 22 passed.

## Risk x recency gate failed (2026-10-02)

Phase 3. `RiskRecencyPriority` (decay/bonus/gate) + 3 registry entries;
unit-caught init-order trap fixed (overridden reset mid-construction).
Pool 5x8x2 (80 cells): decay -203/-105, bonus -186/-251, gate -96/-165
vs teacher -93.5/-85.1 and lancer_v2 -75.2/-91.2. Gate (beat lancer_v2
held-out) FAILED all three: decay adds noise not signal, bonus inherits
fixations, gate doesn't abstain safely. Lesson: 0.67-AUC snapshot signal
can't survive the coverage trade with or without dynamics; only supervised
ordering-learning (attention distill) reached parity. Proposal 01 CLOSED.
Sim 112/7. Manifests `riskx-20261002.json`, `riskx-heldout-20261002.json`.

## Second literature sweep: RL recovery (2026-10-02)

User asked for 200–500 recent papers across weaknesses, max subagents. Ran
10 parallel tracks (cyber-MARL, exploration, offline/IL, BC→RL fine-tune,
MARL credit, architectures, belief/world-models, robustness/UED, hierarchy/
shielding, eval+security-signals): ~300 entries, ~260 unique after cross-
track dedup. Deliverable `docs/research/blue-rl-recovery-literature.md`:
weakness-mapped top-15 bets, track catalogs, ordered P0–P9 program. Top
bets: IQL-discrete on lancer logs, AWAC/PEX/JSRL from attention ckpt,
HAPPO + history-critic audit, PLR⊥ red curriculum, E3B/NovelD + dynamic
PBRS, entity-RL/SR-DRL/GTrXL actor path, DVRL/VariBAD belief, rliable
reporting now. Caveat: search-existence checked, not read; verify DOI
before citing/implementing. Next: pick P0–P2 to convert into proposals.

## Recovery proposals 14/16/17/18 (2026-10-02)

Converted survey P1/P2/P4–P8 into four build proposals, each a response to
a measured failure: 14 offline stitching (IQL-discrete + RvS, do first —
routes on data-vs-algorithm ceiling); 16 collapse-proof fine-tune (D0
plasticity Dx → JSRL+AWAC → PEX, kickstarting-decay fallback); 17 critic
audit (Lyu, parallel) → conditional HAPPO + PBRS-proof densification +
E3B/NovelD as new-family exploration (distinguished from rejected 07/09);
18 entity-RL actor + DVRL/VariBAD belief + PLR⊥ now / ACCEL-PSRO later,
CONDITIONAL on 14–17 producing a surviving updater, with POMCP as the
no-training planning baseline and an explicit deferred list (diffusion,
GAIL, DreamerV3-full, program synthesis, Mamba) with trigger conditions.
README index updated. No code; awaiting build order.

## Proposal 14 rung 1: RvS ties teacher (2026-10-02)

Collector extended (--teacher/--mix + team rewards); 24 mixed eps
8101–8124 (lancer_v2/RR/random, returns -44…-304 bimodal).
`blue_rvs_pretrain.py` factorized+RTG (`rtg_dim` split, default path
unchanged): 22.6% nonsleep-acc. Eval via `rvs_ckpt` + `observe_step`
hook (no-op for heuristics). Target -50 → -112.6±87 (tail -313);
target -80 → regression -87.1±28.1 (parity), held-out -94.0±45.3 vs
RR -85.1±46.1 (paired -8.9, one -60 tail; cf lancer_v2 -6.1). RvS
TIES: conditioning works, stitching missing — Brandfonbrener holds.
Routing: rung 2 IQL-discrete. Sim 112/8; torch rvs/factorized/baseline
tests pass. Manifests rvs/rvs-t80/rvs-heldout-20261002. Ckpt local-only
(results/models/rvs_14/).

## Proposal 14 rung 2: IQL stitches in-distribution, gate failed (2026-10-02)

`blue_iql.py`: expectile-V + Q backup + AWR extraction into stock
factorized GRU (pool via mappo_ckpt_factorized). Pure AWR collapses a
seed (-826, 66 vs 86 hosts: drops zero-advantage sweeps — mechanism
confirmed, beta can't fix, `--awr-alpha` blend is the fix). Alpha=0.5:
regression -82.0±27.7 vs RR -93.5 (paired +11.5 — FIRST learned
regression-beat); held-out -96.0±47.5 vs -85.1 (paired -10.9, no
catastrophe, two seed wins). 60-ep and strong-40 logs both lose to the
24-ep mix on regression (weak tails may sharpen envelope — open). Verdict:
PARTIAL — update vindicated, transfer is the gap; route to 16 (IQL ckpt
replaces attention as best init) + 17. Sim 112/9. Manifests
iql/a05/a03/a07/full/strong/heldout-20261002. Ckpts local-only.

## Phase A: Q-instability kills the scaling story (2026-10-02)

60-ep 50k-iter retrain DIVERGED (loss_q 4→421, adv_max 1159): more budget
without stabilization fails; 20k was early-stopped luck. Train seeds 1–2
on 24-ep: loss_q 8.4/6.4 (vs 1.6 seed 0) → regression -105.6±90.4
(7640 -319 tail) / -108.1±31.6. loss_q predicts return; seed-0 -82 was a
lucky ticket — soften single-seed conclusions incl. data gaps. Gate: no
champion → Phase B held-out look CANCELLED, one-look budget preserved.
Lesson: rung-2 IQL is a lottery ticket; rung 3 needs value stabilization
first (17-adjacent). Manifests iql-s1/s2-20261002 added.

## MLP-structure intervention (2026-10-02)

User asked: would changing the MLP solve it? Did it (LayerNorm +
--reward-scale/--layernorm in blue_iql.py): values fixed (loss_q
1.6→0.062, zero clipping) but seeds collapse the other way (adv_std
0.01 — extraction degenerates to BC). Regression: stab0 -88.8 (+4.8, wins
+50/+40/+37, losses -64/-29); stab1 -107.8 (-244); stab2 -161.2
(-543/-230). MLP was A lottery layer, not THE lottery — extraction
(GRU cloning, never validation-stopped) is seed-fragile on its own.
Missing discipline found: validation-split early stopping on extraction
CE. Manifests iql-stab/stab1/stab2-20261002.

## Second sweep: extraction-lottery literature (2026-10-02)

User ordered deep search on current failure modes pre-/compact. 8 tracks
(OPE/selection, BC fragility, AWR mechanics, transfer, recurrence,
coverage cloning, ensembles, validation); 7 returned ~200 papers (~190
unique). Deliverable `docs/research/blue-extraction-lottery-literature.md`:
failure modes F1-F6 with numbers, top-8 bets, 7 track catalogs, ordered
V1-V8 program (V1 validation+early-stopping first — cheapest, unlocks
all). BC-fragility track empty; re-run post-compact. Top actionable:
ESS monitor (one line), FQE-V(s0)/BVFT selection (ends test-env picks),
snapshot/SWA de-lottery, SPOT/TD3+BC support knob, burn-in/hidden-reset
ablations, diversity-over-size logging.

## Stream C done: FQE-V(s0) rollout-free selection (2026-10-03, V3)

**Superseded interpretation:** the research audit at the top of this file found
that the implementation fits candidate actions to rewards generated by logged
actions. The measurements below remain a historical record, but their FQE
ranking and value-indistinguishability verdict are invalid pending repair.

New `blue/blue_fqe_select.py`: per-candidate MSE FQE fit (fixed 30k-iter
budget, 2 fit seeds) under candidate greedy actions (zero-hidden
memoryless scoring) + V(s0) rank + MC anchor (-133.1 undiscounted) +
fit-seed stability flag. CPU venv `.venv-train` (torch 2.14.1 CPU-only);
~5 min/fit. Manifest `docs/proposals/manifests/fqe-v3-20261003.json`.
RESULT on all six lottery ckpts: value band [-23.1,-20.5] with ~1.2 fit
noise — value-indistinguishable; regression gaps (-82 vs -108) are NOT
offline value differences. Batch1 rank reproduces regression order
(a05>s1>s2), batch2 swaps stab/stab1, both flagged UNSTABLE (spread ~=
range). Verdict: selector correctly refuses to certify near-ties;
reframes F1 as extraction/rollout noise, not value signal. Next
policies must separate by >>1.2 in V(s0). Next: Stream A (V1 validation
splits BC/RvS).

## Stream C redo: TRUE FQE (2026-10-03, audit-driven)

Rewrote `blue_fqe_select.py` as Le et al. 2019 Alg.3: regression on
LOGGED (s,a), bootstrap Q(s',pi_masked(s')) with carried (not zeroed)
GRU hidden + deployment mask, discounted MC anchor (-32.1; undisc
-133.1). v3 manifest marked SUPERSEDED in-file (failure record
retained). New manifest `fqe-v3r2-20261003.json`. Batch1 (IQL seeds):
FQE order a05>s1>s2 reproduces regression, STABLE (range 5.8, spreads
<=2.2). Batch2 (stab): FQE stab1~stab2>stab DISAGREES with regression
(stab>>stab1>>stab2), self-flags UNSTABLE (stab spread 4.0 vs range
2.4). stab2's -161 rollout tail invisible in FQE value (-17.4, trio
best). NOTE (review 2026-10-03): the drift-detector reading is WITHDRAWN --
Stream B later showed FQE-vs-rollout disagreement on policies with no drift
story, so disagreement is not a drift signal. FQE-vs-rollout divergence is
uninterpreted until a validator is calibrated.
Next: Stream A (V1 validation splits BC/RvS).

## Stream A done: V1 validation splits, gate FAILED on rollout (2026-10-03)

New `blue/blue_val_split.py` (episode-blocked 80/20, Prechelt stopper
+ best-restore) wired into `blue_bc_pretrain.py` + `blue_rvs_pretrain.py`
(--val-frac/--val-seed/--early-stop-patience; 0 = legacy). Class weights
now train-episodes-only. BC gate: stopped == legacy (best epoch 30 =
last; 12-ep log, 2-ep val, no overfit signal). RvS gate on
offline_logs_14 (5 val eps): best val @25, restored; rollout target -50
regression: legacy(epoch30) -81.8 vs stopped(@25) -101.9, paired -20.1
(5/8 seeds worse). Manifest `val-v1-20261003.json` + pool manifests
`rvs-v1legacy/rvs-v1stop-20261003.json`. Mechanism works (selects,
restores, reports) but stopped underperformed last-epoch by paired -20.1
on ONE run, 8 seeds, deltas -78..+77: no evidence either way about val CE
as a selector (review 2026-10-03 softens the earlier
replication claim). Do NOT trust early stopping
from val CE alone. Next: rollout-proxy validators (true-FQE rank
stopped-vs-legacy ckpts; perturbed-trajectory val metric), then
Stream B (V2 ESS gating in IQL).

## Stream B done: V2 ESS-gated extraction (2026-10-03; numbers corrected on review)

`blue_iql.py`: `ess_frac` monitor + `auto_beta` bisection (MPO
dual-style) hitting `--ess-target 0.3`, `--awr-mode fixed/auto/binary`
(CRR), uniformity-escape restandardization; manifest records beta_used +
ESS. Reran a05-regime (tau .7/alpha .5) seeds 0/1/2: betas solved
0.72/0.022/0.193 (30x adaptation across spike regimes), ESS exactly
0.30, zero clipping all three. CORRECTION: rollout regression is auto
-89.5/-86.6/-76.0 (s0 was misreported -102.0; true mean of
-58/-38/-100/-93/-93/-58/-120/-156), NOT -102.0/-86.6/-76.0. Corrected:
mean -84.0 vs fixed -98.6 (+14.6), range 13.5 vs 26.1 (roughly halved) --
suggestive but n=3 per arm on selection seeds, no evidence either way;
all earlier claims built on -102.0 are withdrawn (see CORRECTION note in
`ess-v2-20261003.json`). FQE gate fails: order inverted vs rollout,
UNSTABLE, worst for most off-log s2 (resid 4.4). Ordered follow-up:
fixed seeds 3,4 + auto seeds 3,4, then held-out. Manifests
`ess-v2-20261003.json` + `iql-v2auto-s{0,1,2}` pool.

## Stream D done: V6 hidden-reset (2026-10-03; headline softened on review)

`GreedyCheckpointPolicy(reset_interval)` + registry pop-through +
`--reset-interval` pool flag (RvS excluded). Cache key verified by code
inspection to include reset_interval (policy_kwargs in _cell_key + ckpt
sha + source bytes), so ri0/ri1/ri50 runs are keyed apart. stab2:
carried historic -161.2 (per-seed -105/-59/-126/-62/-543/-48/-117/-230,
median -111) vs ri1 -131.4 (with -465 @7704) vs ri50 -194.8 (with -832
@7702) -- single runs where one outlier dominates each mean (non-outlier
means ~-84 ri1, ~-104 ri50; medians -77.0/-112.5 vs carried median -111)
-- could equally be noise in an already volatile policy, NOT evidence of
catastrophe-shifting. Solid finding:
control a05 ri1: -82 -> mean -1725 (7/8 seeds below -1300): memoryless
scoring destroys a good ckpt, so the GRU carries obs-absent information
and resets are no cheap fix. Caveat cutting the other way: ri1 hurts a05
far more than stab2, so stab2's tails are NOT pure memory drift either.
Missing: paired ri0 rerun on same code path, hidden-norm inspection.
Ordered: stab2 ri0/ri1/ri50 reruns with medians. Manifest
`drift-v6-20261003.json` + three pool manifests (verdict text there also
softened).

## Review follow-up: corrections + ordered experiments (2026-10-03)

Review caught a wrong s0 mean (-102.0 reported, -89.5 true: per-seed
-58/-38/-100/-93/-93/-58/-120/-156). ess-v2 manifest carries a
CORRECTION note (not silent); B/D/A verdicts softened; drift-detector
claim withdrawn; cache key verified by inspection (reset_interval in
_cell_key via policy_kwargs). One further correction TO the review:
ri50 non-outlier mean recomputed -103.7, not ~-89; medians
ri0/ri1/ri50 = -111/-77/-112.5.
Held-out (7801-7808, teacher lancer_v2 -85.1): v2auto s0 -77.1, s1
-105.8, s2 -136.8 (tail -413 @7804). s0 beats teacher on means, single
run, needs replication; selection-best s2 becomes held-out-worst --
transfer gap hits the best selection ckpt hardest.
5-per-arm (fixed s3/s4, auto s3/s4 new): fixed mean -99.8 median -105.6
range 36; auto mean -90.8 median -89.5 range 32.1; same-seed paired
+9.0 mixed (3 better/2 worse). No evidence on spread.
stab2 paired reruns: ri0 fresh-compute reproduces historic bit-exact
(-161.25, determinism across code versions); ri1/ri50 served 8/8 from
content-hash cache (bit-identical repro). Per-seed ri1-ri0: 4 better /
4 worse -- pure redistribution, noise verdict stands.
Manifest `review-followup-20261003.json`. Standing: nothing demonstrates
a reliable learned > heuristic; every positive is selection-seed and/or
single-run.

## Phase C: local speedups landed (2026-10-02)

No cloud needed. Four changes on blue/mappo-training: (1) `--q-epochs`
auto-scales Q-iters to data in blue_iql.py (210-epoch stable regime;
codifies the Phase A lesson); (2) `--workers` fork-parallel collection
in blue_collect_bc.py, verified bit-identical vs serial; (3) per-worker
policy cache in the pool (`_cached_build`, reset-per-episode reuse proven
identical by new test); (4) persisted content-hash cell cache
(blue/.cache/, gitignored; key = code + ckpt bytes + config + seed) —
repeat teacher runs go 24.5s → 0.0s. Sim 114/9; torch 27 pass. Next: 16
(JSRL/AWAC from IQL ckpt) + 17 (critic audit), now 5-10x cheaper to run.

## Repo layout cleanup (2026-10-02)

cage-challenge-4/ is now the pristine simulator only (CybORG/ untouched
since bootstrap — verified zero commits; plus setup/Requirements/README/
LICENSE/visualise/team-guide). All Blue code moved to blue/ (23 blue_*.py
+ cc4_epymarl_wrapper.py + tests_blue/ + results/); plan docs to
docs/archive/; pptx+pdf to assets/. Only bridge: wrapper derives the
cage path from __file__ (not cwd). Run Blue with cwd=blue/ (results/ and
relative ckpt paths unchanged); manifests still root-anchored. Fixed
iforest hardcoded paths + telemetry ROOT + doc pointers (bulk sed) +
AGENTS.md ownership/plan pointers + README layout rule. Verified: sim
112/9, torch rvs/iql/factorized 20, heuristic pool smoke, torch spawn
cell rvs_7629=-93.0 identical to pre-move manifest. CI syntax check 200
files OK.

## Resume prompt for a new conversation

> Continue the Blue defender work in your checkout (was `G:\Projects\cyberally`
> on Windows; now also/instead WSL — paths below are relative). Read
> `AGENTS.md`, `docs/current-state.md`, `docs/status/blue-session.md`,
> `docs/status/blue.md`, `docs/contracts.md`, `docs/implementation-plan.md`,
> `docs/coordination/blue-action-space.md`, `docs/archive/blue-agent-plan.md`,
> and the correction at the top of `docs/archive/handoff.md`. Preserve my
> work: inspect `git status --short --branch` and fetch origin first. Blue
> work is on `blue/mappo-training`, pushed, tracking `origin/blue/mappo-training`
> (PR #1 merged to main; stale remote `blue` deleted). The `environment/`
> requirement/script drafts stay uncommitted for Environment; root `.gitignore`
> (`.venv-train/`, `third_party/`) is committed. Do not reset or rewrite
> shared history. If this is a fresh WSL clone, follow the "Fresh-machine
> (WSL) rebuild" section in this file first (it inlines the venv pins because
> `environment/requirements-*.txt` are uncommitted), then run both suites
> green before anything else. Next tasks: seed cycling for training resets,
> then longer multi-seed MAPPO runs until the policy beats round-robin; hand
> Environment the vendor recipe. Never report an unreproduced number as a
> result.

## Update rule

At each meaningful Blue handoff, replace "Current Blue changes," "Verification"
and "Next steps" with the latest state. Include branch and base commit, whether
changes are staged/committed/pushed, exact commands and results, open contracts
and the next task. Update `docs/status/blue.md` as the short index. Never report
a run that did not actually complete.
