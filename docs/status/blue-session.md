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

## Resume prompt for a new conversation

> Continue the Blue defender work in your checkout (was `G:\Projects\cyberally`
> on Windows; now also/instead WSL — paths below are relative). Read
> `AGENTS.md`, `docs/current-state.md`, `docs/status/blue-session.md`,
> `docs/status/blue.md`, `docs/contracts.md`, `docs/implementation-plan.md`,
> `docs/coordination/blue-action-space.md`, `cage-challenge-4/blue-agent-plan.md`,
> and the correction at the top of `cage-challenge-4/handoff.md`. Preserve my
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
