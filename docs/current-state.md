# Current state: what exists, what is measured, what is only planned

**Last updated:** 2026-09-30
**Owner:** Blue (Sourav). Everyone else: read this before your kickoff prompt.
**Branch this describes:** `blue/foundation` at `58b6cc7`. `main` at `05999fb`.

This file exists because the other docs are a mix of design intent, historical
notes and superseded plans. Every number below was produced by running code in
this workspace on 2026-09-30. Anything not measured here is marked as planned.

---

## 0. Read this before anything else: `main` is missing the Blue fixes

| | `origin/main` (`05999fb`) | `blue/foundation` (`58b6cc7`) |
|---|---|---|
| `cage-challenge-4/cc4_epymarl_wrapper.py` | **broken** — `max_hosts=16`, `vecs[:self.max_hosts]` | fixed — `max_hosts=51`, raises on overflow |
| `cage-challenge-4/blue_action_masking.py` | **broken** — Restore unreachable, `CONFIRMED_ROOT` never set | fixed — validity/evidence masks, `VERIFY` state |
| `cage-challenge-4/tests_blue/` | **absent** | 20 tests, all passing |

If you branch from `main` you get the version that silently truncates host
observations, resets the shared simulator five times per episode, and cannot
reach `Restore`. **Environment, Red and Evaluation must wait for the Blue PR to
merge, or work from `origin/blue/foundation` and record that you did.**

The five specific defects in the `main` version, for reviewers:

1. `reset()` calls `cyborg.reset(agent=...)` once per Blue agent, so the five
   agents observe five different worlds.
2. `self.seed = seed` does not reseed; `cyborg.reset()` is called without it.
3. `max_hosts=16` and `vecs[:self.max_hosts]` discard hosts. `blue_agent_4`
   owns three subnets and had 38 hosts at seed 7629.
4. Host-local pending masks let a second host action overwrite `_awaiting`
   while the controller silently discards the new action.
5. Successful `Remove` sets ground-truth `CLEAN`, although Remove can return
   success without removing a privileged attacker.

---

## 1. What is real today

### Team code: 9 of 163 tracked Python files

All Blue work sits in wrappers around upstream CAGE4. **Nothing inside
`cage-challenge-4/CybORG/` has been modified** — 154 files are untouched
upstream.

| File | Role | State |
|---|---|---|
| `cage-challenge-4/blue_obs_features.py` | 10 features per host → vector | works |
| `cage-challenge-4/blue_action_masking.py` | evidence tracker + masks | works |
| `cage-challenge-4/cc4_epymarl_wrapper.py` | 5-agent joint env, EPyMARL-shaped | works, **not** an EPyMARL integration |
| `cage-challenge-4/blue_iforest_experiment.py`, `_v2.py` | anomaly detectors | **broken on this machine** (hardcoded `/home/sourav/...`), results not reproduced |
| `cage-challenge-4/visualise_cc4.py` | SleepAgent spread plot | untested here |
| `cage-challenge-4/tests_blue/test_foundation.py` | 20 live regression tests | all pass |
| `scripts/check_source.py` | syntax-only CI check | 163 files pass |

`red/`, `environment/` and `loop_eval/` contain **only** an `AGENTS.md` scope
file each. Zero code.

### No policy exists

There is no neural network, no EPyMARL registration, no checkpoint and no
training run. `README.md` and `blue-agent-plan.md` describe a proposed
recurrent MAPPO agent. Treat every MARL number anywhere in this repository as a
target, not a result.

---

## 2. Measured behaviour of the simulator

All values from `CC4MARLEnv` with the built-in `DiscoveryFSRed` /
`EnterpriseGreenAgent`, masked-random policy unless stated.

### Throughput — training is feasible

| Measurement | Value |
|---|---|
| Joint step | **44 ms** |
| `reset()` | 0.19 s |
| 400-step episode | ~18 s |
| 4 000-step rollout | ~3 min, single-threaded |

A 3 000-step MAPPO run is ~2.5 h single-threaded. Parallel rollouts are
possible but not yet built.

### Blue subnet ownership is static

`EnterpriseScenarioGenerator._generate_blue_agents` (`CC4/CybORG/Simulator/
Scenarios/EnterpriseScenarioGenerator.py:643`) hard-codes the assignment:

| Agent | Subnets | Routers | Observed hosts (5 seeds) | Bound |
|---|---|---|---|---|
| `blue_agent_0` | restricted_zone_a | 1 | 7–13 | **17** |
| `blue_agent_1` | operational_zone_a | 1 | 9–13 | **17** |
| `blue_agent_2` | restricted_zone_b | 1 | 7–13 | **17** |
| `blue_agent_3` | operational_zone_b | 1 | 8–14 | **17** |
| `blue_agent_4` | public_access_zone + admin_network + office_network | 3 | 29–38 | **51** |

Per-subnet bound = `MAX_USER_HOSTS(10) + MAX_SERVER_HOSTS(6) + 1 router = 17`.
`DEFAULT_MAX_HOSTS = 51` in the wrapper is agent 4's bound applied to everyone.

### Red commits, but Blue cannot see it

Seed 7629, masked-random. True state read from
`env.environment_controller.get_true_state(info).data` over every host in
`state.hosts`; the Blue view read from `env.views[agent][host]` for the same
agent's tracked hosts.

| Step | Red user sessions | Red root sessions |
|---|---|---|
| 0 | 1 | 0 |
| 100 | 13 | 21 |
| 200 | 17 | 27 |
| 300 | 18 | 15 |

Over the same three-seed run (7629, 7630, 7640), `host_ground_truth()` applied
to `blue_agent_4`'s own observation view returned **0 for every host at every
one of 399 steps**. So Blue's only evidence channel is `Analyse`; nothing in the
10-feature vector distinguishes a compromised host from a clean one until it has
been scanned. This is the central problem the agent is being built to solve, and
it is why "did Blue detect it" and "was it compromised" must be recorded as two
separate facts. Note the true-state count above is host-wide while the Blue-view
count covers agent 4's hosts, so the two numbers are not a like-for-like ratio;
both are recorded to show the direction and the size of the gap, not a ratio.

### The learning signal is sparse, phased and throttled

| Measurement | Value | Consequence |
|---|---|---|
| Steps with nonzero reward | **162 / 399 (41 %)** | GAE credit assignment is hard |
| First nonzero reward | step ~135 | one third of the episode is silent |
| Mission phases | `(134, 133, 133)` | phase 0 is pre-planning |
| Agent-ticks forced to Sleep | **1374 / 1995 (69 %)** | agent-wide pending lockout |
| Live nonzero features in a 170-dim agent observation | 22–39 % | observation is mostly padding and zeros |
| Episode end | `terminated=True` at step `steps-1` | `truncated` never fires on the native scenario |

The 69 % forced-Sleep figure is the biggest open risk. It comes from the
one-pending-action-per-agent rule being *agent-wide*: while an agent waits out a
2-step `Analyse` it has exactly one legal action. If that rate survives under a
trained policy, the effective decision budget is small and no policy fixes it.
Instrument it before MAPPO; Environment may need a simulator-level answer.

---

## 3. The action space wastes ~75 % of a shared policy head

Current: `max_hosts=51` for all five agents → `n_actions = 155`.

| Agent | Real hosts at seed 7629 | Permanently masked slots |
|---|---|---|
| `blue_agent_0` | 13 | **114** |
| `blue_agent_1` | 13 | **114** |
| `blue_agent_2` | 9 | **126** |
| `blue_agent_3` | 14 | **111** |
| `blue_agent_4` | 38 | 39 |

A single shared 155-wide actor head would spend most of its capacity on actions
that can never be legal. Proposed fix is in
[`docs/coordination/blue-action-space.md`](coordination/blue-action-space.md):
per-agent bounds of 17 / 51, giving agents 0–3 an observation of 170 and 53
actions, and dropping the concatenated critic state from 2550 to 1190.

**Not implemented.** Until it is, any training config written against
`155 / 510 / 2550` will load on paper and behave like the current wrapper.

---

## 4. Known traps for a coding assistant

1. **`origin/main` is broken.** Section 0. Check before branching.
2. **Dimensions in the old handoff are unsafe.** `cage-challenge-4/handoff.md`
   once instructed `obs_shape=160, state_shape=800, n_actions=50`. Those are
   marked superseded. Always derive shapes from `get_env_info()`.
3. **`TrueStateWrapper` is broken against this CAGE4 snapshot.**
   `SimulationController.get_true_state(info)` returns an `Observation` object,
   not a dict, so `TrueStateWrapper.py:62`'s `.pop("success")` raises
   `AttributeError`. `import CybORG.Agents.Wrappers` also pulls in Ray. Working
   pattern for privileged evaluation labels:
   `env.environment_controller.get_true_state(info).data` → `{host: {...}}`.
   Evaluation should build on that, not on the upstream helper.
4. **`blue_iforest_*.py` hardcode `/home/sourav/Projects/...`** and will not
   import on this machine.
5. **No torch is installed.** `ray`, `torch`, `torch_geometric` and `sklearn`
   are all absent. Nothing here can train yet.
6. **The venv is Python 3.12.10 and local-only.** The documented target is
   Python 3.10; `torch==2.2.0`, `gym==0.26.2` and `numpy==1.26.4` have cp310
   wheels but no cp312 wheel for torch 2.2.0. Environment owns the rebuild.
7. **`cage-challenge-4/team-guide.md` contradicts current rules.** It predates
   `AGENTS.md`. It proposes renaming CAGE zones to restricted/operational/dmz,
   trimming to 2–3 zones, and points at the `oxwhirl/epymarl` fork. All three
   are forbidden or wrong here. See the banner at the top of that file.
8. **Upstream `CybORG/Tests/test_cc4/test_blue_actions.py` cannot collect**
   because its conftest imports Ray. That is an absent dependency, not a
   passing test.

---

## 5. What each role does next

Full detail in `docs/implementation-plan.md`. The short version:

| Role | Branch | Immediate task |
|---|---|---|
| Blue | `blue/foundation` | get BLUE-01 merged; write the action-space proposal; then BLUE-02 heuristic baselines |
| Environment | `environment/reproducible-runtime` | install Python 3.10, rebuild venv, split base vs training dependency profiles, record upstream provenance |
| Red | `red/strategy-adapter` | blocked on the runtime; then scripted strategy factory with no privileged state toward Blue |
| Evaluation | `eval/baseline-harness` | blocked on the runtime; metric semantics and privileged-label extractor against the pattern in trap 3 |

Baseline reproduction is the gate for every RL claim. The historical targets
are unmasked-random-safe sleep = 60 compromised hosts, masked random = 48, both
at step 200 on seed 7629. Neither has been reproduced. Both CAGE4 papers found
strong heuristics beat every MARL submission, so the bar is the heuristic, not
random.

---

## 6. How to tell a result from a target

- **Measured**: reproduced in this workspace, command and seed recorded here.
- **Historical**: from a previous session, not yet reproduced. Marked as such.
- **Proposed**: a design or roadmap item. `docs/contracts.md` is entirely
  proposed; no accepted runtime contract exists beyond upstream CybORG.
- **Aspirational**: from the review deck. Sub-100 ms actions, MTTR under 5 s,
  >90 % false-positive reduction, Docker/eBPF containment and zero-day
  resilience are **not** results and are not achievable inside CybORG.
