# Proposal: blue-action-space

- Status: draft
- Author role and branch: Blue, `blue/foundation`
- Producer: Blue
- Consumers: Environment (runtime/config), Evaluation (metric and log schema),
  Red (any agent-count assumption)
- Related issue/PR and exact dependency commits: depends on BLUE-01 wrapper
  fixes in `7b76fc2` and `58b6cc7`, merged into `main` as `f8a9deb`

## Problem and current behavior

`cc4_epymarl_wrapper.py:24` sets one host capacity for all five Blue agents:

```python
DEFAULT_MAX_HOSTS = 3 * (EnterpriseScenarioGenerator.MAX_USER_HOSTS
                         + EnterpriseScenarioGenerator.MAX_SERVER_HOSTS + 1)   # 51
```

51 is `blue_agent_4`'s bound, because only agent 4 owns three subnets.
`EnterpriseScenarioGenerator._generate_blue_agents` hard-codes ownership:

| Agent | Subnets | Per-subnet max | Agent max |
|---|---|---|---|
| `blue_agent_0` | `restricted_zone_a` | 10 user + 6 server + 1 router | **17** |
| `blue_agent_1` | `operational_zone_a` | 17 | **17** |
| `blue_agent_2` | `restricted_zone_b` | 17 | **17** |
| `blue_agent_3` | `operational_zone_b` | 17 | **17** |
| `blue_agent_4` | `public_access_zone` + `admin_network` + `office_network` | 17 | **51** |

Measured host counts over seeds 7629, 7630, 7640, 7650, 7660:

| Agent | observed hosts | routers | 51-slot padding |
|---|---|---|---|
| 0 | 7–13 | 1 | 38–44 slots |
| 1 | 9–13 | 1 | 38–42 |
| 2 | 7–13 | 1 | 38–44 |
| 3 | 8–14 | 1 | 37–43 |
| 4 | 29–38 | 3 | 13–22 |

Consequence at seed 7629, `n_actions = 2 + 3*51 = 155`:

| Agent | real hosts | permanently masked action slots |
|---|---|---|
| 0 | 13 | **114** |
| 1 | 13 | **114** |
| 2 | 9 | **126** |
| 3 | 14 | **111** |
| 4 | 38 | 39 |

This is *correct* but *wasteful*. `blue-agent-plan.md:59` states the intent as
"five local actors", which implies non-shared heads sized per zone. A single
shared 155-wide head would spend roughly 75 % of its output logits on actions
that can never be legal for four of five agents, and the critic would consume
2550 dimensions of which about 1270 are structural padding.

## Proposed contract change

Derive capacity per agent from the scenario's own subnet assignment instead of
a single global bound. Keep real CAGE4 hostnames; rename nothing.

Before:

```python
DEFAULT_MAX_HOSTS = 51
n_actions    = 155        # same for all agents
obs_size     = 510        # same for all agents
state_size   = 2550       # obs_size * 5
```

After:

```python
AGENT_SUBNET_COUNTS = (1, 1, 1, 1, 3)          # mirrors the scenario generator
PER_SUBNET_MAX_HOSTS = (MAX_USER_HOSTS + MAX_SERVER_HOSTS + 1)   # 17
max_hosts = [17, 17, 17, 17, 51]
n_actions = [53, 53, 53, 53, 155]
obs_size  = [170, 170, 170, 170, 510]
state_size = 1190                              # 170*4 + 510
```

Semantics that do **not** change: one reset per episode, one pending action per
agent, validity vs evidence masks, `VERIFY` after successful remediation, and
`terminated` vs `truncated`. Only the fixed slot count per agent moves.

Bump `WRAPPER_VERSION` from `foundation-v1` to `foundation-v2` and record
`feature_version` and `action_version` in every artifact manifest.

Rejected alternatives:

- **Keep 155 and document it.** Cheapest, but wastes policy capacity exactly
  when the observation is already 61–78 % padding. Defer.
- **Relocate live hosts to the low indices.** Does not reduce the count.
- **Shared per-host encoder with masked pooling.** The correct long-term answer
  for variable topology, already logged as BLUE-08. Explicitly deferred until
  host coverage and padding are correct — it must not conceal missing hosts.

## Required branch changes

| Role | Files/behavior to update | Dependency | Acceptance test |
|---|---|---|---|
| Blue | `cc4_epymarl_wrapper.py` per-agent bounds; `CC4BlueWrapper` facade; `get_env_info()` returns per-agent shapes; `get_state()` recomputes | BLUE-01 merged | `tests_blue`: every agent reports `max_hosts` equal to its own bound, no permanently masked slot is inside its live range, critic state equals the sum of live observations |
| Environment | `environment/` configs and any pinned shape constants; do not hardcode 155/510/2550 | Blue foundation-v2 | a seeded reset on a clean install reports the same per-agent shapes |
| Evaluation | JSONL records gain `max_hosts` and `n_actions` per agent; metric code must not assume a shared width | Blue foundation-v2 | fixture test: a 5-agent episode logs 5 different widths |
| Red | none, unless a strategy config references agent action counts | — | confirm none exist |

## Merge and migration order

1. Blue lands additive per-agent reporting on top of BLUE-01, keeping the
   global-bound behaviour available behind a flag so `main` stays runnable.
2. Environment and Evaluation read per-agent shapes; no checkpoint exists yet,
   so no compatibility burden.
3. Remove the global-bound default in a separate, explicitly breaking change.

Any checkpoint, config or manifest produced before this lands is invalid for a
foundation-v2 loader. There are none, so the cost is zero today and high later
if we train first.

## Validation and decision

Measured 2026-09-30 on `blue/foundation` (`58b6cc7`) with
`CC4MARLEnv(steps=400)`, `EnterpriseScenarioGenerator`, `DiscoveryFSRed`,
`EnterpriseGreenAgent`, seeds 7629–7660. Host counts and router counts read from
`env.hostnames[agent]`; action counts from `env.get_avail_actions()`.

Not yet validated: the foundation-v2 code itself, the test changes, and a
rollout under per-agent bounds. Affected roles (Environment, Evaluation) have
not reviewed this draft. Do not treat it as accepted.
