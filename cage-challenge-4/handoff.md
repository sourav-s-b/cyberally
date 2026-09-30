# Handoff: CAGE4 Blue RL + masking — everything the next AI needs

## 2026-09-30 design-review correction

Read `blue-agent-plan.md` before following the historical training instructions
below. The code was reviewed against the three supplied papers and the zeroth
review slides. No new training run has validated the historical metrics.

Static inspection found blockers: MARL reset resets the shared simulation five
times; seed assignment does not reseed it; 16 host slots can truncate observations
and targets (agent 4 owns three subnets); pending actions can be overwritten by
actions for other hosts; passive Monitor telemetry is not consistently ingested;
Restore is unreachable through the normal detector state transitions; successful
Remove is incorrectly treated as proof of cleanup. EPyMARL constructor options,
tensor actions, common reward and lifecycle integration also need work.

Monitor runs automatically and can expose process/connection events. The earlier
claim that detection data appears only through Analyse is too broad. Likewise,
the 160/800/50 observation/state/action dimensions are not safe defaults for the
full scenario. Reproduce baselines after fixing these correctness issues.

The recommended contribution is temporal evidence and uncertainty in recurrent
MAPPO plus measured adaptation/retention across Red strategy-pool retraining.
Include strong heuristics in evaluation. The slides' real Docker/eBPF containment
and wall-clock performance targets remain separate from the CybORG prototype.

## Historical TL;DR (superseded where noted above)
Blue-side pipeline is built and smoke-tested. Remaining: install EPyMARL,
register the env, train MAPPO, beat Random (48 total / 35 root @ step 200).

## Files (all in repo root, all verified live against seed 7629)
| File | What | Status |
|---|---|---|
| `blue_obs_features.py` | v2 featurizer (10 dims/host) + shared `extract_subnets`/`is_external` | done, tested |
| `blue_action_masking.py` | `MASK_RULES`, `detection_hit`, `BlueZoneTracker` (2-strike rule), `conn_leaves_zone` | done, unit-tested |
| `cc4_epymarl_wrapper.py` | `CC4BlueWrapper` (1-agent dev) + `CC4MARLEnv` (5-agent, EPyMARL-shaped) | done, smoke-tested |
| `blue_iforest_experiment.py` / `blue_iforest_v2.py` | detector experiments (IF degenerate 0.494; file-rule 0.875) | done, evidence only |
| `visualise_cc4.py` / `frames/` | SleepAgent spread visualization | done |

## Verified facts (don't re-derive)
- `reset()/step()` return `Results`, not tuples. Per-agent: `cyborg.reset(agent=...)`,
  `cyborg.step(agent=..., action=...)`. Joint: `env.step({agent: Action})`.
- `SimulationController.step()` **clears** per-agent obs sets every step — accumulate client-side.
- Blue's visibility ceiling (`get_agent_state`) shows **no red sessions**; Monitor deltas are
  empty; detection data appears **only via `Analyse(hostname)`** (2 steps: IN_PROGRESS then TRUE+detail).
- **IN_PROGRESS landmine**: resolve keys on terminal `success` (TRUE/FALSE), never on host-key
  presence (clean hosts resolve TRUE with no host key). Never re-issue while pending.
  Wrapper keys resolves on `self._awaiting`, not the current decode (controller drops new
  actions while one resolves).
- Signatures: `Sleep()`, `Monitor(session, agent)`, `Analyse/Remove/Restore(session, agent, hostname)`.
- `MonitorAgent.__init__()` takes no name arg — needs a wrapper for scenario use.
- Monitor ≡ Sleep bit-for-bit (20/35/51/60) — harness validated.
- Blue reward is **common** (-57 each @ 200 masked-random steps) — suits MAPPO.
- Topology varies per seed (12–16 hosts/zone) — wrapper pads obs to 16 hosts and actions to 50.
- File-rule detector: test AUC 0.875, 0/62 clean FPs, ~25% aimed-Analyse miss rate (hence 2-strike rule).

## EPyMARL wiring (exact next steps)
1. `pip install` EPyMARL deps (torch 2.2 already in venv; network works — shallow clone
   tested at `/tmp/epymarl`, repo `uoe-agents/epymarl`). Interface to match:
   `MultiAgentEnv`: `reset()`, `step(actions)->(obss,reward,terminated,truncated,info)`,
   `get_obs/get_state/get_avail_actions/get_total_actions/get_env_info` — all implemented.
2. Register: in EPyMARL `src/envs/__init__.py` add
   `REGISTRY["cc4"] = lambda **kw: CC4MARLEnv(**kw)` (import from this repo).
3. Config yaml: mappo, `n_agents=5`, `obs_shape=160`, `state_shape=800`, `n_actions=50`,
   `episode_limit=400`. Env already returns avail masks — enable action masking in the
   MAPPO config (that's the whole point of the tracker).
4. Success metric: mean episode reward trending up. Then true-state compromise counts
   at step 200 vs baselines below.

## Baseline targets (true-state compromised hosts @ step 200, seed 7629)
| Policy | total (root) | Beat this |
|---|---|---|
| Sleep / Monitor | 60 (43) | — |
| cc4BlueRandomAgent | 48 (35) | **target: < 48 total, < 35 root** |

## Known limitations / open items
- `get_state()` = concat of 5 zone obs (fine for CTDE start; true-state infusion optional).
- Remove/Restore durations assumed from `action.duration` (Remove/Restore resolve path
  implemented but untested vs live root hosts — test before trusting remediation).
- `DeployDecoy`/`Block/AllowTrafficZone` excluded from discrete space (matches random baseline).
- Stealth flag: file features are strong vs DiscoveryFSRed; expect degradation vs stealthier Red.
- The `if idx...fallback to Sleep` guard means off-mask exploration is impossible by
  construction — EPyMARL must use the provided avail masks, not epsilon-greedy over all 50.

## Reproduce key numbers
- `venv/bin/python blue_iforest_v2.py` (~4 min): TEST AUC line.
- Baseline counts: `CC4MARLEnv` + masked-random rollout (see wrapper smoke tests in chat).
- Frames: `venv/bin/python visualise_cc4.py` → `frames/frame_{000,100,200}.png`.
