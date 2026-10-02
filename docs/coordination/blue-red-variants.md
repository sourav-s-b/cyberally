# Proposal: blue-red-variants

- Status: draft
- Author role and branch: Blue, `blue/mappo-training` (commit `b559b11`+)
- Producer and affected consumers: Producer Environment (simulator +
  scenario generator); consumers Blue (generalization eval), Evaluation
  (scenario suites / scoring).
- Related issue/PR and exact dependency commits: none yet; Blue base is
  `blue/mappo-training` at the red-config commit (see session log).

## Problem and current behavior

The published CAGE-4 analysis (Kiely et al., AAAI 2025) differentiates
agents on red/green variants Blue cannot currently reproduce:

- IncreasedPhishing (green phishing rate 0.01 -> 0.75) and StealthyRed
  (green FP rate 0.01 -> 0.75): `EnterpriseGreenAgent.__init__` already
  takes `fp_detection_rate` / `phishing_error_rate`, but
  `EnterpriseScenarioGenerator.py:741` constructs green agents with fixed
  defaults (`name, np_random, own_ip` only). There is no path from env
  config to these rates.
- AggressiveRed (all red action durations -> 1 step): durations are
  per-action-class attributes (`Action.py:22 DEFAULT_DURATION`); only
  test code overrides them per instance.

Blue's Phase-1 suite (proposal 15) covers red-agent *variety*
(Discovery/Finite/Verbose/Random/Sleep, all in-tree, Blue-side wrapper
flag) but cannot vary rates or durations.

## Proposed contract change

1. `EnterpriseScenarioGenerator(..., green_agent_kwargs=None)`: dict
   forwarded to the green agent constructor at line 741. Keys
   `fp_detection_rate: float`, `phishing_error_rate: float`; defaults
   keep current behavior (0.01/0.01). Additive, backward compatible.
2. Duration overrides: shape TBD by Environment. Suggestion: a
   scenario-level `action_duration_overrides: dict[action_name, steps]`
   applied post-construction (same mechanism as the existing tests), or a
   global `duration_scale: float`. Must not change default durations.
3. No changes to observations, masks, rewards, pending actions, or done
   signals. No checkpoint/config breakage: new kwargs default to current
   behavior.

## Required branch changes

| Role | Files/behavior to update | Dependency | Acceptance test |
|---|---|---|---|
| Producer (Environment) | `EnterpriseScenarioGenerator`: accept + forward `green_agent_kwargs`; duration hook per above | none | construct generator with rates 0.75/0.75, assert green agents carry them; default construction byte-identical behavior |
| Consumer (Blue) | `cc4_epymarl_wrapper.py`: `green_rates` / `duration_overrides` env flags + `get_env_info` + manifests; pool `--green-fp/--green-phishing` | producer merged | existing red-variant tests + new rate test (assert agents carry rates; short-episode completes) |

## Merge and migration order

Environment lands additive generator support first (defaults = current
behavior; main stays runnable). Blue wires flags + eval after. No removal
step (nothing deprecated).

## Validation and decision

Untested (draft). On merge, Blue runs the 8001-8008 suite under
phishing-0.75 / stealthy / aggressive configs and records manifests in
`docs/proposals/manifests/`. Not accepted until Environment reviews.
