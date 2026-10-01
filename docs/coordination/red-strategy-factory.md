# Proposal: Red scripted strategy factory

- Status: draft; local RED-01 baseline implementation, review pending
- Author role and branch: Red, `red/strategy-adapter`
- Producer: Red policy/config; Environment owns the shared scenario factory.
- Consumers: Environment, Blue training, Evaluation strategy pool/evaluator.
- Dependency: main `ad9e5c0423f42fc8970e4ffe1051c1cb47a9530c` (telemetry).
  No issue/PR posted or affected-role review requested on anyone's behalf.

## Problem and current behavior

The Blue wrapper hardcodes `DiscoveryFSRed`. No versioned, replayable Red
configuration or provenance seam existed. The simulator already accepts
`red_agent_class`, inspects its constructor and supplies the shared simulator RNG
plus local subnets to native finite-state agents. A generic wrapper/callable could
break that constructor inspection or change RNG consumption.

## Proposed contract change

`red.strategy_adapter.StrategyConfig` is frozen with exact fields:
`schema_version: red-strategy-v1-draft`,
`strategy_id: native-discovery-fs-red`, `strategy_version: 1`, and
`policy: DiscoveryFSRed` (all strings). This first version accepts no behavior
overrides or dynamically imported code. Defaults reproduce the current baseline.
External JSON must include every field; unknown fields/versions are rejected.

`red_agent_class(config)` returns the original native class, preserving constructor
and RNG behavior. The Red-owned draft `create_simulator(config, seed, steps)`
requires an explicit nonnegative integer seed, integer horizon >= 2 and a fresh
instance for every episode. It uses native Sleep Blue and native Green on the
unchanged full scenario. This is a smoke fixture, not the shared Environment API.

The manifest includes strategy ID/version, source commit and hashes, explicit
factory/config/config hash, seed policy, actual seed list/horizon and scenario
constraints. `discovered_round` defaults to 0 for the built-in baseline;
`outcome_evidence` and `artifact_hash` are null because there is no discovery or
external artifact. No compromise or effectiveness evidence is fabricated.

Simulation-tick Red diagnostics are separate from Blue observations/telemetry.
No privileged state is read or supplied to Blue. Existing masks, reward aggregation,
pending attribution, done behavior, topology, schemas and checkpoints are unchanged.
The smoke runner counts the native duplicated Blue team reward once per tick.

## Required branch changes

| Role | Files/behavior | Dependency | Acceptance test |
|---|---|---|---|
| Red | Adapter, versioned config, provenance, smoke tests | exact main above | Native baseline parity, deterministic reset, valid controller actions |
| Environment | Review ownership and factory/constructor seam | this proposal | Fresh same-seed scenario, original names/topology/RNG |
| Blue | Later additive strategy config selection in wrapper/trainer | accepted producer seam | Default trajectory and feature/action dimensions unchanged |
| Evaluation | Later strategy-pool/manifest consumer, isolated diagnostic reader | accepted producer seam | Version checks, explicit seeds, no Red diagnostic leakage |

## Merge and migration order

Review and merge the additive Red producer first. Environment confirms the shared
factory contract, then Blue/Evaluation opt into it on their own branches. No old
contract removal is required. Do not load arbitrary manifest `factory` strings as
Python code; the current producer supports only the reviewed native baseline.

## Validation and decision

Command: existing Python 3.12 interpreter, `-m pytest -q red/tests
cage-challenge-4/tests_blue`: **75 passed in 46.56 s** (15 Red + 60 Blue).
Two-seed CLI smoke at seeds 7629/7630: 39 native ticks each at horizon 40,
zero invalid executed Red actions, native Blue team return 0.0. Short development
smokes do not reproduce historical step-200 compromise metrics or establish
attacker effectiveness. Artifacts remain ignored under `runs/red-smoke-20261001/`.

Live baseline parity is tested at development seeds 7629/7630 with 40-step native
scenarios: adapted, native, fresh-instance and repeated-reset trajectories match
for Blue-visible observations, Red executed actions and rewards. No invalid
executed actions were observed in those tests. Artifact checks include source/config
hashes, sim-tick provenance and refusal to overwrite outputs. These are smoke
results, not full attacker effectiveness evaluation or accepted shared APIs.

Pending: Environment/Blue/Evaluation review, consumer integration, separated
training/validation/final suites, privileged evaluation metrics, variant policies
and pool sampling. No LLM integration or simulator-core edits are part of RED-01.
