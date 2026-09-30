# Integration contracts: proposal v0.1

**Status: proposed, not yet implemented or accepted as a runtime API.** The initial
team setup records a common target without pretending all adapters exist. Existing
CybORG APIs remain authoritative until tested adapters and fixtures are merged.
Log decisions/proposals in `docs/coordination/`; record accepted versions here.

## Invariants to preserve

- One shared simulator instance and one joint environment step per team timestep.
- One reset per episode; every agent's initial observation comes from that world.
- All random sources and scenario/attacker configs are reproducibly recorded.
- Five Blue agent names initially remain `blue_agent_0` through `blue_agent_4`.
  Keep real simulator host/subnet identifiers; do not invent a renamed topology.
- Distinguish actor-local observations, centralized training views and evaluation
  truth. Only evaluation consumes privileged Red-session labels by default.
- Normalize native shared team reward once. Do not add five copies of the same
  team reward. Log native reward separately from any experimental shaping.
- Preserve completed vs pending vs failed action identity and duration. Distinguish
  natural termination from horizon truncation according to the accepted scenario
  semantics; do not guess a training-bootstrap rule from a boolean alone.

## Proposed adapters

These are conceptual boundaries, not required exact Python signatures yet.

| Boundary | Producer | Consumer | Required semantics |
|---|---|---|---|
| Scenario factory | Environment + Red | Runner/Blue wrapper | Seed, horizon, scenario version, Red factory/config; fresh instance per episode |
| Blue observation adapter | Blue + Environment | Actor/trainer | Local host evidence, freshness/missingness, stable mapping and validity masks |
| Blue policy adapter | Blue | Evaluator | reset episode memory; act on local input; load checkpoint/config with version checks |
| Red strategy adapter | Red | Scenario factory/pool | Native valid action policy, replayable config, observable input scope and provenance |
| Training entry point | Blue | Retrain loop | Scenario mixture + budget + seed + optional checkpoint -> new artifact manifest |
| Episode evaluator | Evaluation | All roles | Common horizon/seeds, raw reward/events, separate truth labels, documented metrics |

Do not expose true attacker identity to the actor when testing attacker inference.
Training supervision for a belief estimator may use known training scenario labels;
execution must use its observable-history prediction, including uncertainty/errors.

## Proposed JSONL step record

One record per agent per tick; team reward duplicated in records must be aggregated
once per tick by the evaluator. `null` means unavailable, not zero. Example is an
illustration, not a claim that these events occurred:

```json
{
  "schema_version": "0.1-draft",
  "episode_id": "example-episode",
  "scenario_id": "cage4-default",
  "scenario_version": "pending",
  "seed": 7629,
  "policy_version": "heuristic-v1",
  "step": 12,
  "agent_id": "blue_agent_0",
  "action": {
    "requested": "Analyse",
    "target_host_id": "restricted_zone_a_subnet_user_host_0",
    "executed": null,
    "status": "pending",
    "fallback_reason": null
  },
  "evidence": {"last_analysis_age_steps": null, "anomaly_score": null},
  "team_reward_native": null,
  "terminated": false,
  "truncated": false
}
```

Actual executed action must come from verified controller semantics, not just the
requested action. Evaluation labels use a separate file/namespace that actor code
cannot read. Host compromise after remediation is measured independently of success.

## Strategy-pool item

Required proposal fields: `strategy_id`, `strategy_version`, `source_commit`,
`factory`, `config`, `seed_policy`, `scenario_constraints`, `discovered_round`,
`outcome_evidence`, `artifact_hash` if external code/artifacts are used.
Only trusted, reviewed strategy implementations are loaded. A sequence of actions
is a diagnostic trace; replay validity depends on observations and preconditions.

Pool versions specify sampling weights for old/new strategies. Evaluation suites
must not change silently when the pool grows. MAPPO collects fresh rollouts from
the chosen strategies rather than treating old trajectories as on-policy data.

## Policy artifact manifest

Required proposal fields: policy ID/version, source commit, training config/hash,
runtime/dependency profile, framework revision, scenario/feature/action/schema
versions, observation/action dimensions, preprocessing metadata, training seeds,
checkpoint location + SHA256, validation suite/version, metrics location and
load/evaluation command. Do not store credentials in artifact locations.

Weights live outside Git; manifests and small reproducibility configs are tracked.
Recipients verify hashes and compatibility before running in the project simulator.

## Metrics requiring agreement before implementation

- Native cumulative team reward and episode horizon.
- Compromised/root-compromised hosts and compromised host-time.
- Green service success/disruption and unnecessary Restore actions.
- Detection precision/recall and a clearly defined false-positive denominator.
- First compromise -> detection; detection -> verified clearance; unresolved and
  undetected incident proportions. Do not report averages only over successes.
- Simulation steps vs wall-clock inference/runtime must be separate units.

## Compatibility and approval record

Before merging a breaking change, fill in proposal ID, accepted schema version,
affected roles, reviewer/PR links, producer-first migration order and fixture tests.
Current accepted runtime contract: **none beyond existing CybORG behavior**.
This document becomes authoritative as individual seams are accepted and tested.
