# Local v2 policy-update audit

Source base: bda8da97b434397295ec6cf58fb04b970f69aabd on gpt-blue.
Additive read-only audit scripts are currently uncommitted. Models, scorer,
learner, simulator, reward, and final seeds remain unchanged.

The six-model sensitivity study completed. Mean native-return gain over the
matching untrained mixture, training seeds 0/1:

| Configuration | Seed 0 | Seed 1 | Average |
|---|---:|---:|---:|
| Reference | +0.250 | +0.125 | +0.1875 |
| Undiscounted credit | +0.375 | -0.625 | -0.125 |
| Higher exploration | -1.625 | -5.125 | -3.375 |

These are four reused development episodes and two nested policy replicas.
The independent training unit is n=2. No arm establishes useful learning gain.

## What the updates actually did

All six actor heads and encoders changed from their seeded initial states.
Final update KL ranges from 0.0000008 to 0.00001531; the configured stop
threshold is 0.02. PPO clipping is inactive in the recorded updates.
The evidence does not support treating clipping or the KL stop as the
binding constraint in these runs. Training critic explained variance does
not measure action quality or establish generalization.

Two frozen full-horizon sampled cells were instrumented. Both reproduced
native return and all 1,995 requested action hashes exactly. Each audit
checks the checkpoint hash after replay; no optimizer step occurs.

| Diagnostic cell on 8233 | Reference s0, replica 0 | Higher exploration s1, replica 1 |
|---|---:|---:|
| Trained return | -50 | -122 |
| Matching untrained return | -49 | -61 |
| Learned investigation decisions | 613 | 509 |
| Mean total variation from zero residual | 0.001779 | 0.001620 |
| Maximum total variation | 0.004721 | 0.007673 |
| Same-state paired sampler pick changes | 5 | 3 |
| Mean total variation after zeroing risk and risk entropy | 0.00004334 | 0.00002676 |

Total variation is probability mass relocation, not percentage of actions
changed or a performance bound. Same-state shadow picks use a cloned policy
RNG state and the zero-residual distribution. They remain on the trained
trajectory: they are not a separate untrained rollout or an action-value
estimate. Small distribution changes can still lead to different simulator
histories and large return changes. Direct risk-input sensitivity is small
in these sampled traces; this does not prove ML is useless in general.

The worse cell's first shadow mismatches occur at ticks 48, 54 and 262.
At tick 48 blue_agent_4 selects admin_network_subnet_router; its zero-residual
same-state draw selects office_network_subnet_user_host_1. At tick 54
blue_agent_3 selects operational_zone_b_subnet_server_host_0 rather than
operational_zone_b_subnet_user_host_3. These are candidate causal leads,
not established missed-compromise mechanisms.

## Bounded mechanism check

A single intervention was frozen before replay: replace only the tick 48
blue_agent_4 request with its same-state zero-residual draw. Consume the
trained draw normally; retain the policy RNG and all later policy rules.
Subsequent observation/action divergence is expected. No hidden labels enter
selection. Result is recorded in update_audit/single_decision_check.json.
A single reused-episode intervention cannot support superiority or attribute
all damage to a particular host without completion/observation/penalty traces.

Artifacts: blue/results/gpt_harness/v2_sensitivity/update_audit/ contains
all candidate host/base/residual/risk/probability records, parameter deltas,
update statistics, frozen replay hashes, and the mechanism check.

The completed single-request intervention returned **-64**, compared with
**-122** for the original actor and **-61** for its untrained control.
Thus the request replacement recovered 58 of the 61-point deficit in this
specific replay. Its downstream trajectory differs; this is evidence for
the consequence of the request change under this simulator seed/policy RNG,
not proof that the router was clean or that the user host was compromised.
Completion, observation and reward-event timelines remain needed to identify
the defense mechanism. No generalized rule or new trained model is introduced.

Validation: all six manifests complete; every recorded clip fraction zero
and KL stop false; two instrumented 399-tick/1,995-request replays match frozen
returns and full digests; summaries finite; checkpoint hashes unchanged;
three additive scripts compile; git diff --check passes. Existing broad tests
were not rerun because actor, learner and simulator implementation are unchanged.
