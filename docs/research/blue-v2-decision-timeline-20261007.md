# What the single-request replay actually establishes

On gpt-blue, frozen more_exploration_s1, development episode 8233,
policy RNG 8208302. Original model hash:
`df44e2710453cb1bd6a7a96facf8ed573bc82dc3dc367c832cea0e0d83ce6d83`.
Source base bda8da97b434397295ec6cf58fb04b970f69aabd; additive trace scripts
are uncommitted. No model update, simulator modification or final-seed use.

## Result and accounting

All three 399-tick, 1,995-request timelines reproduced the saved native
return and full request digest. Every tick's diagnostic penalty-event sum
was checked against native reward. Blue evidence and privileged events are
saved in separate files under `v2_sensitivity/decision_timeline/`.

| Replay | Action costs | Scored attack/service penalties | Native return |
|---|---:|---:|---:|
| Original trained sampler | -70 | -52 | -122 |
| One request changed | -64 | 0 | -64 |
| Untrained matched sampler | -56 | -5 | -61 |

The 58-point improvement consists of six fewer action-cost points and 52
fewer scored attack/service penalties. It is a change in this episode's
outcome, not an established gain from better host prioritization.

## Corrected interpretation of tick 48

Both the admin router and office user host 1 have UNKNOWN beliefs, no
completed analysis, no observed file data, and no Red-session label at
request tick 48. The trained marginal distribution gives the office host
0.529701 probability and the router 0.029711. The zero-residual distribution
gives 0.529224 and 0.029224 respectively. Thus the actor does **not** prefer
the router; the changed request is a sampled outcome near a distribution
boundary. This is not evidence that ML ranked the router as more dangerous.
Blue risk scores are 0.123203 for the office host and 0 for the router.

Original: router analysis completes at tick 50; office analysis completes
at 52, both without detection. Modified: office analysis completes at 50,
also without detection. There was no compromise detected by that early swap.

The changed investigation sequence actually postpones finding admin user
host 5: detection at 56 instead of 54, Remove completion at 59 instead of
57, Restore completion at 66 instead of 64. This rules out claiming that
this intervention helped by repairing that host sooner.

## Shared randomness is a verified limitation

An additional 80-tick diagnostic replay checks every Blue request against
the saved timeline and fingerprints simulator RNG state before selection,
after selection and after stepping. It does not report a truncated return.

- Policy selection does not consume the simulator RNG in either prefix.
- State and every inspected Red/Green agent share the controller RNG object.
- RNG states first differ after request tick 63, when the original Restore
  completes and the modified run still has it pending.
- The Red/Green action histories first differ at request tick 64.
- The Red-session host sets already differ after the tick-63 step.

Upstream SimulationController assigns its RNG to agents and shuffles action
ordering with that RNG. Analyse, Remove and Restore also select a session
through state.np_random. These facts explain why a fixed initial seed does
not preserve the subsequent exogenous attack/workload trajectory once Blue's
execution history changes. We have **not** isolated the particular internal
random draw responsible for the tick-63 split; no simulator hooks were changed.
The return benefit combines downstream defense changes and different realized
attack/workload paths. It cannot be assigned entirely to either component.

## Where real scored losses occurred

Original scored penalties are:

| Host/event | Total |
|---|---:|
| operational A user host 0, failed local work | -23 |
| operational A server host 0, Red Impact | -4 |
| contractor server host 0, Red Impact | -15 |
| contractor user host 1, Red Impact | -5 |
| contractor server host 1, Red Impact | -5 |

The first scored non-action-cost penalty occurs at request tick 96, an
Impact on operational A server host 0. This is far downstream of the swap.
Contractor server host 0 has no Blue investigation/remediation completions
in this trace; ownership/action availability must be checked before treating
its loss as a missed local Blue decision.

Operational A user host 0 is a useful next mechanism target: analysis
completed at 38 with UNKNOWN belief; a Red-session label first appears at
request tick 38; the next analysis completes only at 136 and CONFIRMS it.
Remove completes at 139 and Restore at 146. Failed local-work penalties
start at request tick 112. The 98-tick investigation interval is observable;
the trace does not prove that an earlier scan would detect compromise or
prevent all 23 penalty points. Age guard 80 is an intervention trigger,
not a hard revisit bound while urgent work/busy actions defer investigations.

Original office user host 1 gets a Red-session label from tick 125 onward,
but analyses at 166, 236, 250 and 334 return empty Files and UNKNOWN belief.
Untrained: Red-session label first at 176, an empty scan at 184, positive
unknown files only at 194, followed immediately by remediation. A Red-session
label is diagnostic truth, not a promise that Analyse can reveal it; it may
include session states that are not equivalent to currently observable damage.
No classifier or policy is allowed to consume that label.

## Next design decision

Do not deploy an office-host rule, an increased exploration setting, or a
V(s)-based override gate. The single-swap result is not evidence for those.
Next bounded check should inspect Blue-visible evidence and legal/busy
status for operational A user host 0 between ticks 38 and 136, while urgent
server remediation occupies the zone. Establish whether an earlier detectable
signal was missed, whether forced workload limits investigation capacity,
and what fraction of native penalties Blue can directly influence.

For broader comparisons retain multiple episode seeds and training seeds;
shared-seed comparisons remain valid sampled simulator outcomes, but not
identical attack sequences. An independently controlled RNG/attacker replay
contract would require an Environment/Evaluation proposal and review before
integration, not an unannounced simulator patch.

Validation: three complete return/request-digest reproductions; all native
penalty sums checked; frozen weights unchanged; 80-tick RNG prefixes match
saved requests; two focused logging tests pass; scripts compile; diff check
passes. The first incomplete trace is retained separately. No new training,
Kaggle run, or artifact upload.

## Follow-up: what an immediate scan could have done

The complete host-gap trace shows Blue's process list for operational A user
host 0 grew from 2 entries to 12 at the tick when the Red-session diagnostic
first appears. A new SSH connection points to operational A user host 1.
This is Blue-visible telemetry, though this trace alone does not prove the
connection is malicious.

At the next sweep decision (request tick 38), the frozen risk model scores
user host 0 at **1.0** (novelty percentile 0.8847). The actor's sampler still
chooses the Lancer teacher's router, with probability 0.606. User host 0 has
probability 0.0743 under the trained model and 0.0729 with its residual
zeroed. So the input is present and high, but training barely changes the
selection odds at this state; the teacher mixture and base priority dominate.

A short, predeclared mechanism check replaced that one request with an
Analyse of user host 0. It reproduced every original request through the
intervention, with no earlier differences. The analysis completed at tick
40, returned an UNKNOWN file (`cmd.sh`), and moved the Blue belief to
CONFIRMED. The frozen actor then requested Remove at tick 40. The original
policy did not next analyze this host until request tick 134 / completion
136, when the scan returned `cmd.sh` and `escalate.sh`.

This establishes that an earlier scan can reveal actionable evidence in
this episode, and that this policy did not act on a risk score of 1.0 with
meaningful probability uplift. It does not establish that the file was
available earlier than tick 40, that Remove completed successfully, or that
this intervention improves native return across episodes. The partial
42-tick reward is deliberately not compared as an episode score. Once the
request changes, shared Red/Green simulator randomness and the attack path
may diverge.

Next useful comparison: on several already-used diagnostic episodes, measure
how often a high Blue-visible risk score precedes a successful detection, how
often the teacher mixture selects elsewhere, and whether the subsequent
Remove is completed and verified. Keep privileged labels out of policy
selection and report each episode; do not change the model or launch a long
training run from this single example.
