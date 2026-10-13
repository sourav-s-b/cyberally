# Blue investigation-input audit (2026-10-13)

## Decision

The frozen gradient-boosted score recognizes current compromise in many legal
candidate sets, but adding it directly to investigation priority reduced native
defensive return. Multiplying it by time since completed inspection reduced one
severe loss but remained worse than guard-only overall. Neither rule should be
promoted to the production policy or used to justify long training yet.

The more precise next input target is **expected value of an inspection**:
whether inspecting this host now is likely to reveal *new, actionable* evidence,
given what Blue has already observed and how recently it inspected the host.
That differs from the current label, “host contains a Red session now.” It
needs a decision-to-completion dataset and evaluation by independent episode.

## What was already in the learner

The architecture study does not feed the actor a bare snapshot. Its v2 input
already includes analysis/remediation ages and availability, tracker belief,
telemetry deltas, missingness, host role and zone, risk and novelty scores,
unreviewed positive-change memory with ages, and risk entropy. GRU, graph,
set-context and MLP variants were compared at 32 on-policy iterations. The
architecture results are development tests, and the partially completed
training-seed cohort cannot prove that more capacity is useless. It does mean
“add memory” or “make the network bigger” is too vague a prescription.

## Frozen mechanism check

- Exact finite-Red config from architecture plan SHA-256
  `c3df3baa1003cd0c61e8a0b8915c520e0abd4a03c683052d51d558dfcd351207`.
- Exact scorer SHA-256
  `ec4026a3c81c2746253f306fd7add00a07f42fe13533b856fb6b4d01b8402fcd`;
  its golden predictions matched locally with scikit-learn 1.7.2.
- Eight **reused development** episodes 8221–8228, 399 ticks each. Final
  evaluation seeds 7809–8200 were rejected in code and untouched.
- All policies used the same `HarnessPolicy(max_age=80)`, Lancer remediation
  rules, masks, episode configuration and scorer. Only the sweep ranking hook
  changed. No RL checkpoint was retrained.
- Privileged Red-session labels were attached to the audit row **after** all
  Blue actions for that tick were selected; they never entered the policy.
- The guard-only trace reproduced all eight returns in the frozen architecture
  study exactly.

| Seed | Guard only | Raw risk bonus | Risk × inspection age bonus |
|---:|---:|---:|---:|
| 8221 | -171 | -187 | -146 |
| 8222 | -79 | -201 | -130 |
| 8223 | -142 | -127 | -117 |
| 8224 | -120 | -133 | -126 |
| 8225 | -132 | -240 | -137 |
| 8226 | -98 | -408 | -96 |
| 8227 | -136 | -124 | -231 |
| 8228 | -184 | -185 | -250 |
| **Mean** | **-132.75** | **-200.63** | **-154.13** |

Raw risk beat guard on 2/8 episodes; freshness-adjusted risk on 3/8. The
freshness rule improved on the raw-risk rule on 6/8 but did not clear guard.
These eight attacked trajectories are correlated by policy and constitute a
bounded mechanism check, not a new performance claim or a hyperparameter
selection exercise. The coefficient 0.25 and age scale 80 were held fixed.

## Why a predictive score did not yield a good decision

On guard-only trajectories in the first 100 ticks, among legal candidate sets
with at least one currently compromised host, risk ranked such a host first
on an episode-macro average of 40.7% of sets versus 16.4% for the Lancer
score and 13.7% for uniform random ranking. For never-inspected candidates,
the corresponding numbers were 36.1%, 14.1%, and 14.0% across seven
episodes with positive sets. This is diagnostic current-compromise ranking;
candidate sets within an episode are correlated, and ranking the label is
not equivalent to improving native reward.

The concrete loss mechanism is visible on seed 8226. Raw-risk ranking
reselected the same agent/host within ten ticks at 228 of 697 sweep decisions,
139 while that host was labelled clean. Guard-only did so at 42 of 728
decisions, 41 labelled clean. Risk ranking incurred 22 phase-1 operational-A
local-work-failure penalties (−220 reward) where guard incurred none.
The episode trajectory changed with Blue's actions, so attack onset times
cannot be treated as a fixed counterfactual schedule.

The age-adjusted rule did **not** eliminate recency waste: across the eight
episodes it still made 34–111 same-host reselections within ten ticks per
episode, versus 9–47 for guard. The risk bonus can still outweigh the base
ranking even when its risk term is discounted. On 8227 and 8228 it lost 95
and 66 return versus guard. This rejects that specific fixed heuristic; it
does not reject explicit freshness as a learnable input.

Published mission reward weights were also checked as candidate inputs. At
a given phase all hosts owned by one local Blue agent have the same zone
weight, so those values cannot discriminate between that agent's legal host
targets. They were dropped from this change.

## Research interpretation and next experiment

The [CAGE4 environment paper](https://ojs.aaai.org/index.php/AAAI/article/download/35158/37313)
establishes the multi-agent, partially observed setting. Work on
[belief representations](https://proceedings.mlr.press/v202/wang23p.html)
and [memory traces](https://proceedings.mlr.press/v267/eberhard25a.html)
supports maintaining evidence over time, but this project already has
explicit evidence memory in its v2 actor. The
[value-of-information framing](https://proceedings.mlr.press/r15/chen17a.html)
is the useful distinction here: the value of *taking an inspection action*
is not the probability that a host is compromised at an arbitrary tick.
The papers motivate an experiment; they do not establish this design's
performance in CAGE4.

Next, collect completed `Analyse` requests with the **pre-request** Blue
features and event history, actual completion/result, and whether it yielded
a new confirmation. Keep privileged truth as a separate audit label. This
requires a distinct collector and episode-grouped train/validation/test
split because host rows and decision sets in one episode are dependent.
Evaluate confirmation yield and calibration before feeding an inspection-yield
estimate to the actor. A later matched, on-policy RL comparison must keep
training seed as the replication unit and compare against guard-only and
Lancer on fresh development episodes. No final seeds or long Kaggle training
are justified by this mechanism check.

## Artifacts and checks

Code: `blue/training/input_decision_audit.py`,
`blue/harness/inspection_value.py`, and focused tests. Ignored local raw
traces and manifests: `blue/results/input_audit_finite/` and
`blue/results/input_audit_freshness/`. `risk_bonus_probe.json` in the first
directory is the raw-risk arm. The audit explicitly checks final-seed
exclusion, frozen plan membership, frozen scorer hash and finite-Red config.

`pytest` focused tests passed (see branch handoff). This audit did not run
the full training suite or create a new policy checkpoint.
