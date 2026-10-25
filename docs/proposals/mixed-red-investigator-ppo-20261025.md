# Mixed-Red PPO investigator experiment

**Status:** implementation plan; training has not started.
**Purpose:** test whether an RL investigation specialist trained on fresh
Finite and Discovery episodes can improve the existing rule coordinator.

## Why this is the next experiment

The frozen specialist C0 audit (`blue-specialist-c0-audit-20261018`, seeds
8287–8300) found no repeated actionable failure in recovery, verification, or
coverage. It therefore does not support adding a second specialist. The latest
Finite-Red BC suite found a repeatable Finite gain, while Discovery and Random
comparisons were inconclusive. Keep the existing urgent response, verification,
and age-80 rules fixed; train only the investigation hook.

The current PPO hook starts with a zero residual over Lancer, so greedy behavior
matches the teacher at initialization. The actor sees Blue-visible candidate
features, not the Red regime. Training uses new simulator rollouts only.

## Frozen design

- **Training environments:** 400-step episodes; each PPO iteration contains an
  even Finite/Discovery mixture (with four episodes, two per Red, shuffled by a
  separate seeded sampler). The Red type is selected at episode reset and is
  not exposed to the actor.
- **Policy:** anchored categorical PPO residual over the Lancer investigator;
  fixed response, verification, age-80, action-validity, and fallback rules.
  Native reward is the training score. No privileged label, replayed transition,
  or additional shaping signal is used.
- **Training environment seeds:** 7600–7612. Five model RNG seeds: 0–4.
- **Evaluation environment seeds:** 7641–7656, separately evaluated under
  Finite and Discovery. The same seed is paired across arms within each Red.
- **Arms:** trained RL investigator + age-80 coordinator; age-80 coordinator
  alone; unguarded Lancer. Primary comparison is RL minus age-80 coordinator,
  which isolates the learned hook. RL minus Lancer is the full-system
  comparison. Existing BC results are context only because they used a
  different evaluation block.
- **Primary summary:** equal-weight mean of the paired RL-minus-age-80
  differences across Finite and Discovery. Report each Red separately, the
  interval across the five training RNG means, paired episode bootstrap
  intervals, worst return, 10th percentile, and coverage.
- **Success gate:** primary mean at least +5 with its 95% interval above zero;
  neither Red may have a point estimate below −5; coverage must be at least
  0.98 in every arm/model/regime. Otherwise retain the learned policy as
  exploratory and keep the scripted coordinator.
- **Evaluation budget:** 8 iterations × 4 fresh episodes per model (160
  training episodes total). Pause only at an iteration checkpoint and record
  whether the budget prevents the complete frozen evaluation.

## Implementation boundary

Only the PPO episode sampler and its manifests change. The simulator already
accepts a Red type at episode construction, so no simulator or observation
contract change is needed. The harness remains responsible for legal actions,
pending-action safety, response, verification, and coverage. A separate
learned manager and a second specialist remain deferred until new traces show a
measurable routing failure.

## Seed checks

The authoritative seed audit reported 7600–7612 and 7641–7656 free before this
plan was added. The trainer must re-check these ranges immediately before the
frozen run. Results and checkpoints remain outside Git; commit this plan and
source hashes before training.
