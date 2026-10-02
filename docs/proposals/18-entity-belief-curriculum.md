# Proposal 18: Entity-RL actor + learned belief + red curriculum

- Status: proposed (P6–P8; cheap rungs first; actor/belief AFTER 14–17
  resolve the update question)
- Date: 2026-10-02

## Description

The architecture-and-environment track: carry the attention win into RL
(entity-Transformer + autoregressive host→command + GTrXL stabilization,
per the SR-DRL blueprint); replace hand suspicion with learned belief
(DVRL/VariBAD) with a no-training POMCP planning baseline first; harden
generalization with PLR⊥ replay over the seed/variant pool NOW (Blue-only,
no sim change) and ACCEL/PSRO after Environment hooks land. Explicit
ordering dependency: no big actor build until 14–17 show an update rule
that survives — a better actor trained with a diverging updater is waste.

## Motivation / evidence

W6: attention won by cross-slot comparison, but only as a cloner. SR-DRL
(F16) is the published blueprint for exactly our structure — GNN entity
encoder + host-then-command autoregressive policy TRAINED WITH RL and
generalizing to new sizes; our factorized head is halfway there. GTrXL
(F14) is the stabilization recipe the blueprint needs. UPDeT (F18)
decouples entity-obs from action-groups (our varying 17/51 slots + masks);
Thompson entity-RL (A28) already generalizes 10→40 cyber nodes.

W7: DVRL (G12) particle belief replaces brittle hand suspicion with
learned uncertainty; VariBAD (G16) formalizes probe-vs-remediate. But
POMCP (G23) — online planning with the SIMULATOR as the model, zero
training — tests the hypothesis first: if planning with the sim as model
beats the reactive teacher, belief/planning is confirmed as the gap and
the learned-belief build is justified; if not, skip it.

W5: PLR⊥/DCD (H5) replays high-regret seeds/variants with minimax-regret
guarantees and needs NO sim change — it is a harness-side prioritization
over our existing pool (seeds × red variants), implementable by Blue
alone. This is the curriculum we can run before Environment hooks exist;
ACCEL mutation (H6) and PSRO leagues (H15, Hammar T-FP line A13/A14) wait
on the coordination proposal (blue-red-variants.md).

## Experiment

1. Cheap rungs (days–1 week): POMCP baseline on regression-8 (sim as
   model, particle belief over compromise); PLR⊥ replay over the
   seed×variant pool measuring tail-episode (-414-class) reduction.
2. Actor (3–4 weeks, CONDITIONAL on 14–17 producing a surviving updater):
   entity-Transformer encoder over 51×17 slots → autoregressive
   host→command head (extends factorized 02) → GTrXL-gated RL training;
   UPDeT-style entity/action decoupling for 17/51 transfer.
3. Belief (conditional on POMCP positive): DVRL particle encoder or
   VariBAD context as the learned suspicion module feeding the actor;
   privilege rule unchanged (true state never a policy input).
4. Gate (binding): actor/belief must beat lancer_v2 held-out (-91.2);
   PLR⊥ must shrink held-out variance/tail vs uniform replay at equal
   compute, else it is scheduling theater.

## Pros

- POMCP + PLR⊥ are Blue-only, cheap, and decisive: each can confirm or
  kill a big build before it starts.
- SR-DRL path reuses the two things that already worked (factorized head,
  slot attention) instead of importing an alien architecture.
- Curriculum splits cleanly across the ownership boundary: harness-side
  now, sim-side with Environment later.

## Cons

- The ordering dependency is real: building the actor first risks a
  beautiful architecture trained by a diverging updater. Enforce the gate
  from 14–17.
- GNN/entity encoders add a topology dependency — our wrapper's entity
  view must be stable across red variants or transfer claims are void;
  pin the entity API (host_risk_features contract) before building.
- PSRO/ACCEL are the most expensive items in the whole survey (red
  training infra + Environment coordination); keep them behind PLR⊥
  results, not beside them.

## Explicitly deferred (with reasons, not neglect)

- Diffusion policies (C23–C25): built for multimodal continuous control;
  our 155 discrete actions + IQL stitching (14) cover the need cheaper.
- GAIL/AIRL (C8/C9): adversarial instability on top of our divergence,
  and a reward already exists — IRL answers a question we are not asking.
- Full DreamerV3 (G1): heavy; POMCP tests "model helps" first.
- Program synthesis / LEAPS / HIPO (I20–I22): interpretability is not the
  binding constraint; revisit only if auditability blocks deployment.
- Mamba (F23): no RL stabilization recipe as mature as GTrXL for this
  setting.

## Verdict

PROPOSED, P6–P8. POMCP + PLR⊥ now; actor and belief conditional on
surviving updaters from 14–17. The deferred list is closed unless its
stated conditions trigger.
