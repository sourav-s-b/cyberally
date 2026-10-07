# Proposed: opt-in Blue investigation harness v2

Producer: Blue on gpt-blue. Consumers/reviewers: Evaluation (execution protocol), Environment (cross-platform replay). Status: local draft, not an accepted shared contract or notification.

New version blue-harness-redesign-v2 adds actor event-memory columns, candidate-set context, all-host centralized critic context, exact sampled teacher-mixture policy and independent optimizers. Checkpoints declare algorithm, exploration, feature names, source/runtime and artifact hashes. Defaults: MAPPO, risk input, exploration .2, age guard80, native reward, full episodes. A2C is an explicit alternative. Old checkpoints remain incompatible and supported through their old runner. Greedy residual execution is diagnostic, not the sampled trained distribution. Additional policy RNG replicas are nested evaluation variation.

Tests: mask/permutation/likelihood invariants, busy-visible memory, decision-time credit, actor/critic gradient isolation, both update paths, resume identity and simulator smoke. No simulator, reward or existing shared schema is changed. Ground truth stays in diagnostic labels.

Rollout: review audit and resolve local/remote trained-policy replay mismatch; validate locked-runtime preflight with actor/request goldens; run bounded matched screen; only then propose integration. No teammate has been contacted through this document.
