# Gate utility ablation for Blue architecture

**Status:** frozen for a descriptive screen; do not interpret this two-RNG
comparison as confirmation of an RL improvement.

## Question

Does a learned state-conditioned gate add value over a fixed blend of the same
SetActor and candidate-attention branches, under the same MAPPO budget?

## Arms

1. `set`: SetActor residual policy.
2. `fixed`: the same SetActor and attention branches with a fixed 0.5 score
   blend; router parameters are present for matched actor geometry but do not
   affect the output.
3. `hybrid`: the same branches with the learned state-conditioned sigmoid gate.

All use the same strict Lancer action guard, scorer, reward, optimizer, update
settings, and training schedule. Evaluate strict Lancer and each architecture's
untrained initialization as controls. The untrained policies are not assumed
identical; retain and report the control results per arm.

## Schedule and seeds

- Two paired training RNGs: model RNG 0 and 1.
- Six iterations × four 400-tick episodes per arm/RNG (24 episodes), alternating
  Discovery and Finite Red as in the runner. This matches the completed hybrid
  budget.
- RNG 0 environment seeds: 8465–8472; RNG 1: 8473–8480. The eight seeds in
  each RNG block are cycled three times, equally across all three arms. Reuse
  across architectures is intentional pairing; blocks remain disjoint between
  training RNGs and evaluation.
- Evaluation seeds: 8417–8432, each run under both Discovery and Finite Red,
  for all trained arm/RNG cells. Controls use the same seed/regime pairs.
- Authoritative seed audit on 2026-10-25 reported both blocks unused. Re-run
  immediately before launch and record the output in the run manifest.

## Outcomes and diagnostics

Primary comparisons are paired native-return differences, separately by Red
type: learned gate minus fixed blend, and each trained arm minus its own
untrained control. Also report strict Lancer differences, mean, median, IQM,
paired win fraction, lower-tail return and bootstrap intervals. Report each RNG
separately; only two trained RNGs are available, so pooled seed rows do not
provide 64 independent trained models.

Capture per-decision gate values, SetActor/attention branch score difference,
valid candidate count, normalized Lancer score summaries, Red type (analysis
stratification only; never a policy input), action, and outcome. Test whether
the learned gate responds to Blue-visible context and branch disagreement.
Do not add a gate-balance penalty in this screen.

Advance to the value-aware router only if the learned gate beats the fixed
blend in both Red regimes and both RNGs on the macro-average, while improving
over its own untrained control and without a material lower-tail regression.
Otherwise record a negative or inconclusive screen and keep the larger router
deferred. Any positive screen requires a fresh, separately audited
multi-training-RNG confirmation before claiming a robust RL win.

## Execution

Three isolated architecture workers may run concurrently with one CPU thread
each. Each worker trains RNG 0 then RNG 1 and evaluates each completed model on
the common evaluation block. Preserve per-arm logs/checkpoints and manifests.
If machine load or memory pressure becomes material, reduce concurrency without
changing any model/config/seed settings. No other experiment is to share these
training processes.

The value-aware router proposal is deferred until this screen is analyzed:
`blue-value-aware-router-experiment-20261025.md`.
