# Finite-Red BC vs DAgger pilot results

**Date:** 2026-10-18
**Branch/source:** `blue/finite-red-check`, frozen source commit `2f44db117d98791103fd99329005f7d56f84ffe9`
**Artifacts:** `blue/results/finite_red_dagger_bc_pilot_20261018/`
**Status:** complete; exploratory comparison on the frozen Finite-Red evaluation block.

## Question and protocol

Does on-policy expert-label aggregation (DAgger) improve on behavior cloning (BC) for the investigation specialist? Both methods imitate the same Lancer host ranking from Blue-visible redesigned candidate features. The actor does not receive Lancer's score. Native simulator reward measures the resulting policy. The outer age-80 guard, urgent remediation, and verification rules are shared.

Five independent model-training seeds per method were evaluated on the same 32 Finite-Red episode seeds (`7769–7800`). The prespecified unit is the model-training seed: average paired episode differences within each model, then calculate the two-sided 95% t interval across the five model-level means. The four context controls use the same evaluation block. This interval describes variation across these five trained models conditional on this seed block; it is not a fresh-block confirmation.

All 10 model manifests report `complete`. Their evaluation files each contain all 32 frozen seeds, actor and manifest hashes match, and all source-file hashes match the frozen plan. The four controls each contain 32 episodes. This confirms artifact and seed integrity for the reported comparison.

## Results

| Policy | Mean native return | p10 | Minimum | Mean Lancer agreement | Coverage |
|---|---:|---:|---:|---:|---:|
| BC | -144.57 | -213.5 | -301 | 0.383 | 1.000 |
| DAgger | -146.95 | -202.1 | -492 | 0.379 | 1.000 |
| Lancer | -167.97 | -223 | -577 | — | 1.000 |
| Age-80 guard | -161.56 | -222 | -434 | — | 1.000 |
| Sleep | -4258.91 | -5561 | -5993 | — | 0.000 |
| Masked random | -718.78 | -1066 | -1522 | — | 1.000 |

Paired differences below are policy minus baseline, in native return. Intervals are across the five training-seed means.

| Comparison | Mean difference | 95% interval |
|---|---:|---:|
| BC − Lancer | +23.40 | [+14.25, +32.55] |
| BC − age-80 guard | +16.99 | [+7.84, +26.15] |
| DAgger − Lancer | +21.02 | [+13.91, +28.13] |
| DAgger − age-80 guard | +14.61 | [+7.50, +21.72] |
| **DAgger − BC** | **−2.38** | **[−11.67, +6.91]** |

Per-training-seed DAgger-minus-BC means, paired over the 32 evaluation episodes, are:

| Model-training seed | Paired mean difference |
|---:|---:|
| 0 | +8.72 |
| 1 | −2.16 |
| 2 | −7.78 |
| 3 | −10.53 |
| 4 | −0.16 |

The complete per-episode returns and diagnostics remain in the artifact JSON files; no episodes were discarded. The DAgger minimum is driven by a small number of severe episodes, so its better p10 does not remove the worse minimum.

## Interpretation and next action

On this frozen pilot, both BC and DAgger are ahead of Lancer and the age-80 guard by the prespecified five-training-seed interval. This is encouraging evidence for the learned investigation specialist on Finite Red. It is still an exploratory result from one evaluation block and five trained models; confirm on a fresh audited seed block before making a broader performance claim.

DAgger did **not** improve over BC: its paired difference interval includes zero, its mean agreement with Lancer is slightly lower, and its minimum return is worse. Retain BC as the simpler current imitation candidate. Do not expand DAgger rounds or prioritize OEC on the claim that DAgger helped; OEC would need a separate, explicit efficiency hypothesis to justify a run.

For the specialist-hierarchy plan, treat the BC investigator as a candidate component, not as a confirmed winner for all Red suites. Keep the rule manager and all other specialists fixed while selecting and evaluating one additional specialist. Use a fresh seed block for confirmation and preserve a paired comparison against BC+existing rules, Lancer+existing rules, and the individual specialist arms before judging composition.

## Reproduction and retained artifacts

- Frozen design, hashes, and actual seed lists: `blue/results/finite_red_dagger_bc_pilot_20261018/frozen_plan.json`.
- Per-episode BC and DAgger evaluations: `blue/results/finite_red_dagger_bc_pilot_20261018/evaluation/{bc,dagger}/s*.json`.
- Controls and automated arm-vs-control summary: `blue/results/finite_red_dagger_bc_pilot_20261018/evaluation/controls.json` and `analysis.json`.
- The direct DAgger-minus-BC interval above was calculated from the retained paired episode rows, using the frozen five-training-seed unit.
