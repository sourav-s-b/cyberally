# Finite-Red anchored PPO development pilot

Date: 2026-10-15. Branch: `blue/finite-red-check`, based on `2f44db1`.

## Decision

The bounded pilot **does not justify a confirmation cohort or a claim that Blue RL beats Lancer**. The anchored actor's deployed greedy policy exactly reproduced the guard-only policy on all 24 evaluation seeds for each of three training seeds: 72 paired episodes and 28,728 ticks had zero Blue action/host mismatches. Its +1.58 native-return difference from Lancer is the existing guard-only difference, not a learned improvement. The preregistered gate required anchored greedy return to improve on guard-only; its measured difference is zero for every training seed.

The current residual PPO arm was worse than Lancer under greedy deployment for all three training seeds. Anchoring prevented that large regression, but did not produce a useful greedy deviation within this budget.

## Frozen comparison

The pilot used a fresh source snapshot in ignored `blue/results/finite_red_anchor_pilot_v1/source_snapshot.zip`, with file hashes and protocol in `frozen_plan.json`. Both arms trained from scratch with native team reward, no compromise-state shaping, finite CPU Red, RNG seeds 0–2, eight iterations of four fresh 400-step episodes, and training environment seeds 7706–7709. The current arm used the existing residual actor. The anchored arm used categorical logits `log q + actor(features)`, where `q` assigns 0.9 probability to Lancer's top legal candidate and 0.1 across the other legal candidates, plus a reference KL term and a full-buffer update KL cap. All other Blue rules, including the age-80 guard, remained shared. Greedy zero-head preflight actions matched guard-only across four real simulator seeds.

Fresh Lancer and guard-only controls, and all six actors in greedy and sampled modes, used the same evaluation seeds 7665–7688. Sampled action RNGs were separate from simulator seeds. These evaluation seeds were used in earlier research, so this is a **development pilot**, not a new held-out superiority test. Three training seeds and eight iterations are also too small to establish a general effect.

| Policy / mode | Mean native return by training seed | Mean paired difference vs Lancer by training seed |
|---|---:|---:|
| Lancer | −151.79 | reference |
| Guard-only | −150.21 | +1.58 |
| Current residual, greedy | −175.71, −169.04, −211.12 | −23.92, −17.25, −59.33 |
| Current residual, sampled | −157.08, −172.17, −179.17 | −5.29, −20.38, −27.38 |
| Anchored, greedy | −150.21, −150.21, −150.21 | +1.58, +1.58, +1.58 |
| Anchored, sampled | −166.29, −159.08, −157.79 | −14.50, −7.29, −6.00 |

Anchored greedy minus current greedy was +25.50, +18.83, and +60.92 points by matched training seed, averaging +35.08. Its exploratory three-seed 95% t interval is [−21.11, +91.27], spanning zero. Anchored greedy minus guard-only was exactly zero for all three seeds; the apparent zero-width interval for anchored versus Lancer comes from three copies of the same guard policy and is **not evidence of independent learned gains**. Its mean worst-three return was −289.67, versus Lancer's −300.33, but it merely inherited the guard's tail outcome. All episode returns, including seed 7686's −468 guard/anchored episode, remain in `analysis.json`.

Training diagnostics support the deployment observation: anchored greedy agreement with Lancer was 1.0 in the final iteration for all three seeds, while sampled override rates were 10–11%. Current residual agreement was 0.44–0.72, with sampled override rates around 85–86%. The anchored update KL cap accepted all four epochs in the last iteration of each seed; this confirms only that the capped updates ran, not that they improved native return. The sampled anchored policies were below both Lancer and guard-only on average for all three seeds.

## Integrity and limits

The frozen analyzer completed successfully and marked `recommend_confirmation: false`. A separate read-only audit checked the source snapshot digest, every frozen source hash, all six complete manifests and eight-history lengths, policy versions and training configs, scorer and actor hashes, evaluator source hashes, all 12 actor report modes and seed lists, and the per-episode artifact hashes. Guard-only and anchored greedy traces have different branch labels (`sweep_scorer` versus `sweep_hook`), so their raw files differ; comparing the actual Blue action and host at every tick found zero mismatches across all 72 pairs.

The reference mass of 0.9 is strong enough that this actor did not change any greedy decisions on this block. That is an observed outcome, not proof that the mass alone caused it. The present comparison cannot distinguish insufficient training, weak actor updates, feature limits, or the fixed reference's decision threshold. Do not proceed to LLM Red or a larger confirmation cohort from this pilot. The next concrete step is a smaller diagnostic on a **less rigid, bounded deviation rule**: measure candidate logit margins and actor corrections at the saved decisions, then preregister one change that permits controlled greedy deviations while keeping the native reward and guard-only comparator. Use fresh evaluation seeds for any confirmation attempt.

Artifacts are ignored under `blue/results/finite_red_anchor_pilot_v1/`; implementation is in `blue/training/harness_anchor.py`, `blue/training/harness_anchor_pilot.py`, and `scripts/blue_anchor_pilot.sh`. No remote experiment was launched.
