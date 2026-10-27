# Frozen BC across the available Red suite

**Status:** Complete; all 864 planned episodes and frozen artifacts validated. Frozen evaluation plan is at blue/results/finite_red_bc_redsuite_20261018/frozen_plan.json.
**Question:** Do the five BC checkpoints trained against finite Red also beat Lancer when Red uses the other supported behaviors?

## Frozen comparison

The wrapper exposes finite, discovery, verbose, random and sleep Red. Source inspection and a regression test confirm VerboseFSRed inherits the finite agent's decision and transition code and only enables action/observation printing. Therefore verbose is reported as the finite behavior, not counted as separate evidence. Finite and its verbose alias reuse the complete fresh confirmation on seeds 8253–8284; discovery, random and sleep use the separately audited fresh block 8401–8432. No model is retrained. On each of the three distinct new Red modes, all five BC checkpoints and Lancer, age-80 guard, Sleep and masked-random controls share the same 32 episode seeds.

The new block had 864 episode jobs: three distinct Red modes × (five BC models + four controls) × 32 episodes. The first launch created partial discovery, sleep and verbose files. Those are retained; verbose rows are excluded as redundant. Discovery, random and sleep resumed on the same frozen block. Each worker pool was limited to three; disjoint Red-mode pools could run together, with per-task locks preventing concurrent writes. The scorer, evaluator, training plan, confirmation report, model files and seed list were frozen and checked on resume. All 27 planned reports contain 32 episodes with matching seed lists and frozen-plan hashes; model and manifest hashes match the frozen plan.

Primary result is BC minus Lancer, paired by episode within each of the five fixed model seeds, with a 95% t interval across those five means. “BC beats Lancer across the tested suite” requires the lower interval bound to be above zero on every Red mode, including the existing finite confirmation. If any interval overlaps zero, the all-regimes claim is inconclusive; a negative interval means the current BC models did not beat Lancer on that mode. Report guard comparison, full per-model results, p10, minimum and coverage as well.

## Results

All differences are BC minus baseline. Intervals are 95% t intervals over the five fixed BC model-seed means, after pairing episodes by seed. The same five models share training data, so these intervals do not measure new training-data variation.

| Red behavior | BC mean (p10 / min) | Lancer mean | BC − Lancer (95% CI) | Age-80 guard mean | BC − guard (95% CI) |
|---|---:|---:|---:|---:|---:|
| Finite | −149.75 (−211 / −450) | −180.00 | **+30.25** [+24.65, +35.85] | −169.25 | **+19.50** [+13.90, +25.10] |
| Discovery | −82.31 (−139 / −177) | −83.03 | +0.73 [−5.57, +7.02] | −88.50 | +6.19 [−0.10, +12.49] |
| Random | −204.64 (−345.1 / −420) | −205.78 | +1.14 [−4.49, +6.78] | −198.75 | **−5.89** [−11.53, −0.25] |
| Sleep | 0 (0 / 0) | 0 | 0 [0, 0] | 0 | 0 [0, 0] |

BC coverage was 1.0 in every distinct Red mode. Sleep Red produced zero return for BC, Lancer and the age-80 guard, so that row is a tie with no evidence of policy advantage. Sleep and masked-random remain context controls; on discovery, random and sleep their BC mean returns were respectively −82.31, −204.64 and 0, compared with masked-random means of −235.63, −418.22 and −196.19.

**Verdict:** the all-regimes gate failed. BC-with-Harness clearly beats Lancer and the age-80 guard on finite Red. On discovery and random Red, its interval against Lancer overlaps zero, so those comparisons are inconclusive. On random Red it also loses to the age-80 guard. Sleep Red is a tie. Do not claim that these BC checkpoints beat Lancer across the available Red suite.

The evidence supports BC as a strong finite-Red investigator candidate, not a general winner across Red behavior. The five models share the same training allocation, and intervals are conditional on one episode block per behavior. Verbose is only a logging alias of finite Red, not independent evidence. See `analysis.json` for per-model, tail, control and provenance details.

## Run and artifacts

Run scripts/blue_bc_redsuite.sh. The command resumes valid episode files. If it stops, rerun it after the process has exited. Aggregation is produced by scripts/blue_bc_redsuite.sh --analyze.

The detailed outputs remain in blue/results/finite_red_bc_redsuite_20261018/evaluation/; analysis is written to analysis.json. Do not change the Python evaluator after the plan is frozen; its hash is checked on each task.
