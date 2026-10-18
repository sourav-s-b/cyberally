# GPT Blue main32 completed results

All five Kaggle version 1 jobs COMPLETE; remote logs contain PREFLIGHT PASSED and SHARD COMPLETE. All 15 final actor hashes, 32-iteration manifests, exact frozen configuration/source/runtime, matched episodes and cross-shard baseline returns validated by harness_experiment. Source 6942d584f15231a7f19add0a510a04f7ff3ad229.

Native mean return, higher is better. Shared 32 REUSED development episodes 8221–8252; five independently seeded training replicas, not 160 independent models. Intervals below are paired training-seed t95, df=4, conditional on this fixed development suite.

| Policy | Mean return | Delta versus Lancer | Delta versus guard |
|---|---:|---:|---:|
| lancer | -79.91 | — | — |
| guard | -78.22 | — | — |
| sleep | -2267.28 | — | — |
| random | -245.00 | — | — |
| RL zero | -74.59 | +5.31 | +3.62 |
| RL risk | -81.21 | -1.31 | -2.99 |
| RL both | -74.71 | +5.19 | +3.51 |

Primary BOTH arm versus guard: +3.50625, t95 [-3.14891, 10.16141]; versus Lancer +5.19375, t95 [-1.46141, 11.84891]. Predeclared requirement to beat BOTH controls not met. No learned superiority claim.

Risk minus zero: -6.61875, t95 [-17.57265, 4.33515]. Both minus risk: +6.5, t95 [-6.31759, 19.31759]. Descriptive comparisons do not establish ML benefit. Combined versus zero mean -0.11875. Do not select the zero arm as a proven winner.

| Training RNG seed | zero mean | risk mean | both mean |
|---|---:|---:|---:|
| 0 | -69.28 | -87.72 | -70.00 |
| 1 | -72.16 | -85.62 | -74.69 |
| 2 | -77.84 | -75.88 | -83.66 |
| 3 | -81.88 | -83.19 | -71.22 |
| 4 | -71.81 | -73.66 | -74.00 |

Worst evaluation episode across all models: zero -161, risk -225, both -187. All tails remain included in aggregate.json; no failed run removed. Final seeds 7809–8200 untouched.

Recommendation: hold further expensive scaling. Inspect requested-action hashes/input sensitivity and targeted episode traces to explain the risk arm deficit before changing inputs, policy or budget. Better detector PR-AUC has not translated into demonstrated defensive improvement.

Artifacts: ignored blue/results/gpt_harness/main32_status/aggregate.json and s0–s4/pilot/. Analysis ran under local analysis Python after /tmp runtime cleanup; remote training runtime equality was verified from manifests. No weights deserialized for statistical aggregation.
