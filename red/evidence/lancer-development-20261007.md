# LLM Red versus CAGE Red against local Blue Lancer

Tested locally on 2026-10-07 using Blue dependency
`3096f068f43aa461ac3fe5e659c2e24e6aa86188` and the source hashes in the adjacent
JSON report. No commits, pushes or merges were performed. Local entry point
`red.main` uses LLM Red by default; `CC4LLMEnv` is the default-LLM environment
for Blue consumers. Existing upstream training configs have not been switched.

## Results

Full native 400-step horizon (399 executed ticks), fixed `hybrid_lancer_v2`,
native Green and full CAGE topology. Existing development seeds 7629/7630/7631.
Lancer here is the local priority-sweep variant with evidence-based remediation
and no decoys; it is not published competition Lancer.

| Seed | Improved LLM Red | CAGE DiscoveryFSRed | Same executor, no LLM |
|---|---:|---:|---:|
| 7629 | -676 | -52 | -134 |
| 7630 | -645 | -54 | -116 |
| 7631 | -719 | -59 | -90 |
| Mean native Blue return | **-680** | -55 | -113.33 |

Lower native Blue return favors Red. The improved LLM Red caused more mission
damage in all three development episodes against both controls. All nine
episodes finished with **zero invalid executed Red actions**.

| Other measure, mean per episode | LLM Red | CAGE Red | Executor only |
|---|---:|---:|---:|
| Active compromised host-ticks | 11,006 | 11,734 | 11,088.33 |
| Active privileged host-ticks | 3,480 | 4,823.67 | 4,008.33 |
| Wall time, seconds | 66.23 | 19.70 | 29.75 |

The new attacker favors mission damage; CAGE Red retained more privileged
host-time. Host-time counts unique hosts with active Red sessions after each
tick, including initial footholds. Root/SYSTEM sessions count as privileged.
These are draft evaluation-only diagnostics, never policy inputs. Episode wall
time includes model inference; varying background load prevents treating this
as a controlled speed benchmark.

## What ran

Local Ollama Llama 3.1 8B, pinned digest in the JSON report. Two planning attempts
per agent per episode, 20 actionable turns between plans, global request budget
36. Thirty-five requests: 32 accepted and three invalid-focus plans rejected.
The executor kept attacking after rejection. Planning latency median 3.563 s,
maximum 14.203 s. Usage: 46,901 prompt tokens and 595 output tokens.

All model plans selected spread, with soft target focus. Therefore this run
supports the combined planner/executor against the fixed-spread/no-focus control;
it does not demonstrate effective switching between spread/escalate/disrupt.
Weights were not trained or fine-tuned. Shared CAGE RNG means pairing seeds
does not hold every subsequent Green event identical across attacker policies.

## Reproduce

From `C:/Projects/cyberally/runs/red-blue-test`, with the local model service running:

```powershell
C:/Projects/cyberally/.venv-train/Scripts/python.exe -m red.main --compare --seeds 7629 7630 7631 --steps 400 --plans-per-agent 2 --plan-interval 20 --max-requests 36 --output runs/lancer-repeat
```

The output directory must be new. Model seeds/temperature do not guarantee
identical inference across runtimes/hardware. Use `--mock` only for wiring tests.

## Validation and limitations

41 Red tests passed after the fix. The earlier combined regression command
passed 73 tests: 37 then-current Red tests and 36 Blue foundation, named-Red
and Lancer tests. Four additional Red tests are included in the final 41 total.
Python compilation, whitespace checks and saved source/trace hashes passed.

The first full live attempt exposed CAGE PIDSelective selector compatibility
bugs and stopped at tick 118. That incomplete trace is excluded. Red now uses
a local selector accepting native priority and excluding Red-observed decoy
ports; simulator source is unchanged, and a regression test covers it.

Three reused development seeds are a pilot, not an untouched final benchmark.
Other Blue policies, broader seed suites, repeated model runs and actual Blue
retraining have not been tested. Next: review the additive wrapper injection,
select `CC4LLMEnv` in the intended Blue runner, then run broader evaluation
before collecting fresh rollouts for retraining.
