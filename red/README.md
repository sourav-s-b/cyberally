# Main attacker: LLM Red

Run from this worktree root. `red.main` defaults to local Llama 3.1 8B Red
against the GitHub Blue `hybrid_lancer_v2` policy. This is the local Lancer-style
priority sweep with evidence-gated remediation, without decoys; it is not a
reproduction of the published competition Lancer.

```powershell
C:/Projects/cyberally/.venv-train/Scripts/python.exe -m red.main --output runs/my-red-test
```

Each output directory must be new. This command runs full native 400-step
episodes on development seeds 7629, 7630 and 7631. It requires the installed
local Ollama model. Start it with `./red/start_ollama.ps1` if needed. Models,
runtime tools and logs remain ignored and outside Git.

To compare with the CAGE DiscoveryFSRed baseline and an executor-only control:

```powershell
C:/Projects/cyberally/.venv-train/Scripts/python.exe -m red.main --compare --output runs/my-paired-test
C:/Projects/cyberally/.venv-train/Scripts/python.exe -m pytest -q red/tests
```

`--mock` is a deterministic wiring fixture, not LLM inference or performance
evidence. `--blue sleep` selects the undefended reference. `--plans-per-agent`,
`--plan-interval` and `--max-requests` bound inference. Default: four planning
attempts per Red agent per episode, every 12 actionable turns, 72 requests
across the entire invocation. Pending actions do not call the model.

The LLM chooses spread, escalation or disruption and optionally a known target.
The executor maintains persistent Red-local host states, responds to action
results and session loss, checks enabled parameters, handles discovered decoys,
and constructs native CAGE actions. Each rejected/failed plan retains the last
validated strategy and continues execution. It never instantiates DiscoveryFSRed
as a fallback. With no legal action it Sleeps. Episode reset clears all local
knowledge, plans and history. The global request budget survives resets.

The generic upstream finite-state knowledge-transition engine is reused under
the existing CAGE attribution. The existing DiscoveryFSRed attacker is used
only when explicitly requesting `--compare`. Do not delete upstream framework
classes: the simulator and comparison tests depend on them.

## Blue training entry point

`red.blue_env.CC4LLMEnv` has the same Blue observation/action/step interface as
`blue.core.wrapper.CC4MARLEnv`, with LLM Red selected by default. It accepts
`llm_config` and `planner_config` dictionaries; the existing Blue wrapper's
attacker default is preserved for historical benchmark reproduction.

```python
from red.blue_env import CC4LLMEnv
env = CC4LLMEnv(seed=7629, steps=400,
    llm_config={"model": "llama3.1:8b", "timeout_seconds": 60, "max_requests": 72})
env.reset(seed=7629)
```

This is a tested environment entry point, not a completed Blue retraining run.
Existing Blue training registries/configs must explicitly select this class.
Set the inference budget to match the rollout budget; budget exhaustion keeps
tactical execution but means later rollouts are no longer receiving new plans.

## Evidence and limits

See [the completed development comparison](evidence/lancer-development-20261007.md)
and its adjacent JSON report for actual live results.

`report.json` records actual seeds, source hashes/commit, model digest, prompts,
configuration, inference usage, paired episode results and metric definitions.
`decisions.jsonl` separates Red selections from controller-executed actions.
Lower native Blue return favors Red. Compromised/privileged host-time counts
unique hosts with active Red sessions after each tick, including initial footholds.
Privileged labels are evaluated separately and never enter either policy.

The CAGE RNG is shared: pairing scenario seeds does not guarantee identical
Green events after policies consume different random draws. Three development
seeds support a pilot comparison, not a universal superiority claim. We have
not trained/fine-tuned Llama, changed the simulator, or run long Blue training.
