# Implementation roadmap and branch assignments

## Current position

The local source contains CybORG, preliminary Blue features/masks/wrappers and
research experiments. Static review found correctness blockers. There is no
reproduced training run, unified evaluator, Red adapter or closed-loop pipeline.
The review slides describe the long-term system; they are not acceptance evidence.

**Read `docs/current-state.md` for measured behaviour.** Two facts change the
order of work here:

- BLUE-01 wrapper fixes are **merged**: `blue/foundation` was fast-forwarded into
  `main`, and both refs are at `f8a9deb`. Branch from `main` normally; earlier
  docs warning against `main` are obsolete.
- No pinned runtime exists. The local venv is Python 3.12 with no torch;
  Python 3.10 is the agreed target. Every live test and all training depend on
  Environment closing this. **This is now the top shared blocker.**

Blue host capacity is **per agent** — 17 for agents 0-3, 51 for agent 4 — so
no shared shape constant should be written into any config before
`docs/coordination/blue-action-space.md` is resolved. Derive shapes from
`get_env_info()`.

Keep the full existing CAGE4 scenario for the first integration. A reduced topology
and real Docker/eBPF actions are separately versioned Environment deliverables.
Plan by exit criteria, not by an assumed number of weeks; training runtime is unknown.

## M0: reproducible foundation (first parallel tasks)

| ID / branch | Work and expected changes | Acceptance evidence / dependencies |
|---|---|---|
| ENV-01 `environment/reproducible-runtime` | Establish Python version, minimal CybORG dependencies, isolated setup instructions and pinned dependency files under `environment/`. Assess Windows vs WSL/Linux. Record upstream provenance. | Fresh environment imports CybORG and executes reset/joint step. Document actual install commands and tests. Do not install all optional frameworks by default. |
| BLUE-01 `blue/foundation` | Repair reset/seed semantics, host capacity, agent-wide busy state, passive-event ingestion, remediation state and masks in the existing wrappers/tracker. Add focused regression tests. | One reset per episode; reproducible seeds; no dropped hosts; pending target stays stable; Restore reachable; user/root remediation verified in live simulation. Requires ENV-01 runtime for live tests. |
| RED-01 `red/strategy-adapter` | Wrap the built-in scripted Red behind a scenario factory, stable config and provenance record. Keep existing Red behavior as a baseline. | Same seed/config is reproducible under an unchanged build; factory produces valid simulator actions without exposing hidden state to Blue. Coordinate injection seam with Environment/Blue. |
| EVAL-01 `eval/baseline-harness` | Define metric semantics and JSONL records, implement seed-suite runner and privileged-label evaluator in `loop_eval/`. | Sleep rollout produces validated logs and expected episode length; labels isolated from policy inputs. Can draft against fixtures before BLUE-01 merges. |

Shared first decisions: Python/runtime; joint-step/reset ownership; episode horizon;
native reward aggregation; test/train seed split; action/observation versions.
Record acceptance in `docs/contracts.md` and role statuses. No long training yet.

## M1: comparable baselines

- BLUE-02: implement round-robin investigation/remediation heuristic; add Sleep,
  built-in random and masked-random adapters. Keep the historical reduced action
  set for a clearly labeled comparison, not a silent change to all baselines.
- EVAL-02: run policies on paired seed suites; save small aggregate metrics and
  an external artifact manifest. Recompute seed-7629 step-200 numbers after fixes.
- ENV-02: validate topology/agent coverage including the three-subnet HQ actor;
  ensure passive telemetry and mission transitions are observable as intended.
- RED-02: supply named attacker configurations and a held-out suite. Evaluate
  Multi-Adversary-CAGE4 compatibility in an isolated branch, pin source revision,
  preserve attribution; do not overwrite the local simulator with a whole fork.

Exit: one command/config runs the same evaluator against every baseline, with
unambiguous reward, host compromise, service disruption and episode-length metrics.

## M2: working MARL training

- BLUE-03 `blue/mappo-training`: pin EPyMARL revision; implement its actual runner
  contract including tensor actions, constructor options and common reward; add
  config, checkpoint save/load and evaluator adapter. Derive shapes from env info.
- ENV-03: provide a separately pinned training dependency profile if needed and
  measure simulator throughput before choosing rollout workers.
- EVAL-03: record training curves, config/source hashes, held-out validation and
  inference latency independently of simulation time.

Exit: complete rollout + optimizer update + saved/reloaded policy produce valid
actions; training can resume reproducibly enough to audit; compare against M1.
No requirement to claim RL wins if it does not.

## M3: research improvements, one controlled change at a time

1. BLUE-04 `blue/temporal-policy`: recurrent actor and explicit observation age,
   missingness, verification state and mission context. Compare recurrence alone
   with recurrence + explicit temporal features.
2. BLUE-05 `blue/anomaly-input`: fit anomaly model to representative runtime
   telemetry, with episode-level train/validation separation. Compare no score,
   file-rule score and IF/LOF score. Historical IF performance was weak.
3. BLUE-06 `blue/hierarchical-baseline`: reproduce released Hierarchical-MARL PPO
   policies in a separate dependency environment if required. This is a baseline
   independent-PPO implementation, not automatically MAPPO.
4. BLUE-07 `blue/attacker-belief`: estimate attacker strategy from local observable
   history, condition policy on uncertain estimates, and train with realistic
   estimator errors. Compare ordinary recurrent MAPPO, belief conditioning, and
   belief conditioning + freshness under switching attackers.
5. Optional BLUE-08: shared host/entity encoder for variable topology. Try only
   after baseline host-padding/coverage is correct; it must not conceal missing hosts.

Exit: individual ablation results and held-out scenarios identify useful additions.
The 2026 posterior-conditioned CAGE4 paper is close prior work, so attacker belief
alone is not our novelty. See `docs/references.md` and the detailed Blue plan.

## M4: self-evolving loop

- RED-03: emit replayable strategy configurations/factories with seed, version,
  prerequisites and outcome. A successful action trace alone may not replay after
  defenses or topology change. LLM Red integration is optional after scripted variants.
- EVAL-04 `eval/retraining-loop`: orchestrate pool version -> fresh rollouts ->
  Blue update -> fixed-suite evaluation -> candidate checkpoint manifest. Compare
  frozen policy, new-attacker-only tuning and mixed old/new scenario tuning.
- BLUE-09: expose train/load/evaluate hooks and checkpoint compatibility validation.
  Add EWC only if mixed-scenario training still shows meaningful forgetting.
- ENV-04: ensure comparable resource limits and scenario isolation across rounds.

Exit: at least three recorded adaptation rounds with new-attack improvement and
old-attack retention, plus a held-out final suite. Keep a fixed benchmark suite so
changing training attackers does not masquerade as improvement.

## M5: integration, review and simulator deployment

- All roles deliver runnable adapters/configs, manifests and fresh-environment
  instructions. The evaluator loads policy artifacts by hash/version and fails
  clearly on incompatible schema or action mappings.
- Dashboard/auditor consumers use structured observation/action/outcome records.
  An LLM explanation is not itself evidence of a correct remediation.
- Run multiple independent training seeds and paired held-out evaluation suites;
  target >=3 training seeds and >=30 evaluation seeds when resources permit.
- Update slides: use simulated steps for MTTD/MTTR, measured inference p50/p95 for
  wall-clock policy latency, and explicit baselines for false-positive claims.
- Real eBPF/container execution, SIEM deployment and zero-day resilience require
  separate experiments. Do not silently treat CybORG success as those deliverables.

## Cross-branch changes required

| Producer changes | Consumers must change | Rollout/test |
|---|---|---|
| Environment seed/reset or observation source | Blue wrapper, Red factory, Eval runner | Contract proposal; deterministic same-world integration test |
| Blue observation/action dimensions | Training configs, checkpoint loader, evaluation adapter | Bump feature/action versions; reject old incompatible checkpoints |
| Blue masks or pending semantics | Heuristic, rollout collector, action trace logger | Busy-agent test and intended-vs-executed action validation |
| Red strategy factory/config | Environment scenario builder, Blue training sampler, pool manager | Versioned fixture + scripted rollout before LLM integration |
| Reward or episode-boundary semantics | Critic targets, score aggregation, all baselines | New benchmark config/version; rerun affected baselines |
| Eval event schema or metric definition | All log producers and dashboard/auditor consumers | Additive fields first; fixture validation; explicit migration |
| Runtime/dependency versions | Every role and CI | Clean install smoke test; pin revisions before merge |

## PR-sized work and reporting

Keep correctness fixes separate from architectural research. Each PR states one
concrete behavior change, dependent issue/PR, test evidence, contract changes and
next consumer action. A documentation-only PR does not need simulation training.
Never label a stage done solely because code exists; use the stated exit evidence.
