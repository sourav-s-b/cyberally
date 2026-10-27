# Phase 1 Final Review: Detailed Project Analysis and Deck Plan

**Prepared:** 27 October 2026  
**Project:** Self-Evolving AI Cyber-Range for SOC Automation  
**Source deck:** runs/first_review/First_Review.original.pptx (12 slides)  
**Checkout reviewed:** phase-one-final-review, HEAD 5429349, with existing local changes preserved.  
**Scope:** project implementation, experiments, and documentation. Dashboard details are excluded. Zeroth Review vision and claims are assessed below.

## 1. Executive assessment

Cyberally is now a simulator-backed cyber-defense research project with materially more implementation and evidence than the original first-review deck communicates. The repository contains an operational CAGE4 Blue wrapper, Blue-visible observation and action handling, scripted baselines, several imitation/offline/RL policy families, evaluation and training utilities, a scripted Red strategy adapter, an optional LLM Red planner/executor, regression tests, and a large set of frozen experiment records. The most valuable Phase 1 output is the combination of a corrected experimental substrate and a body of results that identifies where current learning approaches fail or remain uncertain.

The first review presents the project as a proposal with a MAPPO pilot “started” and a containerized, decentralized, self-evolving defense system as the target. A final review should instead show the working simulator pipeline, exact model comparisons, limitations exposed by evaluation, and the next research gates. Implementation volume must not be presented as demonstrated defensive improvement.

**Evidence-based conclusion:** the CAGE4 research prototype and experiment tooling are substantial. The project has not established a robust learned-policy advantage over strong heuristics on fresh held-out conditions, has not completed an end-to-end retraining-and-retention loop, and has not implemented real-world host telemetry or containment. The strongest controlled Discovery Red comparison did not meet its predeclared success criterion. A separate planner-driven LLM Red comparison is suggestive but has a provenance caveat and uses one trained checkpoint. Architecture screens give useful leads but have only two independent training RNGs and must be described as exploratory.

The repository does not define an approved Phase 1 rubric, weighted completion measure, or confirmed final-review date. Use a deliverable/evidence matrix with complete, partial, and open labels rather than an unsupported completion percentage.

## 2. Review method and source precedence

This analysis uses the preserved 12-slide original deck, current project code, experiment records and status summaries, AGENTS.md, docs/current-state.md, team workflow, contracts, implementation plan, Blue plan and handoff corrections, and role status files. Where dated status pages disagree with the current checkout, the 27 October checkout correction in docs/current-state.md takes precedence. Older status paragraphs often describe another branch or earlier source paths; they are useful as dated experiment records, not as present-tense descriptions of this checkout.

The configured origin remote could not be fetched because .git/FETCH_HEAD is read-only in this environment. This report describes local evidence and does not claim that the remote has no newer work. Existing project changes in the working tree were left untouched.

## 3. Project architecture and code-level findings

### 3.1 Environment and Blue wrapper

The local simulator is the supplied CAGE Challenge 4 / CybORG Enterprise environment. The CAGE4 simulator tree is treated as upstream and remains unmodified by this project. Team behavior is added in Blue and Red adapters around it. The exact upstream source commit of the copied simulator has not been established; do not call this checkout a current upstream release.

The active wrapper is blue/core/wrapper.py, class CC4MARLEnv, not the old root-level paths named by historical notes. It runs one joint simulator and retains the five native Blue identities. It resets the shared environment once, applies seeds, gathers all five initial observations from the same world, builds stable host lists, and errors on a host-bound overflow rather than silently truncating hosts. It exposes dimensions through get_env_info(); dimensions must be derived from the selected mode rather than copied from an old handoff.

Current full-topology per-agent host bounds are 17 / 17 / 17 / 17 / 51. Agents 0–3 each own one subnet; agent 4 owns three subnets and may observe up to 51 hosts. At a 51-host global bound, a shared action vector has 155 slots per agent. Per-agent mode instead has widths 53 / 53 / 53 / 53 / 155. Per-agent bounds reduce padding, but the stock EPyMARL shared-head setting requires a common shape, so this is a consumer/configuration choice.

Observation construction in blue/core/obs_features.py converts Blue-visible host records into numeric features. The base representation has ten channels per host, including visible session/process/file/connection summaries and a host/session signal. Optional event-time features append age, evidence, action, or mission-context values; feature flags change dimensions and are recorded in environment metadata. Padding/presence is distinct from a real zero-valued host. The wrapper retains observation timestamps and merges available passive observations across ticks.

blue/core/masking.py separates simulator validity from evidence-based policy preference. The default validity mask permits every simulator-valid action; evidence-mode restrictions are a policy choice and can be ablated. Actions are global Sleep/Monitor plus host-targeted Analyse/Remove/Restore. Each agent has one pending-action lock: while an action is in flight, only Sleep is available to that agent, and the eventual result is attributed to its original host/action. Analyse can establish visible evidence; remediation moves the belief to VERIFY. A successful Remove is not labeled clean; verification requires later observations. Two empty analyses clear the current belief to UNKNOWN, not to privileged ground-truth CLEAN.

These details fix high-impact correctness defects documented in early audits: resetting the shared world repeatedly for each Blue agent; not applying requested seeds; losing agent-4 hosts to a 16-slot truncation; overwriting pending actions; treating removal success as proof of cleanup; and conflating action validity with evidence policy. This is a core Phase 1 deliverable because policy comparisons only mean something on a consistent environment.

### 3.2 Baselines and policy control

blue/core/baselines.py implements Sleep, masked random, unmasked-random compatibility, round-robin, stalest-first, and suspicion-sweep policies. Round-robin is a strong control: it remediates visible confirmations, verifies remediation, escalates to Restore when a post-remediation Analyse re-detects evidence, and otherwise investigates hosts. It uses Blue-visible tracker information rather than privileged Red labels. Beating Sleep or random alone would not establish a meaningful result; the strong heuristic is the useful performance bar.

The repository includes flat and factorized actors, SetActor and candidate-attention actors, risk/novelty residual actors, behavior cloning, RvS, IQL/offline AWR, PPO/MAPPO, A2C, guided/anchored variants, a strict Lancer framework, fixed blends, state gates, and value-aware-routing proposals. There are many scripts because the project has investigated multiple failure modes. **Code presence, a checkpoint, or a successful short loader episode is not evidence that a policy outperforms another policy.** Only a matched, documented comparison supports a performance claim.

### 3.3 Red behavior

Red-owned code includes red/strategy_adapter.py, a native scripted strategy adapter and config, and LLM components in red/llm_agent.py, red/planned_agent.py, and red/blue_env.py. The scripted adapter preserves native behavior/configuration and has parity/fresh-reset tests. The LLM path uses an observation-only planner with validation and a persistent tactical executor. The intent is not to leak privileged simulator truth into Blue. LLM inference depends on local Ollama/model/runtime and is not bit-for-bit repeatable across versions.

There are multiple attacker regimes in the evidence: built-in DiscoveryFSRed/Discovery, FiniteStateRed/Finite, and planner-driven LLM Red. They have different behavior, difficulty, and budgets. Returns from these regimes must not be pooled or ranked as one benchmark. Red strength is scenario-dependent; one attacker winning on a small development block does not prove it is universally strongest.

### 3.4 Evaluation, data boundaries, and integration

The project has substantial Blue-specific evaluation and analysis code: metrics and recorders, training/evaluation harnesses, seed audits, experiment manifests, trace and replay checks, and aggregate reports. Privileged true-state labels are used for evaluation and some explicitly internal reward-shaping experiments, not actor observations, masks, or deployed risk inputs. Project contracts insist on separate records for requested action, action execution/completion, Blue-observed verification, and privileged evaluation truth.

Shared team contracts are explicitly proposed rather than accepted runtime APIs. The environment/ and loop_eval/ areas are ownership/scope areas without a completed unified team runtime/evaluator integration in this checkout. Red implementation exists, but all producers and consumers are not yet wired through a reviewed scenario factory, shared schema, strategy pool, and retraining loop. Experiment-specific training/evaluation harnesses do exist under blue/training; that is distinct from the full project-level adaptive loop.

The research plan calls for fresh on-policy rollouts for PPO/MAPPO retraining, separate training and held-out scenario seeds, source/config/runtime hashes, artifact manifests, and retention measurement on old attack suites. Show these as Phase 2 gates, not achieved results.

## 4. What the experiments establish

### 4.1 Environment facts and problem motivation

The project’s early measured CAGE4 run (seed 7629, masked random) reported roughly 44 ms per joint step, 0.19 s for reset, and approximately 18 s for a 400-tick episode on that measured setup. Rewards were sparse: 162/399 ticks carried nonzero reward, with the first nonzero reward around tick 135. A follow-up found forced Sleep was pending-action lockout, not session loss. These observations explain why coverage, evidence age, action duration, and attribution matter. They are historical hardware/scenario measurements, not general performance guarantees.

Blue has partial observability. Privileged state can show Red sessions when Blue observations do not; host compromise and Blue detection are separate quantities. Some old notes count different host sets, so those counts must not be turned into a detection-rate ratio without matched denominators. Describe the task as evidence-based investigation under incomplete information, not direct compromise sensing.

### 4.2 Controlled main32 Discovery Red comparison

The best controlled cross-arm comparison in the reviewed reports is docs/status/blue-gpt-main32-results.md. Five independent training RNG seeds each trained all three input arms (zero, risk, both) for 32 iterations under the same frozen runtime. Every arm was evaluated on the same 32 **reused development** Discovery Red environments, seeds 8221–8252, at a 400-tick horizon. The statistical unit for the paired intervals is training RNG, n=5, df=4; the 32 environments are shared blocks, not 160 independent trained models. Higher native return is better.

| Policy | Mean native return | Difference vs Lancer | Difference vs coverage guard |
|---|---:|---:|---:|
| Lancer heuristic | -79.91 | — | — |
| Coverage guard | -78.22 | — | — |
| PPO/MAPPO zero-input arm | -74.59 | +5.31 | +3.62 |
| PPO/MAPPO risk-input arm | -81.21 | -1.31 | -2.99 |
| PPO/MAPPO risk + novelty arm (predeclared primary) | -74.71 | +5.19 | +3.51 |
| Masked random | -245.00 | — | — |
| Sleep | -2267.28 | — | — |

The primary both-minus-guard mean was +3.50625 with paired training-seed t95 interval **[-3.14891, +10.16141]**. Both-minus-Lancer was +5.19375 with interval **[-1.46141, +11.84891]**. The preregistered rule required positive lower endpoints versus both controls, and it was not met. Risk-minus-zero was -6.61875 with interval **[-17.57265, +4.33515]**. Combined return was effectively tied with zero (-0.11875). The conclusion is **no learned superiority demonstrated**, not “zero is the winner” and not “risk has no effect.”

The suite was reused for development, not fresh held-out confirmation. Seeds 7809–8200 were protected from this experiment. Manifests, hashes, matched cells, and five completed jobs were validated per the report. Use this as the lead model-comparison slide because it has a frozen protocol, strong controls, a declared primary arm, and a clear non-success result.

### 4.3 Detector evaluation versus policy outcome

The expanded detector audit compared one fixed HGB risk model with logistic regression on eight previously consumed development episodes. Those episodes were disjoint from detector training/calibration, but were not a fresh final evaluation. Macro PR-AUC was **0.72794** for HGB and **0.63107** for logistic regression; paired episode difference **+0.09687**, conditional episode interval **[+0.06861, +0.12514]**. This supports a ranking comparison for that dataset and model fit. It does not establish model-seed robustness, generalization to a new attacker, or a defensive reward benefit.

The policy experiment answers a separate question. In main32, risk-only return was below the zero-input mean, with an uncertainty interval crossing zero. Novelty alone was weak (pooled PR-AUC 0.2514 vs prevalence 0.2412) and should not be described as compromise probability. Suggested wording: “Risk ranking improved on reused development episodes; downstream defense benefit was not demonstrated.”

### 4.4 Planner-driven LLM Red comparison

The first three-seed Blue-vs-LLM-Red run on seeds 7629–7631 compared Lancer, an unguarded learned actor, and the same actor with a coverage guard at evaluation. Means were -670.7, -1333.7, and -557.7. The unguarded actor was much worse than Lancer; the guard materially improved the same weights, but guarded-vs-Lancer interval crossed zero. These are development seeds and one trained checkpoint, so they cannot establish general learned advantage.

A later 24-seed paired block on 7641–7664 reports Lancer mean -637.3 and guarded actor -516.7, difference +120.6, 95% paired interval **[+1.4, +239.8]**, 16/24 wins. The underlying evaluation has a provenance caveat: rows 0–42 used an uncommitted working-tree version of the evaluator that is no longer available; rows 43–47 used committed code. It is also one checkpoint and one training seed. Call this **suggestive/source-caveated** and request a clean rerun before a headline claim. This study is specific to pinned llama3.1:8b LLM Red and separate from main32 Discovery Red.

The Red-side LLM comparison against a fixed local Blue reports mean Blue returns around -680 for LLM Red, -55 for DiscoveryFSRed, and -113.33 for the tactical-only ablation on three development seeds. Lower return favors Red in that report. LLM Red caused more mission damage on those seeds, but only three development seeds and one fixed Blue policy were used. The 8 October deck audit could not recover source outputs for the original four-model Red table. Keep those four cells source-pending until outputs are recovered and matched.

### 4.5 Architecture and learning-method research

The Blue program summary and chronological session record document both positive and negative results. Use the table as a compact research-progress slide and retain per-run details in an appendix.

| Research path | Documented result | Evidence strength and wording |
|---|---|---|
| Round-robin teacher | Earlier native means -93.5 ± 27.8 regression and -85.1 ± 46.1 held-out in an eight-seed suite | Strong heuristic reference within that protocol; not numerically comparable with main32 LLM/Discovery runs. |
| Fixed hybrid_lancer_v2 ordering | -75.2 ± 33.6 regression; -91.2 ± 48.5 held-out | Best fixed variant in that suite, but no held-out transfer edge over teacher. |
| Attention behavior cloning | -85.5 ± 31.1 regression; -93.9 ± 45.8 held-out | Teacher-like performance without online RL in that suite; does not establish online RL benefit. |
| Flat behavior cloning | Regression mean -393.4 ± 276.9 | Unstable/poor starting point in that pipeline. |
| PPO/MAPPO fine-tuning, earlier distillation line | Several runs diverged and returns degraded substantially | Negative result; do not conceal. Values belong to that historical protocol. |
| Risk and risk-recency ordering variants | Several variants hurt return or collapsed coverage; later main32 risk-only arm did not beat zero | General risk-priority hypothesis not supported. Different datasets/protocols; do not combine effect sizes. |
| Age/coverage guards | Metric repair found unguarded Lancer already had full coverage in 32/32 seeds; tighter max-age guards lowered age but did not show a return benefit | Better freshness did not automatically improve defense return. No detectable benefit at that resolution, not proof of no effect. |
| SetActor vs candidate-attention pilot | Two training RNGs, two evaluation seeds per Red type; small/mixed learned-minus-Lancer differences | Too small to rank architectures. Pilot only. |
| Longer learned state-gate hybrid | Two RNGs; Discovery and Finite; 16 paired evaluation seeds per regime/RNG in actual run. Positive mean differences but tail-sensitive and seed-dependent. Gate means ~0.60 vs ~0.08 across RNGs; no repeatable regime routing | Promising screen, not confirmation. Actual evaluation used 16 seeds per regime, twice proposal count. |
| Gate utility ablation | Learned gate -11.7 mean return vs fixed blend; fixed blend +9.8 vs untrained control and +12.2 vs strict Lancer by macro mean | Two RNGs; descriptive. Medians smaller; tail-sensitivity caveat. |
| Qwen event-time credit screen | Five GAE runs averaged -162.8 training return/episode vs five Monte Carlo runs -165.6; reported training coverage 1.0 | Archive had no held-out evaluation; cannot claim generalization or confirmed GAE advantage. |
| Persistent-reference KL screen | Frozen protocol prepared; first CPU attempt stopped before update; GPU v2 was active in the last status entry | No completed result in reviewed status. Exclude as result; include only as ongoing if reconfirmed at presentation cutoff. |

These outcomes support a balanced story: the team tested hypotheses, found useful controls, and recorded failed or inconclusive approaches. Keep the method, attacker, seed, and confidence context attached to every score. Do not combine the rows into a cross-study “best model” ranking.

### 4.6 Checkpoint availability versus comparative evidence

The 27 October session log lists 137 local agent.th checkpoints and several manifest-backed entries, with seven highlighted candidates. That inventory is useful for reproducibility but is not itself a result. Pre-existing checkpoint examples include the main32 risk+novelty actor, attention BC, RvS, and IQL alpha 0.5. Three gate-ablation actors were newly retrained locally from a six-iteration schedule with RNG 0 and train seeds 8465–8472; their evaluation seeds were recorded but evaluation was not run. The reference_s0 SetActor is a 400-tick historical pilot with +0.25 mean over four development episodes; it is not the later fixed 0.5 SetActor/attention blend. Do not attach the earlier architecture screen effect sizes to those newly retrained artifacts.

Only policies with comparable results belong in the primary comparison table. Put the broader checkpoint inventory in an appendix and label untested entries “available; not yet compared.”

## 5. Zeroth Review: claims and feasibility against the project

The Zeroth Review has 17 slides. Its central idea is an air-gapped, containerized, multi-zone cyber range with an LLM Red team that chains and mutates exploits, an eBPF/packet telemetry pipeline, Isolation Forest/LOF features, decentralized MAPPO containment, an LLM reasoning auditor, enterprise SIEM output, and a closed-loop policy-retraining system. The architecture and workflow images also explicitly name Juice Shop/DVWA, SQL injection/RCE, lateral subnet pivoting, bandwidth-limited links, eBPF packet drops, container quarantine, honey-net diversion, token/session revocation, EWC, MITRE ATT&CK mapping, and sub-100 ms / sub-5 second behavior.

The current project implements an in-process CAGE4 simulator and adapters. The Red planner selects among validated native CAGE actions and goals; the Blue code reasons from simulator observations and has simulator-native host actions. That is a substantial but different implementation boundary. The matrix checks whether each Zeroth claim exists now, whether it can be tested within CAGE4, or whether it requires a new environment/integration. “Feasible” means a plausible engineering/research direction, not that a promised target is likely or validated.

| Zeroth claim (slide) | Evidence in current project files | Can we do it? | Correct final-review wording / work needed |
|---|---|---|---|
| Air-gapped, multi-zone Docker cyber range with strict egress (2, 4, 8–12) | Team simulator code wraps the existing CAGE4 Enterprise simulator. No team Docker Compose/Dockerfile, egress policy, or deployment validation was found outside the upstream simulator tests. CAGE4 already supplies a multi-zone topology. | **Possible as a separate environment stage; not present or security-validated now.** Air-gapping is an operational property that needs configuration and testing. | “Phase 1 runs the CAGE4 simulator locally. A containerized range with egress controls is a proposed extension.” Build a pinned image, document network policy, test denied egress and reset isolation, and review the threat boundary. Do not imply CAGE4 itself proves air-gap security. |
| OWASP Juice Shop/DVWA targets; SQLi/RCE exploit chains (8, 10, 11) | No Juice Shop/DVWA target integration or real app exploit adapter was found in team-owned code. CAGE4 has its own simulated attack actions and native transition rules. | **Not with current CAGE4 setup.** Could be a separate safe cyber-range integration. | Keep attack validation inside named CAGE4 Red variants for Phase 1. To add web apps, deploy isolated vulnerable targets and map observable state/actions to a new environment contract; validate no external targets or network access. |
| LLM Red dynamically probes, chains exploits, mutates payloads, pivots laterally and simulates continuous APT (2, 7–11) | red/planned_agent.py uses an LLM to propose a bounded goal and optional focus; the proposal is validated against current native action space, and a persistent CAGE tactical executor chooses valid actions. No arbitrary payload/code generation or payload mutation was found. Existing Red comparison used a capped number of plans/requests. | **Strategic planning within CAGE4: implemented in part. Real exploit chaining/payload mutation: absent.** Strategy variation is testable in simulation; “zero-day” is not established. | “An optional local LLM planner changes goals/focus within native CAGE4 actions.” To extend, add replayable Red strategy/config variants and evaluate on fresh held-out variants. Call this attack-strategy generalization, not zero-day discovery. |
| eBPF kernel hooks, packet streams and Packetbeat telemetry (4, 8, 10, 11, 12) | blue/core/telemetry.py and collect_telemetry.py provide simulator-derived telemetry records. No eBPF, packet capture, Packetbeat, or OSQuery producer was found in team code. | **Possible only as a separate real telemetry producer.** Current features are generated from simulator observations. | “Blue features are derived from CAGE4 observations.” A real producer needs collection agents, field provenance, privacy/security review, event-time normalization, missingness handling, and replayable fixtures. Do not call simulator telemetry eBPF or packet telemetry. |
| Isolation Forest/LOF anomaly scoring suppresses benign noise (5, 8, 10) | Blue training harnesses contain Isolation Forest and HGB detector code. The reviewed detector audit found HGB macro PR-AUC 0.72794 vs logistic 0.63107 on eight reused development episodes; novelty's pooled PR-AUC was 0.2514 vs prevalence 0.2412. No active LOF use was located in the inspected team code. Historical detector notes include unreproduced results. | **Simulator-scoped detector experiments: implemented. Real-time noise suppression/LOF: not demonstrated.** | Present PR-AUC as a reused-development detector result, separate from policy return. Define threshold and false-positive denominator; repeat on fresh scenarios/attack regimes. Real telemetry performance requires a real data set and producer. |
| Decentralized per-zone MAPPO with CTDE and local actors (2, 7–11) | The wrapper exposes five local Blue observations and a joint action step. Specific PPO/MAPPO paths use a centralized value critic with actors selecting from local features. Agent/subnet ownership is CAGE4-native; agent 4 owns three subnets. | **A multi-agent simulator policy is implemented; final performance target is not achieved.** The slide's exact “DMZ/App/DB per-zone actor” abstraction does not match native zone/agent ownership. | Say “five-agent CAGE4 Blue policy experiments, including centralized-critic/local-observation methods.” Preserve native zone/host names and agent assignments. Main32 did not prove learned superiority over controls. |
| Bandwidth-limited agent links / cooperative coordination (4, 7, 8, 11) | No Blue peer-message channel or bandwidth budget was found in active team code. Independent local observations and common reward do not constitute communication. | **Can be added as a separately scoped simulation feature, not current behavior.** | Remove “bandwidth-limited” and “cooperative messaging” claims until a message schema, delay/drop/budget model, and message/no-message ablation are implemented. CTDE alone is not runtime coordination. |
| Sub-100 ms policy action (2, 7–11) | No controlled policy-inference p50/p95 benchmark on named hardware was found. Early 44 ms per joint simulator step is a dated simulator-throughput figure, not neural inference latency. Red LLM planning latency was seconds (median 3.563, maximum 14.203 in one pilot), not defender latency. | **May be attainable for a small local actor; unmeasured.** Full action/containment latency includes other components and cannot be inferred from simulator speed. | Treat as a target. Benchmark actor-only p50/p95 and end-to-end simulator dispatch/completion separately, with model, batch size, device, warm/cold state, and load. Do not call a simulator tick sub-100 ms policy inference. |
| eBPF packet drop, container freeze/quarantine, honey-net routing, token/session revoke (4, 8, 10, 11) | The current Blue wrapper maps host actions to CAGE4 Analyse, Remove and Restore, plus global Sleep/Monitor. Zone-level traffic-blocking actions are not enabled. No real host actuator exists in team code. | **CAGE4 response comparison is possible within existing actions; these specific controls require new simulator semantics or a real adapter.** | State current action boundary explicitly. Add simulator interventions only through affected-role review and measure service disruption. Real containment needs a separately secured actuator and failure/safety tests. |
| MTTD/MTTR, total containment under five seconds (4, 9, 11–13) | Blue analysis code computes simulator-step investigation/detection delays and censored cases. There is no validated wall-clock mapping from a CAGE4 step to real response time; verified cleanup semantics need to remain distinct from action completion. | **Step-based detection/response metrics: feasible and partly implemented. Five real seconds: not established.** | Report ticks/steps for CAGE4. Define incident start, first Blue detection, verified clearance, reinfection, and censoring. Seconds require a real target and an explicit wall-clock protocol. |
| >90% false-positive reduction / benign-noise suppression (4, 8, 9, 13) | Detector classification experiments use simulated labels and reused development episodes. There is no operational SOC alert stream or agreed false-positive denominator in the current project. | **Can be tested in simulator with an explicit definition; target value unsupported.** | Define FP per host-step, alert, or incident; fix threshold on calibration data and evaluate on held-out episodes. Do not claim 90% or imply reduced analyst workload without a real workload study. |
| LLM reasoning auditor, validated MITRE ATT&CK trail, enterprise SIEM (4, 7–12) | The project has experiment/action traces and Red planner metadata. No independent LLM audit evaluator or full team-owned MITRE mapping was located in the inspected code. JSON/JSONL artifact formats exist; no connected enterprise SIEM integration is evidenced. | **A structured simulator audit trail is feasible; validated reasoning or enterprise integration remains separate work.** | Start with deterministic action/evidence/result records and explicit source mappings. Add ATT&CK tags only when each mapping is reviewed and tested. An LLM explanation is not proof of a correct decision. SIEM integration needs schema compatibility and end-to-end ingestion validation. |
| Closed-loop Red pool → MAPPO retrain → hardened redeployment (7–13) | Project has many training/evaluation scripts, scenario-mixture proposals, saved checkpoints, and isolated retraining experiments. It does not show a completed loop that discovers new attacks, collects fresh on-policy rollouts, verifies gain, checks old-suite retention, and promotes a version. | **A simulator-only loop is feasible and partly scaffolded; end-to-end adaptation evidence is open.** | Phase 2 should freeze versioned Red configs, collect fresh rollouts, compare new-only vs mixture retraining, evaluate fresh held-out suites and old-suite retention, and preserve manifests. Do not call checkpoint creation “co-evolution.” |
| EWC fine-tuning and “experience replay buffers” (10, 13) | EWC appears in research/future-scope documents, but no active EWC implementation was found in blue/ or red/. Replay/rollout storage in training does not by itself implement EWC or PPO replay-safe updates. | **Implementable research extension; absent now and not yet shown necessary.** | First measure forgetting under frozen vs new-only vs mixture retraining. Add EWC only if mixture retraining leaves a measured retention problem. PPO must use fresh on-policy trajectories. |
| Zero-day resilience / resilient against mutating threats (2, 4, 7–13) | Project has multiple simulator Red types and held-out seed/strategy work, but no real unknown vulnerabilities or payload mutation. Main32 uses Discovery Red; other studies use Finite/LLM Red. | **Held-out simulator robustness is testable; real zero-day resilience is not supported.** | Replace “zero-day” with “held-out simulator strategy generalization” if that experiment passes. Use unseen attack configurations and fresh seeds, not merely different seeds for the same narrow policy. |
| 16-week plan and Phase 1/2 milestone claims (12) | Actual project records show work from September through October, extensive experiment branching and evolving gates. The 16-week graphic schedules Docker/eBPF, single-zone RL, 2–3-zone MAPPO, retraining, benchmark, and final defense, but those are not the project’s actual completed sequence. | **Historical planning graphic only; no evidence it was delivered to.** | Replace with the actual dated milestone record and future dependency sequence. Do not retroactively mark imagined weeks as completed. |
| Local workstation GPU 32GB, AWS/GCP, OpenAI/Anthropic APIs, Docker/K8s/ELK/OSQuery/Packetbeat (14) | Experiment records show local Python/Ollama use and Kaggle jobs, including CPU cohorts and a separate CUDA screen. No committed cloud budget, API use, or team Docker/Kubernetes/ELK/OSQuery/Packetbeat integration was found. The exact 32GB local GPU availability is not established by reviewed project artifacts. | **Resource choices are possible but the Zeroth slide is not a verified inventory or budget.** | Replace with actual per-experiment runtime/device and confirmed resources. Report actual spend only from receipts/quotas; do not assume services merely because listed. |
| Classification as container security/Threat Intel and SDG 9 / SDG 8 benefit (15–16) | AI/cybersecurity research classification fits the project. Containerized range, threat intelligence generation, reduced SOC workload, and resilient-infrastructure benefit are not implemented/measured. | **Classification/motivation is fine; impact statement remains aspirational.** | Keep SDG alignment as a motivation statement only. Avoid claiming improved infrastructure resilience or analyst wellbeing without a measured real-world study. |
| Literature-review tables (5–6) | Project documentation already warns that review-deck bibliography entries need validation; the original final-review audit specifically found unvalidated references. | **Not a system capability question; citation quality is an open document check.** | Verify title, authors, venue, DOI, and whether the cited work supports the described advantage/drawback before presenting. Replace unverifiable entries with exact primary sources. |

### 5.1 Overall feasibility conclusion

The core idea is feasible in a staged form if the claim is narrowed: a CAGE4-based, partially observed, multi-agent simulator; scripted and bounded LLM planning over native actions; controlled Blue policy research; and a later simulator-only retraining/retention loop. Those components are consistent with existing code and project ownership. The complete Zeroth vision is **not** a near-term extension of the current code: it combines a cyber-range build, real telemetry, real containment, multi-agent communication, zero-day claims, enterprise audit/SIEM integration, and adaptive retraining. Each needs a separate implementation boundary and independent evaluation.

Recommended staged sequence:

1. Finish one clean CAGE4 comparison with strong controls and fresh held-out seeds.
2. Freeze the shared Red scenario, evaluation metric, artifact, and policy interfaces.
3. Demonstrate a simulator-only fresh-rollout retraining/retention loop.
4. Add one scoped integration at a time (for example, a sandboxed web target or one telemetry source), with its own reset, safety, and metric tests.
5. Measure runtime response and false-positive impact only on the platform that supports those units and denominators.

Do not take on Docker + eBPF + Juice Shop/DVWA + MITRE/SIEM + EWC/federated MARL as one Phase 2 deliverable. The project’s own team instructions prioritize simulator correctness, strong baselines, fresh rollouts, and reviewed cross-role contracts first.

### 5.2 Tracks required to realize the Zeroth vision

The full vision needs coordinated workstreams. They can start partly in parallel, but each integration depends on shared interfaces and a common evaluation plan. The sequence below translates the Zeroth slides into buildable deliverables; it does not promise that the numerical performance goals will be met.

| Track | Deliverable | Zeroth claims addressed |
|---|---|---|
| 1. Contracts and evaluation foundation | Agree on scenario, observation/action, event-log, metric, and artifact contracts across Blue, Red, and Environment. Define training, development, and held-out scenarios/seeds. | Makes later results comparable and traceable. |
| 2. Isolated container range | Run one deliberately vulnerable application in a resettable lab, then add zones and services. Verify network isolation and egress policy. | Docker range, actual software targets, multi-zone environment. |
| 3. Telemetry and detection | Add one real telemetry source, map it into versioned Blue-visible features, then test anomaly scoring against a fixed baseline and held-out data. | eBPF/packet telemetry, anomaly scoring, benign-noise suppression. |
| 4. Red strategy generation | Extend the bounded planner with versioned, validated strategy variants for the chosen environment. Restrict generated choices to authorized targets and supported actions. | LLM Red planning and attack variation. |
| 5. Blue defense and coordination | Adapt and compare Blue policies against strong controls on CAGE4 and the container environment. Add a message channel and bandwidth limits only if cooperative communication is required. | MAPPO, decentralized execution, low-bandwidth coordination. |
| 6. Response actions | Add and validate supported actions, measuring service impact and verified outcomes. Real host actions need a separate actuator and safety review. | Containment, isolation, traffic control, recovery. |
| 7. Evolution and retention | Add Red strategies, collect fresh Blue rollouts, train a candidate, evaluate on new and old suites, then promote or reject. Test EWC only if simpler mixture training leaves a measured forgetting problem. | Closed-loop retraining, co-evolution, EWC. |
| 8. Audit and SIEM integration | Produce structured action/evidence/result records, review MITRE ATT&CK mappings, and validate SIEM ingestion. Add an LLM auditor only after the evidence trail is reliable. | Reasoning audit, MITRE trail, enterprise SIEM. |
| 9. Benchmark and claim validation | Freeze definitions and protocols for detection delay, verified clearance, service impact, false-positive rate, actor latency, and end-to-end response time. | Sub-100 ms, MTTR under five seconds, >90% false-positive reduction, benchmark report. |

**Recommended order:** begin the CAGE4 evolutionary-training study and isolated container-range build in parallel; add telemetry, Red variants, and a Blue adapter for one containerized app; complete a small end-to-end slice; then expand zones, actions, agents, and integrations after reset/evaluation are reliable. A CAGE4 checkpoint will not directly control a web app; it needs an adapter and fresh evaluation on that app.

Juice Shop and DVWA are deliberately vulnerable training targets, not production applications. Keep the range isolated and test egress. Docker documents internal networks without a default external gateway; avoid attaching a dual-homed service that restores external access. See the [OWASP Juice Shop guide](https://owasp.org/www-project-developer-guide/assets/exports/OWASP_Developer_Guide.pdf), [OWASP DVWA listing](https://vwad.owasp.org/app/damn-vulnerable-web-application-dvwa/), and [Docker Compose networking](https://docs.docker.com/compose/how-tos/networking).

Even after these tracks, the Zeroth targets remain empirical gates, not guaranteed outcomes. The current controlled Blue comparison did not establish learned superiority over both strong controls. A CAGE4 result also does not establish response time or false-positive reduction on real software/SOC telemetry.

## 6. Zeroth → First → Phase 1 Final Review

The final-review column is proposed evidence-based content; no final presentation file has been created here.

| Topic | Zeroth Review: vision | First Review: interim report | Phase 1 Final Review: recommended state |
|---|---|---|---|
| Environment | Air-gapped Docker range, OWASP targets, eBPF/packet telemetry | CAGE4/CybORG simulator; future container work acknowledged in places | State the CAGE4-only implementation boundary; list container and live telemetry as unstarted integration |
| Red | LLM dynamic exploit chaining, mutation, pivoting, zero-day APT | LLM Red model comparisons and planned attacker integration | Bounded LLM planner selects goals/focus for native CAGE actions; separate simulator strategy generalization from payload mutation/zero-day claims |
| Blue | Decentralized per-zone MAPPO, bandwidth constraints, sub-100 ms containment | MAPPO pilot and baselines reported; pilot uncertainty limited | Corrected multi-agent wrapper, strong baselines, several methods; controlled primary comparison did not clear its gate |
| Detection | eBPF/packet feature pipeline, Isolation Forest/LOF, benign-noise suppression | Ten features/preprocessing and detector tests | Simulator-derived features; detector PR-AUC result separated from defensive return; no real telemetry or >90% reduction claim |
| Response | Packet drop, container freeze, honey routing, token/session revoke, <5 s MTTR | Simulator Remove/Restore actions and aspirational deployment goals | Current simulator actions and verified-remediation boundary; report step-based delays; real containment/time remains separate |
| Adaptation | Red mutation pool, MAPPO retraining, EWC, continual co-evolution | MAPPO pilot/training work underway | Many training experiments and mixture proposals, but no demonstrated end-to-end improvement+retention loop |
| Audit and SOC integration | LLM auditor, MITRE trail, enterprise SIEM | Structured experiment logs/exports proposed or partial | Report trace/manifests as research artifacts; no validated independent auditor or connected enterprise SIEM |
| Project result | >90% FP reduction, autonomous sub-5s response, robust low-bandwidth defense | Interim pilot returns and preliminary model table | Transparent evidence matrix, null/inconclusive findings, seed/protocol provenance, and explicit open gates |
| Timeline/resources | 16-week schedule and cloud/API/Docker stack budget | W1–W6 status and local runtime notes | Actual completed milestone chronology; verified runtime/resources only; future dates assigned after review |

| Topic | First Review: original slide/content | Phase 1 Final Review: replacement/update | Reason |
|---|---|---|---|
| Project framing | A self-evolving cyber-range to reduce alert fatigue and response latency | A CAGE4 simulator research prototype evaluating Blue policies under partial observation and multiple Red behaviors | Separates motivation from measured outcome. |
| Objectives | Container range; mutating LLM Red; decentralized MAPPO; sub-second containment; retraining/auditor/SIEM | Status-code wrapper, baseline/policy research, Red path, and evaluation; mark fresh-rollout retraining and real integrations as future work | The real-world targets are not demonstrated. CTDE does not mean peer messaging. |
| Scope | CAGE4, five Blue, “six Red,” Green; platform target and milestone mixed | Name the CAGE4 scenario and action/observation boundary; identify Red variants separately; verify exact Red count before using it | Prevents ambiguous topology and agent claims. |
| Progress since Zeroth | MAPPO “started”; CybORG, EPyMARL and Ollama “running”; local Windows CPU | Corrected wrapper, baselines, Red adapter/LLM path, evaluation harnesses, and named policy research; distinguish implementation, smoke, completed experiment, and open integration | EPyMARL is not the current main32 runner; runtime stacks differ by study. |
| Architecture | Conceptual end-to-end loop, with no current/planned distinction | CAGE4 → Red behavior / Blue-visible wrapper and policy → native reward plus evaluation-only labels → experiment artifacts; solid tested elements, dashed future integration | Shows policy/evaluation trust boundary. |
| Method | Ten features, padded slots, masked MAPPO, 2,394-step pilot | Explain seeded joint reset, host bounds, observation/evidence ages, validity vs evidence masks, pending-action lock, and remediation verification; name the protocol for each result | Current correctness work supersedes the simple pilot summary. |
| Technology | Python 3.12, EPyMARL and Ollama together | Exact runtime per study; main32 Python 3.11.16, NumPy 1.26.4, SciPy 1.12.0, Torch 2.2.0 CPU, scikit-learn 1.7.2; Ollama/LLM runtime separately | Avoids claiming one shared environment or an EPyMARL integration for main32. |
| Blue results | Two-seed Blue-vs-LLM-Red and other model table combined; weak uncertainty treatment | Five-training-seed main32 table with strong controls and paired intervals; state that the predeclared primary criterion failed | Keeps the sample unit and reuse status visible. |
| Red model results | Four Red model cells and timing, source outputs unavailable per audit | Use separately documented Red comparison with caveat; keep original cells in backup until raw sources are recovered | Prevents unverified numbers from leading the result section. |
| Detector | Preprocessing and tests presented as supporting evidence | Separate detector PR-AUC from defense return; show risk-only policy result and reused-episode limitation | Predictive ranking and helpful action selection are different questions. |
| Completion | “4 of 5 milestones complete (80%)” | Deliverable/evidence matrix with complete/partial/open; say no accepted rubric or weighting is recorded | The first percentage is arbitrary. |
| Remaining work | Longer MAPPO, features, retraining, incident metrics, documentation | Fresh held-out evaluation; clean LLM Red provenance rerun; architecture confirmation; shared interfaces; retraining and retention; real integrations separately scoped | Turns generic tasks into reviewable gates. |
| Timeline | W1–W6 with TBD review/final dates | Dependency-ordered next steps; add calendar dates only when confirmed | Stale week bars hide ownership and readiness. |
| References | Generic repositories, papers, frameworks, model cards | Verified source citations plus frozen config, revision, runtime, seed lists, and exact report | Makes every result traceable. |

### 6.1 Direct result comparison with the first-review slide

Slide 9 of the original deck reported a two-seed Blue comparison against “LLM 8B” with returns: MAPPO pilot **-1606.5**, Sleep **-2530.0**, masked random **-1372.0**, and round robin **-573.5**. The slide did not show the two seed IDs, actor/checkpoint identity, Red request configuration, uncertainty, or a linked artifact. The 8 October accuracy audit therefore removed this table from its edited deck as obsolete/source-insufficient.

The later Blue LLM-Red report measures a different, better-described experiment: three development seeds 7629–7631, 400 ticks, pinned llama3.1:8b digest, same actor checkpoint for unguarded/guarded settings. It reports Lancer **-670.7**, unguarded actor **-1333.7**, and guarded actor **-557.7**. The later report says Lancer is the local no-decoy variant, not published-Lancer parity. A 24-seed follow-up then reports Lancer **-637.3** versus guarded actor **-516.7**, but carries the recovered-evaluator caveat described above.

These numbers should appear as a **historical-to-current evidence comparison**, not as one continuous leaderboard. The first-review values do not have enough provenance in this checkout to calculate changes or attribute them to training. The later 3-seed report supports the more specific finding that guard behavior dramatically changed the actor’s result; the 24-seed report is a suggestive follow-up and needs a clean reproduction. The main32 values are not a replacement for these LLM Red numbers because main32 used Discovery Red and a separate protocol.

## 7. Slide-by-slide final review content

This is a 16-slide proposed deck. It updates all 12 original slides and adds four research slides. Dashboard content is intentionally omitted.

### Slide 1 — Title and scope

**Title:** Self-Evolving AI Cyber-Range for SOC Automation.  
**Subtitle:** Phase 1 Final Review: CAGE4 Simulator Research Prototype.  
Include team, guide, actual date, and evidence cutoff of 27 October 2026. Do not use “final” to imply production deployment or completion of future phases.

### Slide 2 — Problem statement and Phase 1 question

Suggested copy: “Security operations need repeatable ways to test investigation and response policies before operational use. Phase 1 builds and evaluates a simulator-based Blue defense in CAGE4 under partial observation and scripted or planner-driven attacks.”

Research question: “Can Blue-visible evidence, action history, and learned host prioritization improve native defense return over strong heuristic controls across attack regimes?”

Boundary statement: “Alert-fatigue reduction and operational response latency motivate this work; neither has been measured in a live SOC.”

### Slide 3 — Objectives and status

| Objective | Current evidence | Status |
|---|---|---|
| Build repeatable simulator-backed Blue interface | Seeded joint wrapper, complete host bounds, action masks, pending/completion tracking, regression coverage | Implemented in checkout |
| Compare scripted and learned defense policies | Sleep/random/heuristic controls; PPO/MAPPO and other experimental families; frozen reports | Implemented; learned superiority unresolved |
| Exercise multiple attackers | Scripted Discovery/Finite and local planner-driven LLM Red paths | Implemented as separate regimes, not one benchmark |
| Adapt to attacks and retain old capability | Research plans and experiment harnesses; no completed closed-loop retention result | Open / next phase |

Use “centralized training/decentralized execution” only for a specific architecture that follows that design. Do not claim agent-to-agent messaging without an implemented and evaluated message interface.

### Slide 4 — Scope and observation boundary

Show CAGE4 Enterprise, five native Blue agents, named Red configuration, background Green behavior, and 400-tick episodes where applicable. Draw two data paths: Blue policy input receives Blue-visible observations and evidence memory; evaluation only receives privileged simulator truth. State that compromise and detection are not the same measurement. Mention that CAGE4 source is locally preserved and unmodified, but its exact upstream commit is unknown.

### Slide 5 — System architecture and decision cycle

Diagram: Seeded CAGE4 reset → all five Blue observations → per-host features and presence mask → simulator validity mask → policy/baseline → joint action → pending-action completion → updated Blue evidence and native team reward → evaluator/manifests.

Under the diagram: one shared reset; one pending multi-step action per Blue agent; requested action, completion, observed verification, and privileged truth are separate facts; actor cannot read evaluation labels. Report native reward separately from any optional experimental shaping.

### Slide 6 — Correctness work before policy comparison

1. **Reset and seed:** one shared simulator reset per episode and requested seed applied.
2. **Host coverage:** no silent truncation; bounds reflect 17-host single-subnet agents and the 51-host three-subnet agent.
3. **Pending actions:** busy agent cannot replace an in-flight action; results remain attributed to the original target.
4. **Evidence and remediation:** passive observations merge; Remove success does not assert clean; remediation enters verification; simulator validity and evidence restrictions remain separate.

Explain that policy comparisons on inconsistent worlds, missing hosts, or fictional verification labels are not interpretable.

### Slide 7 — Features, action space, and policy families

Explain the ten-channel base host representation from Blue-visible session, process, file, and connection observations plus role/zone context. Optional temporal groups add freshness, belief, last-action/result, and mission context. Host presence is separate from padded zeros. Actions include Sleep and Monitor plus host-targeted Analyse, Remove, and Restore; zone traffic blocking is not enabled.

List policy families at a useful level: heuristic scheduling; flat/factorized actors; SetActor/candidate attention and behavior cloning; risk/novelty inputs; PPO/MAPPO and A2C; RvS and IQL/offline AWR; guarded and hybrid variants. Add: “Implemented families and saved weights are research inventory. Only matched, documented comparisons support performance claims.”

### Slide 8 — Evaluation protocol

Explain paired seeds, fixed horizon and Red suite, training RNG as the model replication unit, reused vs held-out evaluation seeds, native reward as primary score, source/config/runtime hashes, and complete tails. Explain that PPO/MAPPO retraining uses fresh on-policy rollouts. List outcomes separately: native return, host compromise-time, service impact, requested/executed/completed action, Blue-observed verification, and evaluation-only truth. Note that not every historic number follows the later metric repair.

### Slide 9 — Main Blue model comparison (new)

Use the full main32 table from Section 4.2, preferably as paired delta intervals with a zero reference. State prominently: **“The primary risk + novelty arm did not meet the predeclared requirement to beat both Lancer and the coverage guard.”**

Footnote: five training RNGs; same 32 reused Discovery Red development environments, seeds 8221–8252; 400 ticks; higher native return is better; df=4 paired training-seed t interval; final block 7809–8200 untouched. Put Sleep/random in a second panel so small control differences remain visible.

### Slide 10 — Detector score is not defense outcome (new)

Left panel: HGB PR-AUC 0.72794 vs logistic 0.63107 over eight reused development episodes. Right panel: main32 zero-input return -74.59 vs risk-input -81.21. State that the first measures ranking; the second measures a policy under a distinct protocol. Evidence does not show that the higher detector ranking improved defense. Label novelty discrimination weak and avoid “probability of compromise” wording without calibration.

### Slide 11 — Planner-driven Red results

Present the 3-seed pilot as a failure-mechanism lesson: unguarded learned actor performed worse than Lancer; adding a coverage guard improved the same weights; guarded-vs-Lancer uncertainty crossed zero. The 24-seed follow-up may be shown only with a visible caveat: +120.6 return, interval [+1.4, +239.8], 16/24 seeds won, one checkpoint/training seed, recovered rows 0–42 from an unavailable uncommitted evaluator. Call it suggestive/source-caveated. Keep it separate from Discovery Red. Hold the original four-model table in backup until source outputs are recovered.

Takeaway: “Coverage discipline can limit a severe investigation-concentration failure; added learned value remains unsettled.”

### Slide 12 — Architecture studies: leads, not winners (new)

Summarize three screens: attention behavior cloning reached teacher-like performance in an older eight-seed protocol without online RL; longer two-RNG hybrid screen had positive but tail-sensitive means and did not show repeatable Red-regime gate routing; gate ablation’s learned gate was 11.7 below fixed blend while fixed blend was positive against its controls. Label all “exploratory; two RNGs/historical protocol; requires confirmation.” Keep detailed per-regime counts in notes.

### Slide 13 — Negative findings and research lessons (new)

Show that strong heuristic scheduling remains a high bar; detector accuracy can rise without policy benefit; guard constraints can improve safety but also constrain useful behavior; and coverage, freshness, detection, verified remediation, and native return are distinct outcomes. Report documented PPO fine-tune divergences, risk/recency failures or coverage collapse, max-age changes that reduced age without detectable return gain, and a learned state gate that was nearly constant and training-seed dependent.

### Slide 14 — Phase 1 deliverable matrix

| Deliverable | Evidence/status |
|---|---|
| CAGE4 Blue wrapper correctness | Implemented; regression/live checks recorded in Blue handoffs |
| Baseline and policy research | Broad policy inventory; matched comparisons exist; robust learned win not established |
| Scripted and LLM Red paths | Local simulator implementations; distinct protocols/runtimes |
| Reproducible experiment evidence | Strong for selected frozen studies; some historical/recovered artifacts caveated |
| Unified shared contracts/runtime/evaluator | Proposed contracts; cross-role acceptance/integration open |
| Closed-loop retraining and retention | Not established as an end-to-end result |
| Real telemetry and response | Outside current simulator scope; future integration |

Footer: “Phase 1 is a simulator research prototype; completion against committee rubric must be confirmed.” Do not use an unweighted percentage.

### Slide 15 — Phase 2 sequence and dependencies

1. Freeze review claims, source revisions, configs, runtimes, and artifacts.
2. Rerun source-caveated LLM Red comparison under committed code with fresh audited seeds.
3. Select an architecture lead (fixed blend is a candidate) and run preregistered multi-training-seed confirmation on fresh suites in Discovery and Finite regimes.
4. Complete shared scenario factory, metric, artifact, and Blue/Red/Evaluation interface reviews.
5. Demonstrate retraining with fresh rollouts; measure new-strategy response and retention on frozen old suites.
6. Scope real telemetry/container/SIEM work as a separate stage with explicit validation criteria.

Only add dates after the schedule is confirmed. Show owners and dependencies in the appendix.

### Slide 16 — References and result provenance

Use verified primary references and place exact source commit, frozen config/manifest, runtime lock, training RNG seeds, evaluation seeds, attacker variant, result report, and hashes in notes or appendix. Verify paper titles, authors, venues, DOI/model cards. The 8 October audit says some original bibliography entries were not verified.

## 8. Phase 1 evidence and open-gate matrix

| Area | Evidence present | Gate before a stronger claim |
|---|---|---|
| Simulator correctness | Reset, coverage, pending-action, verification fixes; wrapper tests/live checks in handoffs | Consolidate exact passing-check inventory and bind it to the review source revision |
| Baselines | Sleep, random, round-robin/Lancer and guarded heuristics in studies | Publish an agreed baseline table within one attacker/horizon protocol |
| Learned policy | Multiple training/architecture families and frozen cohorts | Fresh multi-seed confirmation against strong controls, including lower-tail behavior |
| Detector | HGB/logistic ranking report with separated train/calibration/test episodes | Fresh attacker/scenario validation, calibration/threshold definition, intervention benefit |
| Red | Scripted adapter, LLM planner/executor, trace reports | Stable scenario factory, pinned model/runtime, agreed attack/mission metrics, clean provenance |
| Metrics | Action and privileged-label audit work; seed ledgers/manifests | Accepted shared schema and incident/service metrics across roles |
| Adaptation | Training scripts and scenario-mixture proposals | Demonstrated fresh-rollout retraining and retention without leakage/regression |
| Real-world translation | Research goal only | Separate telemetry/action adapter, review, execution validation, and operational study |

## 9. Claim rules for the presentation

1. Name the scenario: Discovery, Finite, and LLM Red are separate.
2. Name the uncertainty unit: main32 uses five training RNGs, not 32 episodes or 160 model-episode cells.
3. Mark reused development versus fresh held-out evidence.
4. Report all arms and tails; do not select a checkpoint after seeing evaluation results.
5. Keep detector PR-AUC, Blue reward, verified remediation, and privileged truth distinct.
6. Do not pool absolute returns across attackers, horizons, runtimes, observation versions, reward settings, or guards.
7. An interval crossing zero means no directional effect was established; it does not prove equality.
8. Sub-100 ms actions, sub-second containment, MTTR under five seconds, >90% false-positive reduction, zero-day resilience, real eBPF/Docker containment, and production integration remain future targets unless measured on matching systems.

## 10. Final presentation checklist

- Confirm committee rubric/date; neither is an accepted project contract in the reviewed docs.
- Use the direct comparison in Section 5 to revise all original slides, not only append results.
- Lead with main32 and disclose that its predeclared gate failed.
- Separate detector results from policy results.
- Treat LLM Red 24-seed result as caveated until clean rerun; exclude the original four-model Red table pending raw output.
- Include negative architecture/guard outcomes.
- Keep exact provenance in notes/appendix for every figure.
- Replace “80% complete” with the evidence matrix.
- Do not describe a real cyber range, production SOC integration, SIEM connection, or closed-loop retraining result that is not present.
- Recheck local and remote source state before distributing the deck.

## 11. Sources reviewed

### Project instructions and design

- AGENTS.md
- docs/current-state.md (27 October correction; historical snapshots treated as historical)
- docs/team-workflow.md and docs/contracts.md (proposed APIs, not accepted runtime contracts)
- docs/implementation-plan.md
- docs/archive/blue-agent-plan.md
- docs/archive/handoff.md (correction at the beginning)
- docs/status/blue-session.md

### Source files inspected

- blue/core/wrapper.py, masking.py, obs_features.py, baselines.py, telemetry.py, collect_telemetry.py
- blue/harness/README.md, architectures.py, attention_actor.py, scoring.py
- blue/training/harness_rl.py and training/evaluation inventory
- red/README.md, strategy_adapter.py, llm_agent.py, planned_agent.py, blue_env.py, evidence/lancer-development-20261007.md
- Blue and Red test inventories; tests were not rerun for this documentation task

### Results and status

- docs/research/first-review-deck-accuracy-audit-20261008.md
- docs/status/blue-gpt-main32-results.md, blue-gpt-main32.md, blue-gpt-pilot-audit.md
- docs/status/blue-llmred-evaluation.md, docs/status/blue.md, docs/status/blue-gpt-harness.md
- docs/research/blue-specialist-c0-result-20261025.md
- docs/proposals/blue-gate-ablation-experiment-20261025.md and blue-task-modular-architecture-20261024.md
- docs/status/environment.md, eval.md, red.md (dated status narratives; caveats retained)
- runs/first_review/First_Review.original.pptx
- assets/Zeroth_main.pptx (17 slides; text and architecture/workflow/timeline diagrams inspected)

## 12. Provenance and limits

This is an evidence synthesis and proposed presentation content, not a completed PPTX or certification that Phase 1 meets an external rubric. It uses local code and records visible on 27 October 2026. Remote freshness could not be verified because the configured Git metadata directory is read-only. Some experiment outputs are ignored by Git and are represented through their status reports/manifests; recorded run caveats are carried into proposed wording.

