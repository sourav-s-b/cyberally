# Blue learning follow-up plan for Luna

**Date:** 2026-10-18
**Branch:** `blue/finite-red-check` at `2f44db117d98791103fd99329005f7d56f84ffe9`
**Purpose:** Decide whether to continue imitation learning, test a newer DAgger-style variant, or build a hierarchical Blue policy. Keep the current finite-Red evaluation frozen.

## Run-ready status for the three approaches

Executable preparation: [implementation and execution card](blue-followup-execution-card.md), draft `blue/configs/followup-preparation.json`, and read-only `scripts/blue_followup_readiness.py`. Use the checker for live progress; counts below are a historical snapshot. The execution card distinguishes technical blockers from research decisions and specifies the work/tests still needed for each track.

The fresh BC/DAgger confirmation is complete: 448/448 episode rows, all frozen source/scorer/model hashes, and paired seed lists validated. Results are recorded in [the confirmation report](finite-red-dagger-bc-confirm-20261018.md). BC and DAgger both beat Lancer and the age-80 guard on this fresh episode block with positive intervals across the five fixed training models. DAgger did not beat BC; that interval overlaps zero. This confirms these checkpoints on new episode seeds, not general performance across new training data or Red strategies.

The cross-Red BC evaluation is also complete: all 864 episode jobs and frozen artifacts validated. BC clearly beats Lancer and the age-80 guard on finite Red, but its intervals against Lancer overlap zero on discovery and random Red; on random Red it loses to the age-80 guard. Sleep Red is a zero-return tie. The all-regimes claim therefore fails. Full results and limits are in [the cross-Red report](finite-red-bc-redsuite-20261018.md). Keep BC as a finite-Red investigator candidate; do not call it a general winner across Red modes.

The literature-based method-to-role review is in [the specialist policy review](blue-specialist-policy-literature-20261018.md). Direct CAGE4 evidence supports PPO-trained task subpolicies under an expert rule manager more strongly than training a manager from the start. Apply that as a design lead, not a score prediction: first audit separate development traces, choose one justified second specialist, and test composition with the manager fixed. Train a manager only after a routing failure appears in the specialist comparison.

| Approach | Preparation status | What must happen before a launch |
|---|---|---|
| OEC | Protocol drafted; no implementation or frozen expert-query target | After confirmation, continue only if reducing Lancer queries is a goal. Implement student-prefix/expert-suffix collection, match the BC episode budget, audit the new seed block, then freeze. |
| Posterior-conditioned MARL | **No-go for current Red suite** | Needs a genuinely new/mixed Red regime. Existing finite/discovery checks failed their gates; mid-episode switching needs Environment support. No training command should be prepared against the old suite. |
| Hierarchical specialists | Runtime/interface and four-arm protocol drafted; second specialist not selected | After confirmation, inspect Blue-visible traces for one repeated recovery/verification/coverage error and verify usable data. If no signal/data, stop. If evidence exists, select one specialist, freeze its method and four arms, then audit fresh seeds. |

The next concrete action is hierarchy C0: audit development traces for a repeated, observable non-investigation failure and verify adequate data. OEC is conditional on an explicit expert-query-efficiency goal; the policy-return results alone do not justify it. Posterior conditioning stays deferred pending a useful Blue-visible regime signal. Audit and reserve any new seed blocks before training.

## Executive decision

Three follow-up directions are worth investigating, but they answer different questions:

1. **On-Policy Expert Corrections (OEC):** can a cheaper, segment-level expert correction scheme improve on DAgger/BC?
2. **Posterior-conditioned MARL:** can Blue adapt to mixed or switching Red strategies using a Blue-visible belief over attacker behavior?
3. **Hierarchical specialists trained with different methods:** can each task benefit from a suitable objective and training method, and do those specialists improve defense when combined under a shared manager?

Work on the literature, protocol, interface, and isolated prototypes for all three in parallel. Do **not** launch their simulator experiments in parallel with each other or with the active DAgger evaluation. Training jobs share a CPU-heavy simulator, fresh-seed ledger, and evaluation budget. Each implementation session must use its own worktree. Freeze one experiment at a time before spending its seed block.

The common runtime should remain one harness: one source of simulator resets, seed accounting, Blue-visible observations, legal-action masks, action/result ledger, and artifact manifests. Specialist policies plug into that runtime; they do not create separate collectors or seed registries.

## Completed experiment: use as the gate for follow-up training

The completed matched-budget finite-Red pilot is in:

`blue/results/finite_red_dagger_bc_pilot_20261018/`

Its frozen allocation is:

- Five model-training RNG seeds, `0–4`.
- 25 per-arm training episode seeds, `7730–7754`.
- 14 whole-episode validation seeds, `7755–7768`.
- 32 shared evaluation seeds, `7769–7800`.
- Four completed controls: Lancer, age-80 guard, Sleep, and masked-random.
- Five BC and five DAgger model manifests; both use native reward at simulator evaluation. DAgger uses five rounds with beta `1, 0.5, 0.25, 0.125, 0.0625`.

**The run is now complete.** All five BC and all five DAgger manifests are complete; each policy was evaluated on the same 32 seeds, with four controls on that block. Frozen source hashes and model/evaluation artifact links validate. The full report and per-episode artifact paths are in [the DAgger–BC results report](finite-red-dagger-bc-pilot-20261018.md).

The paired DAgger-minus-BC difference is `-2.38` native return, 95% interval `[-11.67, +6.91]` over the five model-training seeds, so DAgger did not demonstrate an improvement over BC. BC and DAgger each exceed Lancer and the age-80 guard on this block with positive five-seed intervals. Treat that as promising but exploratory pending a fresh confirmation block. DAgger agreement is slightly below BC (`0.379` vs `0.383`), and its worst episode is worse (`-492` vs `-301`); both have full coverage. Retain BC as the simpler investigation candidate for follow-up planning.

**Do not overwrite or retune this run.** The frozen script reports each arm versus Lancer and the guard but omits the direct DAgger-minus-BC comparison. That comparison has now been calculated from the saved rows without modifying the frozen script. The audit verified:

1. Require complete manifests, artifact/source/scorer hashes, and identical 32-seed lists for every arm and control.
2. For each training RNG seed, calculate the mean paired episode-return difference `DAgger - BC` over the same 32 evaluation seeds.
3. Report the mean and two-sided 95% t interval over the five training-seed means. Also report mean/p10/min return, every episode return, per-seed paired differences, Lancer agreement on learner-visited decisions, and coverage.
4. Separately compare both arms with Lancer and age-80 guard using the predeclared five-training-seed interval. Sleep and masked-random are context controls.
5. An interval overlapping zero is inconclusive. The positive BC/DAgger intervals versus Lancer and guard are exploratory because they use only five training seeds and one evaluation block; confirm on a fresh audited block before a broader claim.

### Decision tree after current evaluation

- **DAgger improves agreement and return versus BC, and its `DAgger - BC` interval is above zero:** retain it as the better imitation method. If it still does not beat Lancer/guard, claim only improved teacher imitation, not superior defense. Consider OEC as an efficiency follow-up.
- **DAgger raises agreement but return is neutral/worse:** imitation accuracy is not the bottleneck or some Lancer decisions are harmful under student features. Stop adding imitation rounds; inspect paired episode tails and action disagreements before another method.
- **DAgger is neutral/worse on both agreement and return:** do not expand DAgger/OEC now. Retain the simpler BC or existing rules, depending on the paired return and tail results.
- **Both BC and DAgger look better than Lancer/guard but intervals overlap zero:** no superiority claim. Add fresh training seeds and a fresh audited confirmation block; do not tune on `7769–7800`.
- **A policy’s mean is better but its lower tail or coverage materially worsens:** do not advance it. Inspect the affected episodes and preserve the age guard and remediation/verification rules.

## Parallel work tracks

These tracks can proceed independently through paper review, protocol drafting, API/interface design, and toy/unit-level prototypes. Keep changes in separate worktrees and separate draft files. Do not change shared contracts or launch simulator runs without the gates below.

### Track A — OEC, the closest DAgger variant

**Question:** Can student-started episode prefixes reduce expert-query cost while preserving the BC investigator's return?

**Evidence:** The 2026 ICML OEC paper proposes student-started trajectories followed by an expert switch, reporting gains over standard imitation on multi-turn software tasks. Those results are not cyber-defense evidence and do not establish a gain here. [Paper](https://proceedings.mlr.press/v306/lauffer26a.html)

**Preparation actions (parallel-safe):**

1. Map the paper's procedure to the investigation hook. OEC starts with student-controlled rollouts and switches to the expert partway through. The reported gains are on multi-turn software tasks, not cyber defense. [PMLR paper](https://proceedings.mlr.press/v306/lauffer26a.html).
2. Freeze the switch tick and training data before collecting episodes. Start from the BC checkpoint, run the student for the prefix, then Lancer for the suffix. If Lancer labels are also queried on the student prefix, describe that as a DAgger/OEC hybrid rather than attributing it to OEC alone.
3. Keep policy inputs and expert targets Blue-visible; no privileged state or reward labels.
4. Keep urgent remediation, verification, pending-action handling, and the age-80 guard identical to BC.
5. Match episode and expert-query budgets. Compare OEC with BC and DAgger on new training seeds. Record expert decisions, student/expert prefix/suffix actions, and final whole-agent returns.

**Run gate:** The fresh confirmation of fixed BC/DAgger models is complete and found no DAgger gain over BC. OEC is justified only by a concrete expert-query or data-efficiency question, or a newly identified student-distribution failure. Freeze that question and protocol before running. Use `blue.training.seed_audit` for new training/validation seeds; do not reuse `7730–7800` or `8253–8284`. Use at least 32 paired held-out seeds and five training seeds if budget permits.

**Follow-up:** If OEC beats DAgger and BC on the direct paired interval without worse tails/coverage, test a fresh confirmation block. If it only matches DAgger, prefer the simpler implementation. If it loses, stop the OEC branch.

### Track B — Posterior-conditioned MARL for Red shifts

**Question:** Can a Blue team infer whether the current attacker behaves like one of several Red regimes and adapt its policy under mixtures or mid-episode switches?

**Evidence:** A 2026 *Artificial Intelligence* paper evaluates posterior-conditioned MARL on CAGE4 under mixed and switching attacks. This targets attacker-regime adaptation, not finite-Red parity. [Paper](https://www.sciencedirect.com/science/article/pii/S0004370226001323). Existing project evidence records a failed 32-seed skill gate and a 48-cell mode/regime screen where no mode cleared its gate; Blue's session notes also say Red is fixed at environment construction, with no mid-episode switching ([prior results](../status/blue-session.md)). Repeating a learner on the same regimes is not justified.

**Preparation actions (parallel-safe):**

1. Audit whether this checkout contains reviewed, reproducible Discovery/Finite/Random or switching Red implementations and identify their exact commits. Do not assume old docs or branch names mean the adapters are available here.
2. Draft a contract proposal before coding the shared belief. Name producer/consumers, fields, defaults, tests, and rollout order. The policy input must be derived from Blue-visible joint histories and any cross-agent communication must be explicitly available to the deployed policy.
3. Keep Red identity out of policy inputs. The contract allows known training-scenario labels to supervise a separate belief estimator, provided runtime belief is inferred only from Blue-visible history. Name and log that estimator separately.
4. Define fixed-Red and domain-randomized baselines; train on multiple Red regimes; hold out both complete seeds and attacker switches/rates. Freeze the regime suite before training.
5. Report native return, p10/min, coverage, switch-detection delay, posterior calibration using Blue-observable evidence, and performance before/after a switch.

**Run gate:** No-go for the current queue. Reconsider only with a new Red regime or switching capability and a frozen mixed-regime suite. Episode-level mixtures could use the existing Red variants, but the policy may infer the regime only from Blue-visible history. Mid-episode switching needs Environment support first.

**Follow-up:** If new regime evidence clears the gate, advance only if posterior conditioning improves held-out mixed-regime return without material finite-Red or tail regression. Otherwise keep this track closed.

### Track C — Hierarchical specialists trained with different methods

**Question:** Can specialists trained with task-appropriate methods improve individual decisions and the combined Blue agent? BC, DAgger, and IQL are training methods; the trained policy is the specialist that executes. Different methods are permitted, but algorithm variety is not an objective. A deterministic specialist remains a valid winner.

**Evidence:** CAGE4 hierarchical MARL research decomposes investigation, recovery, and traffic control. The local review reports that the expert-rule master outperformed flat IPPO, while the learned meta-controller was approximately flat-IPPO. This favors a rule manager with frozen specialists before attempting a learned manager. [CAGE4 H-MARL paper](https://arxiv.org/abs/2410.17351); [local review](blue-architecture-review.md).

**Method references:** [DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses imitation on learner-visited states. [IQL](https://arxiv.org/abs/2110.06169) learns from offline transitions and rewards; it is a candidate for choosing among supported behaviors, not a substitute for missing action/outcome data. The proposed combination below is our hypothesis, not an established result of these papers. Check earlier local BC/IQL experiments before proposing repeats; identify precisely which decision, data, or objective changes.

#### C0 — Audit decisions, data, and method fit (parallel preparation)

Map actual `HarnessPolicy` branches and prior tests. The harness already has scripted remediation/verification, pending-action handling, an age guard, and an investigation hook. The local H-MARL review favors the rule manager; prior skill and regime tests often failed their gates. Confirm which decisions are available; the investigation hook cannot intercept forced branches.

| Specialist | Decision and initial baseline | Candidate method and objective | Required evidence/data |
|---|---|---|---|
| Investigation | Choose a host; direct Lancer ranking and current BC/DAgger | BC for expert ranking; DAgger for imitation on learner-visited states; OEC only through Track A | Blue-visible candidate features/masks and expert picks; episode-held-out validation; agreement plus whole-agent return/coverage |
| Recovery | Choose among eligible responses or their timing; current rules | Imitation if a reliable teacher exists; IQL only after offline-data audit, optimizing native return | Observed state, executed action, duration, subsequent observations, reward, and episode boundaries; enough examples of alternative actions in relevant states |
| Verification | Decide when to recheck after remediation; current rules | Supervised timing/outcome prediction if observable targets exist; policy choice evaluated separately | Blue-visible recheck results, delays, missing/censored outcomes; action completion must not be labeled as confirmed clearance |
| Coverage | Prioritize neglected hosts; age-80 guard | Keep rule initially; consider learned scheduling only after demonstrated rule failure | Observable visit ages, pending work and resource use; full host coverage and service impact |
| Manager | Route between available specialists; rule manager | Learn option selection only after specialists are frozen | Valid option masks, duration/termination, action conflicts, and native team returns under each routing alternative |

For every candidate, Luna must produce a one-page specification: observable input fields, allowed decisions, baseline, training target/reward, data source and counts, missing evidence, method choice, validation metric, integration metric, and stop condition. If expert labels suffice, start with supervised learning. If delayed consequences matter and offline action coverage is adequate, investigate IQL. If coverage is inadequate, report that blocker and specify a bounded development-data collection experiment before training. Do not infer alternative-action outcomes from logs that always take the same action.

#### C1 — Specify one shared runtime and specialist interfaces

Draft a proposed API, then map it to existing code before implementation. Each specialist receives Blue-visible observation/history, eligible candidates, simulator-valid masks, task context and remaining decision budget. It returns a legal action proposal or explicit abstention. Document required memory reset, pending-action behavior, option termination, and deterministic fallback. Confidence may be emitted only with defined semantics and validation.

The harness owns the single simulator, resets, seed ledger, action arbitration, executed-action attribution and episode logging. Keep simulator-valid masks separate from policy restrictions. Specialists may have separate models and training datasets but must use compatible feature/action versions and a shared episode identity. The manager resolves conflicts and preserves urgent-response precedence; two specialists must not independently step the simulator. Cross-agent observations require an explicit communication contract. Several modules inside one Blue policy do not alone establish a MARL implementation.

Every specialist artifact must record its training method, objective, feature version, source/scorer/model hashes, data and seed provenance, supported action scope, and fallback. Local objectives may differ; native team return, tails, and coverage judge the composed agent. Propose any reward or shared-interface change for affected-role review before merging. Privileged compromise labels remain evaluation-only by default.

#### C2 — Freeze the investigator, then qualify one additional specialist

1. The finite-Red confirmation and cross-Red evaluation are complete. Freeze BC as a finite-Red candidate; broad superiority was not established, and BC lost to the common-harness guard baseline on Random Red. Diagnose on separate development traces before a broad composition claim. Retain the Lancer investigator as the baseline for that comparison.
2. Use development traces to identify one specific, repeated error in recovery, verification, or coverage. Name the observable signal that might resolve it. If none exists, stop at the architecture/data audit.
3. Select one candidate method for that decision using C0. Compare against its existing rule with every other module fixed. Any extra method sweep consumes development data and budget, never confirmation seeds.
4. Split datasets by whole episodes, recording temporal history and action durations. For offline RL, audit reward attribution and terminal/truncation handling; missing transitions or hidden-state targets block training. For supervised verification, assess calibration and missing outcomes, then evaluate the resulting action rule in the simulator.
5. Before simulation, check reset isolation, illegal-action rejection, pending-action attribution, fallback, and privileged-input exclusion. Tests must exercise interface behavior, not merely mirror model code.

#### C3 — Test specialists alone and together

Use a fixed rule manager and a four-arm comparison. Let I be the proposed investigator and S the second specialist. For recovery/verification experiments, use the same age-80 guard and remaining harness in all arms: B is the existing guard baseline, and unguarded Lancer remains a separate control. For a coverage experiment, replace coverage only in S/IS. This isolates specialist changes from adding the guard:

| Arm | Investigator | Second specialist |
|---|---|---|
| B | Existing Lancer investigator | Existing rule |
| I | Frozen learned investigator | Existing rule |
| S | Existing Lancer investigator | Frozen learned specialist |
| IS | Same learned investigator as I | Same learned specialist as S |

Use at least five independent training replicates for new learning comparisons, paired evaluation seeds across all arms, and an initially planned 32 or more episodes per replicate. Reuse the exact component checkpoints between I/S and IS so composition is the changed variable. Include the common control policies. Estimate simulator cost before launch; do not count repeated evaluation of deterministic B as independent training replicates.

Predeclare IS−B as the primary integration comparison. Report I−B, S−B, IS−I, IS−S, and the interaction `(IS−I)−(S−B)` as diagnostics. Calculate differences within training replicates and intervals across replicates; preserve every episode. These intervals are conditional on the evaluation block, so require a fresh confirmation block for a broader claim. Record native mean return, p10/min, coverage, response delays and service impact where measurable. Freeze numerical tail/coverage regression tolerances and minimum useful gain before evaluation. An interval overlapping zero is inconclusive; it does not establish parity. Multiple exploratory comparisons must not become a selectively reported win.

**Outcome → follow-up:**

- **Both individual specialists help and IS improves on B without violating tail/coverage gates:** confirm IS on a fresh audited block. Claim added value from composition only when the comparisons with the individual arms support it.
- **Individual arms help but IS loses or develops worse tails:** freeze training; inspect routing conflicts, workload and changed state distributions. Fix one interaction on development data, then repeat with fresh confirmation seeds.
- **Only one specialist helps:** retain that specialist and the existing rule for the other. Do not require multiple neural modules to call the experiment successful.
- **Local accuracy improves but team return does not:** inspect the objective-to-action link and workload effects. Do not promote on imitation accuracy or prediction metrics alone.
- **Intervals remain wide:** estimate additional seed cost from observed variance; expand a frozen comparison if worthwhile, or report inconclusive. Do not search more algorithms on the confirmation block.
- **Neither specialist helps:** retain the existing harness and report the negative result. Larger networks or more levels need a new evidence-backed hypothesis.

#### C4 — Learn a manager only after C3

Name a routing error that can be detected from Blue-visible history, define available options and their termination/duration, and freeze all specialists. Compare a learned manager against the rule manager with identical specialists, observations, episode budgets and held-out seeds. Account for elapsed simulator ticks in returns/discounting when options have different durations. Keep native team return as the common objective. Do not jointly train manager and specialists in this first test.

Advance only after the manager has a positive paired return interval over the rule manager and meets the predeclared tail/coverage gates on fresh seeds. Otherwise retain the rule manager. Joint fine-tuning is a later experiment requiring a demonstrated benefit and a new frozen protocol.

## How to parallelize safely

| Work | Can proceed now? | Constraint |
|---|---:|---|
| OEC method and efficiency protocol | Yes | Only run for a frozen expert-query/data-efficiency hypothesis |
| Posterior-MARL regime audit | Complete | Existing gates failed; no-go until Red capabilities/suite change |
| Specialist decision/data audit (C0) | Yes | Parallel read-only investigations by specialist; shared data/seed inventory |
| Hierarchy API and composition protocol (C1/C3) | Yes | Separate draft files/worktrees; reconcile interfaces before integration |
| DAgger–BC fresh confirmation | Complete | Keep artifacts frozen; use the confirmation report to choose the next experiment |
| OEC or hierarchy training | No | Wait for active confirmation and a frozen protocol/seed plan |
| Posterior-conditioned training | No | Requires new Red-regime evidence and a reviewed interface |

If multiple people or models prepare tracks, give each a separate worktree and ask for the same deliverables: (1) exact hypothesis, (2) cited method and what is genuinely new, (3) code/interface changes, (4) source and seed plan, (5) baseline/evaluation protocol, (6) go/no-go criteria, and (7) estimated simulator cost. Reconcile their proposals before any training run.

Within Track C, parallelize investigation-method review, recovery/verification data audit, and manager/coverage interface design. A coordinator owns the seed inventory and reconciles their outputs into one C2 candidate and the C3 protocol. Track A supplies an optional investigation training method; Track B supplies a future attacker-belief input only after its own gates. Keep both out of the initial composition comparison so improvements remain attributable. After the current evaluation, independent jobs within one frozen experiment may share an explicitly measured worker budget, but no worker may launch an unbounded nested pool; allocate seeds and artifact directories centrally before launch.

## Shared scientific and engineering gates

- Only Blue-visible observations enter policy inputs, masks, and attacker beliefs. Keep evaluator-only privileged state in its own code path and artifact namespace.
- Match simulator episode budgets when comparing learning methods. Use common held-out seeds within each paired comparison and disjoint train/validation/evaluation blocks.
- Freeze the code/source hash, scorer hash, preprocessing, config, Red version, and actual seed lists before runs. Record exact dependency commits.
- Include Lancer, age-80 guard, Sleep, and masked-random when evaluating a new policy. Preserve every evaluation episode and report mean plus p10/min tails and coverage.
- Report intervals over independent model-training seeds; do not treat 32 episodes for one trained model as 32 independent training seeds.
- Do not adjust the current experiment after seeing its evaluation rows. Any tuning must use a fresh development block and a later untouched confirmation block.
- Keep experiment outcomes in a dated report with null and negative results. Update only Blue-owned status files after results are complete; do not claim that a local doc notified other roles.

## Expected handoff from Luna

For each track, return a short proposal, not an unreviewed training launch. Mark it `ready`, `blocked by contract/runtime`, or `not worth pursuing`; list exact branch/dependency commits, files to change, seed block request, runtime estimate, validation tests, and decision thresholds. The next actual simulator run should be selected only after the DAgger-vs-BC comparison is complete.

For Track C, additionally deliver the completed C0 specialist matrix, proposed module interface, offline-data coverage report, one selected second-specialist experiment, the four-arm C3 configuration, numerical promotion/stop thresholds, and a result-to-next-action table. Mark all new interfaces and unrun experiments as proposed. This handoff authorizes preparation; it does not change the frozen DAgger run or establish any new performance result.
