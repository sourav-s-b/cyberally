# Follow-up implementation card

Preparation date: 2026-10-18. Branch: `blue/finite-red-check`, base `2f44db117d98791103fd99329005f7d56f84ffe9`, with local source changes recorded in the frozen confirmation manifest.

The machine-readable draft is `blue/configs/followup-preparation.json`. Empty seed arrays and null thresholds are deliberate unresolved decisions, not defaults. These configs are specifications; there is no three-track training launcher yet.

## Check readiness while evaluation runs

From the worktree:

```bash
python scripts/blue_followup_readiness.py --scorer /home/sourav/projects/cyberally/blue/results/gpt_harness/scorer.pkl
```

This reads the live episode files, validates source/model/scorer hashes and episode provenance, and lists remaining preparation. It uses the standard library and starts no simulator processes. A partial JSON write is reported as pending: retry it; persistent unreadability needs investigation. Exit 2 means a validation error; exit 0 can still mean incomplete evaluation. Read `confirmation_ready_for_analysis`. Track readiness remains false because the listed research and implementation requirements have not been completed.

The completed launcher already wrote analysis. If aggregation alone needs repeating:

```bash
bash scripts/blue_dagger_confirm.sh --analyze
```

Do not rerun the whole launcher while its current workers are active. Evaluate the completed BC and DAgger results, tails and coverage before choosing checkpoints. The confidence intervals across model seeds are conditional on the shared episode block; report that limitation. Do not select a lucky individual model seed.

## A: implement expert-suffix imitation

Proposed files: `blue/training/harness_expert_suffix.py`, its focused tests, and a bounded launcher. Reuse `DaggerActor`, `fit`, `pack_rows`, `unpack_rows`, and the existing feature provider through imports. Keep the frozen training modules intact.

1. Initialize all three arms from the same BC checkpoint within each of five model seeds. Compare BC continuation, DAgger continuation and expert-suffix continuation. Count the shared warm-start data in the overall data budget.
2. Use five new 400-tick episodes per model per arm. Pair the environment seeds across arms. In the suffix arm, the student acts before tick 200; Lancer acts from tick 200 onward. Collect labels only for suffix investigation-hook decisions. The switch is once per episode, shared across agents; reset it between episodes. Forced actions remain under the common rules.
3. Keep prefix labels out of the learner dataset. The harness already calculates Lancer scores before invoking the hook: distinguish computed scores, stored labels, expert actions and simulator episodes. This experiment cannot claim less expert computation without a separate change to the scoring path.
4. Freeze a numerical training-label budget before collection; apply a deterministic episode-stratified label cap to each arm. Report discarded labels and actual counts. Match simulator episodes and optimizer update caps; do not claim matched label budgets from episode counts alone. Use the same whole-episode validation block and early-stopping rule.
5. Keep all data, manifests and model checkpoints in a new artifact directory. Freeze source/scorer/checkpoint hashes, exact seeds, switch tick, label cap and sampling RNG. Resume must reject drift in any of these.
6. Test switch boundaries/reset, no prefix labels, candidate legality, no teacher-score input, identical arm initialization and resume refusal after drift. Start with an isolated short smoke episode after the current job finishes; only then freeze the full pilot.

This is an OEC-inspired adaptation, not a faithful paper reproduction. The existing DAgger result does not show a gain over BC. Only pursue this comparison for a specific label-efficiency or distribution-shift hypothesis. Primary comparison: suffix minus BC continuation. Report DAgger comparisons too. Failure or inconclusive return with no label-efficiency benefit closes this track.

Proposed cost: 75 training episodes across three arms, plus validation collection, plus 608 evaluation episode jobs (three arms × five models × 32 seeds, plus four controls × 32). Do not quote a wall-time estimate until a measured episode throughput is available. Freeze useful-gain/tail/coverage thresholds before evaluation.

## B: prepare posterior conditioning

The wrapper currently offers discovery, finite, verbose, random and sleep Red at episode construction. Episode-level mixtures are technically possible. Earlier skill/mode failures make the research case weak; they do not prove a new belief method cannot help. Mid-episode switching is a separate missing capability.

First deliver a development-data regime-separation audit using Blue-visible histories, holding out whole episodes. Compare a learned belief against the training prior; assess log loss and calibration. Known scenario labels may supervise the separate estimator, never enter the deployed actor directly. If histories do not distinguish strategically relevant regimes, stop before policy training.

For a viable result, draft a coordination proposal naming Environment/Red as regime producers and Blue/Evaluation as consumers. Specify optional belief probabilities, regime-vocabulary version, source tick, missing-history prior, reset, and whether history is local or explicitly communicated. Default policies must work without this field. Specify normalization, future-timestamp rejection, reset isolation, missing-input fallback and hidden-label exclusion checks. Review the shared interface before merging.

Keep three equal-budget arms: fixed-regime policy, domain-randomized policy without a belief, and the same architecture conditioned on the inferred belief. Hold out seeds and mixture weights. Conditioned minus domain-randomized is the main test of belief value. Switch-delay metrics apply only after actual switch support exists. No simulator edits or posterior trainer are implemented by this preparation.

## C: prepare the first specialist composition

Investigation already has an executable hook; recovery and verification branches are outside it. Current imitation datasets contain candidate features and expert labels. They do not supply a validated recovery dataset with complete action/outcome transitions. The older IQL audit identifies terminal/reward and target-network issues; resolve those against current code before reusing that pipeline.

First deliver an offline evidence table from development traces: decision branch, eligible actions, chosen action, observable cue, count of opportunities, pending-action duration, observed outcome, and missing fields. Do not use the active confirmation block to choose a specialist or tune thresholds. If current traces lack these fields, implement a separate collector and budget development episodes; do not infer missing counterfactual outcomes.

Choose one repeated observable failure and one second specialist. Prefer supervised learning when a trustworthy observable target exists. Recovery IQL requires complete `(observation, executed action, duration, reward, next observation, terminated, truncated)` records and adequate action support; completion of Remove is not a clean-host label.

**C0 audit result (2026-10-18): no specialist selected.** The available finite-Red candidate-audit traces contain investigation, age-sweep, recovery, verification and pending-wait branches, with Blue-visible features and observed action completions. They do not record eligible alternatives for recovery, verification or age-sweep decisions, and they do not provide a reliable clean-host target. Every host was investigated in all 24 audited episodes per model, so coverage is not a demonstrated failure here. Verification often leaves Blue belief at `VERIFY` or `UNKNOWN`, but this alone does not show that the rule is wrong. See [the C0 evidence table and data gaps](blue-specialist-c0-audit-20261018.md). The next step is a frozen development-only collector/criterion; do not train until it exposes a repeated actionable failure.

Proposed interface: `propose(observation, eligible_actions, context) -> action_or_abstain`, plus episode reset. The rule manager owns precedence, pending actions, legality, fallback and the single environment step. Track a proposal separately from the executed action. Start with local observations; any shared history requires the communication proposal above. Verify fallback, reset isolation, action conflicts and attribution before simulation.

Freeze B=Lancer investigator+common harness, I=learned investigator+common harness, S=Lancer investigator+second specialist, IS=both specialists. For recovery/verification, all four arms retain identical age-80 coverage and other rules; B is the guard baseline. For coverage, only S/IS replace that coverage rule. Keep unguarded Lancer as a separate control. Reuse exact component checkpoints across arms; primary comparison is IS−B, with IS−I and IS−S exposing whether composition adds value. With five learned replicates and 32 paired episodes: 512 episode jobs for B/I/S/IS, plus 96 for unguarded Lancer/Sleep/random = 608. Collect/train the second specialist separately; those costs are additional.

If individual specialists help but composition fails, inspect manager conflicts on development traces. If only one helps, keep that one. Train a manager only after a measured routing failure with frozen specialists. Freeze numerical gain/tail/coverage limits before the held-out comparison.

## Execution order

Preparation for A/B/C is independent. Finish confirmation, choose the justified experiment, audit and reserve fresh seeds, implement and check its missing components, freeze its config/source, then launch with one shared worker budget. No hard-coded new seeds are allocated in this draft. Existing seed reservations still apply. Preparation has not established any new policy performance result.
