# Blue harness design audit and opt-in redesign

Branch: gpt-blue. Audited main32 experiment source: 6942d584f15231a7f19add0a510a04f7ff3ad229. Existing checkpoint architecture and native simulator rewards are preserved. This is a tested implementation proposal, not a demonstrated defensive improvement. Older foundation/no-agent prose in current-state.md disagrees with the implemented branch; current experiment artifacts establish the measured state here.

## What the completed comparison establishes

Five training RNG seeds, 32 shared reused development episodes per model. Independent training replication is five, not 160. Intervals below are conditional on that fixed environment suite.

| Agent | Mean native return | Difference from guard | 95% training-seed t interval |
|---|---:|---:|---|
| Lancer | -79.906 | -1.688 | deterministic control |
| Guard | -78.219 | 0 | deterministic control |
| RL zero | -74.594 | +3.625 | [-2.751, +10.001] |
| RL risk | -81.213 | -2.994 | [-10.630, +4.642] |
| RL risk+novelty | -74.713 | +3.506 | [-3.149, +10.161] |

The registered superiority criterion failed. Risk minus zero is -6.619 with interval [-17.573, +4.335]. Neither PPO superiority nor ML input value is established. Severe episodes remain part of performance. Initial action-request hashes match across arms for 5, 1, 3, 1, 1 iterations respectively: identical first four iterations is not a universal observation.

## Audit of the components

| Component | Finding | Implemented response or next check |
|---|---|---|
| Simulator, masks, completion | Simulator legality differs from evidence restrictions; busy actions and completed observations matter | Reuse tested wrapper and protected rules; no simulator edits |
| Remediation | Hook selects sweep investigation targets only, after CONFIRMED and VERIFY rules | Keep enforced remediation isolation; fewer repairs alone cannot establish cause |
| Coverage | Age guard prevents indefinite starvation but cannot guarantee useful prioritization | Keep max_age=80, compare against guard alone |
| Observation deltas | One-tick deltas may expire while an agent is busy | Retain positive observable event stamps until completed analysis, with ages; no ground truth |
| ML objective | Compromise probability is not the causal value of investigating a host | Keep score as information, not action-value; no V(s)-based override gate |
| Detector fitting | Highly correlated repeated host rows; calibration has only two independent episodes | Equal episode/host weights, free-agent decision context, predefined sigmoid calibration; preserve per-episode reports |
| Novelty | Weak standalone prediction; decision utility unresolved | Default risk-only, novelty retained for explicit ablation, never called compromise probability |
| Actor | Pointwise ranking lacks candidate-set context and explicit base priority | Masked local encoding plus candidate mean/max context and base priority |
| Central critic | Legal-candidate means erase zone observations while agents are busy | All owned-host summaries, busy bits and agent identity; no privileged state |
| Optimization | Joint actor/critic gradient clipping couples unrelated gradient scales | Separate optimizers and clipping, fixed-scale Huber value loss |
| Credit | Old returns already discount elapsed ticks; do not invent a duration bug | Explicit decision-time returns/GAE with intervening forced rewards, lambda=1 default; partial rollouts deliberately unsupported |
| Exploration | Old sampled training overrides ~84–87%; greedy deployment differs | Exact teacher-mixture behavior likelihood, 20% maximum departure probability on unguarded choices; primary execution samples same distribution |
| Evaluation | Greedy residual and sampled mixture are different policies | Name greedy diagnostic separately; nested policy RNG replicas; zero-residual mixture control |
| Data/scenarios | Four training episodes and fixed attacker/horizon limit generalization | Expand only after generated-range seed audit; separate episode and training RNG variation |
| Reproduction | Detector golden predictions alone do not validate trained-policy behavior | Require policy replay/action hash golden checks before another remote comparison |

A read-only shadow backward pass on frozen risk_s1 and training episode 7706 measured actor gradient norm 0.01974 and critic norm 215.103. Combined clipping multiplies both by 0.004649. This establishes coupling, not a corresponding reduction in Adam parameter updates or a cause of the failures.

### Reproduction gate discovered during this audit

Downloaded risk_s1 on episode 8236 scores -97 locally versus -225 remotely. Guard reproduces -81. A plain training-runner evaluation also returns -97, so trace logging alone does not explain the discrepancy. A second risk_s2 replay on 8233 gives -100 versus remote -205. Runtime package versions and scorer hashes were matched. A controlled local replay now establishes thread-sensitive ranking: risk_s1 gives -136 with one Torch thread and -97 with two. Its first differing choice is tick2 with identical features, candidates and base scores; two residual rankings differ by only 7.45e-9. A declared 1e-6 near-tie intervention gives -152 and identical full request digests with either thread count. Stability improved on this episode, defense did not. This does not establish the full reason for the remote -225. Platform effects and hash-dependent ordering remain unresolved hypotheses. These local traces cannot explain the remote failures until requests and returns reproduce. The new trace audit explicitly marks mismatches. This does not erase the remote evaluation results.

## Why MAPPO and what to use instead

The existing shared actor with centralized state-value training and decentralized decisions already follows CTDE/MAPPO structurally, restricted to investigation. MAPPO is PPO in this multi-agent architecture, not a separate cure for preprocessing defects. The [MAPPO paper](https://arxiv.org/abs/2103.01955) emphasizes the importance of critic representation, value treatment and training choices. Its benchmark success does not establish superiority in CAGE4.

The new runner exposes `--algorithm mappo` and `--algorithm a2c`. Both share observation processing, architecture, protected rules, rewards and rollout budget. MAPPO uses clipped likelihood-ratio updates and exact categorical KL checks. A2C makes one fresh full-batch policy-gradient update per rollout, without PPO ratio/clipping or replay. This synchronous implementation is related to the actor-critic family studied by [Mnih et al.](https://arxiv.org/abs/1602.01783); it is not asynchronous A3C. Policy-update counts differ and must be reported.

[Discrete SAC](https://arxiv.org/abs/1910.07207) is a plausible later alternative: it adds action-value learning and replay. It requires a verified next-candidate/mask replay contract and partial-observation treatment, so it is not a one-line optimizer replacement. Off-policy methods are not categorically rejected as sample-hungry. QMIX ([paper](https://arxiv.org/abs/1803.11485)) imposes a value-factorization assumption that needs testing for coupled defenses. Continuous SAC/TD3 is not a natural representation of discrete host selection.

A recurrent policy may help partial observability ([DRQN](https://arxiv.org/abs/1507.06527)), but requires sequence training and episode/busy-state reset verification. Persistent observable event memory is the smaller implemented first step. A contextual bandit cannot simply replace RL while investigation changes future detection and remediation state.

The set architecture is inspired by [Deep Sets](https://arxiv.org/abs/1703.06114). Decision-time GAE adapts [GAE](https://arxiv.org/abs/1506.02438) to actual elapsed ticks. These are engineering proposals, not proven optimal CAGE4 designs. Sigmoid versus isotonic calibration follows [scikit-learn calibration documentation](https://scikit-learn.org/1.7/modules/calibration.html); the small independent calibration episode count remains a limitation.

## Implemented and checked

New code is opt-in: blue/harness/{redesign,learning,calibration}.py and blue/training/harness_{redesign,ml_redesign,design_audit,trace_audit}.py. No old checkpoint silently migrates. Manifests include feature version, source hashes, package versions, scorer and actor hashes; resume rejects drift. Artifacts remain ignored.

Two full 400-step episodes per iteration, two iterations, completed for each learner using the original detector. Both produced training means -85 then -61; neither is performance evidence. Sampled override rate was ~17.7%, demonstrating the exploration constraint works. Full-horizon MAPPO evaluation completed on reused seed 8241 with Lancer, guard, Sleep, random, sampled and greedy-diagnostic execution. Further source changes added a matching untrained-mixture control; short-horizon end-to-end checks verify that path separately.

The weighted detector candidate reports macro PR-AUC 0.742 versus its matched weighted logistic comparator 0.655 across eight reused test episodes. This is not a fair old-versus-new detector comparison without re-evaluating the original HGB on the identical free-agent mask. It does not establish defense improvement. Two calibration episodes and one model fit cannot support broad generalization.

## Next experiment, before scaling

1. Resolve old checkpoint local/remote replay mismatch; freeze trained-policy golden predictions, full request digest, package and CPU/thread/hash settings. Fail closed on mismatch.
2. Run one-change v1 optimizer ablation to distinguish optimization repair from the bundled v2 redesign. Hold architecture, inputs and policy execution fixed.
3. Screen v2 MAPPO versus A2C with identical rollout budgets, at least five training RNG seeds, common reused development episodes and nested policy RNG replicas. Include Lancer, guard, Sleep, random and the untrained mixture. Report all runs, tail failures, completed remediation and coverage. A sampling limit does not bound defense loss.
4. Test risk information versus zero under the selected fixed harness. Evaluate novelty separately; do not select based on one episode or best checkpoint.
5. Audit additional disjoint environment/attacker suites before claiming generalization. Preserve 7809–8200. Main32 episodes are used development data, never fresh final tests.

No new training comparison is launched by this redesign. A separate bounded private Kaggle replay audit is prepared to check the cross-machine discrepancy using frozen weights only. Promotion requires beating both non-learned strong controls under the predeclared analysis, with no hidden checkpoint selection. Component-specific claims require isolated ablations, not credit for the entire bundle.

### Follow-up execution contract

V2 now pins one Torch thread and records it in the runtime manifest. Greedy-diagnostic selection uses explicit 1e-6 near-tie resolution; primary sampled execution is unchanged. The frozen v1 policy is unchanged. `blue.training.harness_numerical_audit` records threads, tolerance, model hashes, complete request digest and decision margins, and correctly counts hook choices via a read-only recorder. `ops/kaggle/harness/replay_audit.py` runs only declared diagnostic replays. No training and no reserved episodes are included.
