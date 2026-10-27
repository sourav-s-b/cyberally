# Persistent baseline anchoring for Blue policy updates

**Status:** frozen two-RNG screen; no confirmation claim. 2026-10-25.

## Question

With the fixed 0.5 SetActor/attention architecture held constant, does a
persistent KL penalty to the frozen zero-residual Lancer-mixture distribution
reduce harmful policy drift and improve native return compared with the current
PPO update?

## Why this is a distinct policy change

The current learner already uses PPO clipping, early stopping at measured
old-policy KL 0.02, and `explore=0.5`, which puts fixed probability mass on the
Lancer-selected action. This screen does not call those controls a new method.
The intervention adds a continuing forward KL penalty from the current policy
to the frozen zero-residual policy distribution on every actor minibatch. The
reference is the exact masked distribution produced by the same Lancer base
scores, temperature, teacher action and exploration mixture, with actor
residuals set to zero. Coefficient is fixed at 0.1; no tuning on evaluation
returns.

This is a neural online adaptation inspired by conservative policy updates and
baseline bootstrapping. The tabular/offline guarantees in those papers do not
apply to this implementation.

## Arms and fixed settings

| Arm | Actor | Policy update |
|---|---|---|
| PPO control | fixed 0.5 SetActor/attention blend | existing clipped PPO, measured old-policy KL stop 0.02, no persistent reference penalty |
| Reference-KL | identical actor and rollout | same PPO plus `0.1 * KL(reference || current)` |

Both retain the strict Lancer action guard and identical candidate support,
scorer, native reward, event-time Monte Carlo credit (`lambda=1.0`),
`bonus=1.0`, `explore=0.5`, `temperature=0.5`, actor LR 0.003, critic LR
0.0003, six iterations, four balanced 400-tick Discovery/Finite episodes per
iteration, minibatch 256, four actor epochs, four critic epochs, and one CPU
thread. The exact marginal categorical probability is used for PPO likelihoods.

## Seeds and scope

- Training RNG 0: environment seeds 7730–7737, cycled for 24 episodes.
- Training RNG 1: environment seeds 7738–7745, cycled for 24 episodes.
- Development: 7746–7753, evaluated under both Discovery and Finite Red.
- The authoritative seed audit reported 7730–7753 free immediately before
  freeze. Train and development seeds are disjoint. These do not overlap the
  7809–8200 final evaluation reservation or the 8501+ reservation.
- Two training RNGs make this a descriptive screen only. The two algorithms
  share environment seeds within each training block and development seed / Red
  pairs for paired analysis.

## Outcomes and decision rule

Primary contrast: Reference-KL minus PPO-control paired native return, reported
separately by Red regime and training RNG. Also report each arm against its own
untrained policy and strict Lancer. Include mean, median, IQM, win fraction,
10th percentile, host coverage, learned/forced decision counts, old-policy KL,
reference KL, and the return distribution. The current evaluation harness does
not expose verified remediation counts, so those are unavailable in this screen
and must not be inferred from action requests. Keep training RNG as the
independent model unit; evaluation episodes are paired observations, not
additional trained models.

The screen advances only if Reference-KL has a positive macro mean versus PPO
control in both RNGs, is nonnegative versus strict Lancer in both Red regimes,
and does not materially worsen the lower tail or host coverage. Otherwise record
the result as negative/inconclusive and do not tune this coefficient on the
same seeds. Any claim of a robust RL win requires a new preregistered
confirmation with at least eight training RNGs and a separately audited
development block.

## Sources

- Kakade & Langford, [Approximately Optimal Approximate Reinforcement
  Learning](https://dl.acm.org/doi/10.5555/645531.656005), conservative policy
  iteration.
- Laroche et al., [Safe Policy Improvement with Baseline
  Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html), baseline
  fallback under uncertainty.
- Schulman et al., [Trust Region Policy
  Optimization](https://arxiv.org/abs/1502.05477), explicit policy-distribution
  constraint. PPO clipping/early stopping is retained as a separate safeguard.

## Context from the completed Kaggle Qwen run

The output archive is stored at
`blue/results/event_time_rl/kaggle_output/qwen_eventtime_results.tar.gz` and
was verified against the kernel log SHA256. Five GAE runs averaged about
`-162.8` training return per episode versus about `-165.6` for five Monte
Carlo runs; both groups reported full training coverage. These are training
episodes against the LLM Red regime, not held-out results for the scripted-Red
policy screen. Keep `lambda=1.0` here so the only treatment is the persistent
reference-KL term. Treat the Qwen comparison as motivation for a separate
credit-assignment follow-up, not evidence that GAE wins on this screen.
