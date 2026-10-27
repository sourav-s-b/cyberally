# Value-aware policy router for Blue: experiment plan

**Status:** deferred second-stage experiment. Do not launch this value-aware
router until the equal-budget SetActor / fixed-blend / learned-gate ablation
has shown that the learned gate adds value over a fixed blend. The current
hybrid run is complete; this revision reflects its results. The plan remains
a proposal, not a preregistration or a claim that value-aware routing will
improve Blue.

**Revised order:** first run the gate-isolation comparison in the final section
of this document and in `blue-routed-hybrid-architecture-20261025.md`. Only
advance to this value-aware router if the learned gate reproducibly beats the
fixed blend and the learned policy improves over its untrained counterpart.

## Question

Can Blue improve on the current situation-routed SetActor/attention hybrid by
choosing between complete candidate policies using learned estimates of their
return, while falling back to the existing Lancer prior when the learned value
estimate is uncertain?

The target remains a real RL policy: PPO updates the candidate policies and
router from native team reward. The Lancer policy is a fixed reference/fallback,
not the learned solution and not a replacement for the RL objective.

## Why this is a later architectural test

The current hybrid learns one sigmoid gate and interpolates the two branches'
score residuals. That does not directly ask which branch would yield better
return, and its gate may get weak or noisy credit through the shared PPO update.
The next model should make the routing decision explicit and give it a value
signal.

Prior work provides motivation, with limits:

- Context-aware policy-reuse work learns state-dependent choice among policies
  using option values; it also identifies sparse rewards and value-estimation
  errors as problems for contextual routing. This motivates a value head and
  makes calibration a required check, not an assumption. [CAPS, AAMAS 2019](https://people.iiis.tsinghua.edu.cn/~zhang/pub/aamas19-CAPS.pdf)
- Option-Critic learns intra-option policies and termination conditions along
  with a policy over options. This motivates testing slower switching if
  per-decision route changes prove unstable. [Bacon et al., AAAI 2017](https://ojs.aaai.org/index.php/AAAI/article/view/10916)
- A Q-ensemble offline-RL paper uses prediction uncertainty to penalize
  out-of-distribution value estimates. Using ensemble disagreement to temper
  our router's confidence is an adaptation, not a result established by that
  paper for this task. [An et al., NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/3d3d286a8d153a4a58156d0e02d8570c-Abstract.html)
- Mixture-policy work shows mixtures can improve flexibility, but also notes
  that gradient estimators for mixtures can have higher variance. We will
  implement and test the categorical mixture likelihood carefully. [He et al.,
  ICLR 2025 submission](https://openreview.net/forum?id=TImqVqOQuC)
- Risk-sensitive mixture work supports separating average-return behavior
  from cautious behavior in selected states. We will report lower-tail results,
  but will not add a CVaR training objective in this first comparison. [Luo et
  al., Reinforcement Learning Journal 2024](https://rlj.cs.umass.edu/2024/papers/Paper81.html)

These works do not validate the proposed architecture on CAGE4. The experiment
tests the transfer of these ideas to our environment.

## Model under test

At each eligible Blue decision, retain the current strict action-validity mask
and construct three masked candidate distributions over the same legal
decisions:

1. **Set policy:** current pooled SetActor proposal.
2. **Attention policy:** current candidate-attention proposal.
3. **Lancer prior:** the existing deterministic/scored Lancer proposal under
   the exact same strict guard and candidate set.

A small state-conditioned router produces probabilities over these three
proposals. Sample the proposal and then its action; calculate the PPO log
probability for the resulting marginal mixture, including the router
probability. Do not interpolate logits as the current hybrid does. Log the
selected proposal, router probabilities, action probability, and gate entropy
for every eligible decision.

The router may use only existing Blue-visible input: candidate features,
evidence age/belief, normalized Lancer score summaries, and public role/zone
context. It must not receive Red identity, hidden sessions, privileged state,
or future outcomes.

### Value and uncertainty heads

- Train one centralized, branch-conditioned critic on native team return,
  using the current runner's Blue-visible centralized input and actual elapsed
  decision duration. Condition it on the selected proposal and executed joint
  Blue decision so the router gets an estimate of each proposal's downstream
  return.
- Use a small bootstrapped ensemble (three heads sharing an encoder) to
  estimate disagreement. The routing score is `mean predicted return -
  kappa * ensemble disagreement`; `kappa` is selected in a separate calibration
  stage using prediction error only, never evaluation return.
- Apply the uncertainty penalty to the **router's value comparison**, not to
  the environment reward. Under high disagreement, shrink learned proposals'
  routing logits toward the fixed Lancer prior. Do not claim this is a safety
  guarantee.
- Because Blue decisions are coordinated and rewards are shared, the critic
  must condition on the team decision context. Independent per-agent critics
  are not acceptable for the primary experiment.

Start with per-decision routing. Add an option-style minimum route duration or
learned termination head only in a later ablation, and only if the current
experiment logs show frequent route switching associated with poor outcomes.
This avoids bundling temporal abstraction with value-aware selection.

## Comparison and controls

| Arm | Actor / router | Purpose |
|---|---|---|
| A | Current SetActor PPO | architecture baseline |
| B | Value-aware Set/attention/Lancer router PPO | proposed RL architecture |
| Controls | Strict Lancer; untrained versions of A and B | heuristic and exploration controls |

Keep the current hybrid's completed results as context; do not reuse it as a
control unless its saved artifacts and source hashes pass a fresh integrity
check. All arms use the same scorer, strict guard, observations, action masks,
native reward, optimizer settings, PPO update count, Red regimes, and evaluation
seeds. No online changes to the simulator or reward.

## Candidate screening protocol

Use two independent PPO RNGs, with four balanced iterations of two 400-step
episodes each per RNG (one Discovery and one Finite episode per iteration).
That gives eight training episodes per RNG/model and 32 model-training
episodes across the two arms and RNGs. Paired methods share the same environment
seed within each RNG; the two RNGs use disjoint seed blocks.

Candidate seed blocks, based on a read-only seed audit on 2026-10-25:

- RNG 0 train: 8465–8472
- RNG 1 train: 8473–8480
- Development: 8417–8432, each seed evaluated under both Discovery and Finite
  (16 paired seeds per Red regime)

The ledger currently reports these blocks free. Re-run the authoritative audit
immediately before freezing; honor any intervening reservation. Do not use
8481–8500 or 8501+ reserved blocks. Preserve seed separation from all training
and prior comparisons.

Evaluate every trained checkpoint on the same development seeds and both Red
regimes. Run each reference policy once per seed/regime where deterministic;
verify this reuse is exact before reducing duplicate control episodes. Save
raw per-episode JSON, router traces, model/source hashes, actual seeds, and a
manifest. No checkpoint selection by dev return: use the final fixed iteration.

## Measures and decision rules

Primary outcome: paired native team-return difference, value-aware router minus
strict Lancer, reported separately for Discovery and Finite, then macro-averaged
across regimes. Also report mean, median, paired win fraction, bootstrap 95%
interval, and 10th-percentile / worst-quartile return. Report both training RNGs
separately and pooled descriptively.

Diagnostics: action coverage, verified remediation, time to first detection,
free/forced decision counts, branch selection frequency, route entropy and
switch rate, critic error by branch, and ensemble disagreement calibration.
Native reward is primary; privileged compromise/service labels stay evaluation
only.

**Screening advance rule:** proceed to confirmation only if (a) pooled paired
mean improvement over strict Lancer is positive in both Red regimes, (b) the
paired bootstrap interval for the macro-average excludes zero, (c) neither
regime's 10th-percentile return is materially worse than Lancer, and (d) both
RNGs show a positive macro-average difference. Otherwise record a negative or
inconclusive screen and do not open a confirmation seed block. These small
screening runs cannot establish a final RL win.

If screening passes, preregister a fresh confirmation with at least eight
training RNGs and an independently audited evaluation block. Do not tune
`kappa`, architecture, duration, or reward on the confirmation results.

## Runtime estimate and stop conditions

Use the active hybrid run's measured simulator/evaluation throughput to set a
wallclock budget before launch. The candidate screening includes 32 training
episodes and at most 288 evaluation episodes if all four model/RNG cells run
both untrained and trained policies on all 32 seed/regime pairs plus 32 shared
Lancer episodes. At the current observed rate of roughly 25 seconds per
episode, this is about 2 hours for evaluation plus training and analysis; refine
the estimate using the completed active-run logs. Run detached/restartable,
persist each completed seed, and do not interrupt the live hybrid run to start
this experiment.

Stop and preserve partial artifacts on: a source/config hash mismatch, invalid
action mask, missing/invalid native reward, critic non-finite values, a lost
checkpoint, repeated WSL/runtime restart without resumable logs, or exhausted
predeclared compute budget. Partial data remains descriptive and is never
presented as a completed comparison.

## Before freezing

1. Finish and analyze `architecture_hybrid_long_v1`; record both RNGs and both
   Red regimes, including the already identified actual evaluation count.
2. Implement the router as a separate actor/runner version; leave the current
   run and its frozen code untouched.
3. Unit-test mixture log-probability/gradients against a small enumerated
   categorical example; test action masking, zero-residual parity, deterministic
   reset, and route logging.
4. Test critic targets on a tiny known MDP and measure ensemble calibration on
   held-out episodes before training.
5. Audit candidate seeds again, freeze all source hashes/configuration, then
   launch screening. Do not start confirmation without the screening rule.

## Update after the situation-routed hybrid run (2026-10-25)

The two-RNG hybrid run completed. There were 16 evaluation seeds per Red regime
per RNG (the frozen hybrid proposal said 8 per regime; the runner used the full
16-seed list for both regimes). The result suggests a smaller but fairly
consistent training gain over the untrained stochastic hybrid, alongside much
larger and more outlier-sensitive gains over strict Lancer:

| RNG / Red | trained − Lancer mean | trained − untrained mean | trained − untrained median / wins |
|---|---:|---:|---:|
| 0 / Discovery | +11.4 | +12.1 | +6.5 / 10 of 16 |
| 1 / Discovery | +1.6 | +8.9 | +2.5 / 8 of 16 |
| 0 / Finite | +31.6 | +14.6 | +9.5 / 10 of 16 |
| 1 / Finite | +39.9 | +9.2 | +1.0 / 9 of 16 |

The 64 seed/RNG/regime rows pool two trained RNGs only; they are not 64
independent training runs. Across those rows trained-minus-untrained is mean
+11.2, median +5.5, IQM +6.7, and wins 37/64. A conspicuous Finite case
(RNG 1, seed 8407) was -543 for Lancer, -128 untrained, and -127 trained; its
large Lancer-relative improvement was almost entirely present before learning.

Most directly, gate capture does not yet support the hypothesis of reliable
situation-specific routing. Mean attention weight was approximately 0.60 for
RNG 0 and 0.08 for RNG 1, while each RNG's mean was almost identical under
Discovery and Finite. This may be seed-dependent branch preference or collapse;
it is not by itself proof of collapse because branch outputs may be redundant.
The next experiment should therefore be a small, equal-budget gate ablation:

1. SetActor PPO.
2. Fixed 0.5 blend of SetActor and attention PPO.
3. Learned situation gate PPO.

Use the same strict Lancer and untrained controls, environment seeds, optimizer,
and iteration budget. Require both (a) learned-minus-untrained gains and
(b) a difference against the fixed-blend arm before crediting the learned gate.
Log branch score disagreement and gate values by episode/context; test whether
gate selection changes by Red-relevant observable context, not just by training
RNG. Do not force equal gate usage with a balance penalty unless diagnostics show
an avoidable collapse; indiscriminate balancing could force use of a worse
branch.

Only if the learned gate beats the fixed blend consistently should the larger
value-aware router proceed. For that later mixture, compute the marginal
categorical action probability exactly as
`pi(a|s) = sum_k g_k(s) pi_k(a|s)` and use its log probability in PPO where
feasible, instead of sampling a hidden expert and using that single sample's
gradient. The candidate action set is small enough to enumerate. Czarnecki et
al.'s Mix & Match analysis distinguishes explicit policy mixing from sampling
one latent policy: the latter is a Monte Carlo estimate with higher variance,
and their experiments use explicit mixing for variance reduction. This is
supporting evidence from a different RL setup, not a proven guarantee for this
Blue runner. [Czarnecki et al., ICML 2018](https://proceedings.mlr.press/v80/czarnecki18a/czarnecki18a.pdf)
