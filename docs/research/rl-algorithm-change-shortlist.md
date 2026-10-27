# RL algorithm change: evidence-linked shortlist

Date: 2026-10-06. Author: Blue role. Branch `blue/metric-repair-maxage`.

**Purpose.** The risk-input ablation was inconclusive because the learned
policy *under-remediates*: on dev seed 8241 the actor issued 23 Remove and
28 Restore against Lancer's 33 and 64, and accrued per-tick penalties of
−40 where Lancer never went below −3. Damage is late-episode and
compounding. This is a credit-assignment and policy-improvement problem, not
an ML-feature problem, so the response is to change how the policy learns —
not to swap the risk model.

**What we already tried** (do not repeat): BC (+ many variants), IQL, CQL,
AWR/AWAC, RvS, PPO/MAPPO, SAC/TD3, DQN, RND, PLR, COMA, IMPALA, and the
hybrid/priority/recency scorers. Full list in
`docs/research/blue-rl-recovery-literature.md`; failure mapping in
`docs/research/blue-sparse-reward-literature.md`.

## The measured failure, restated precisely

From `blue/training/collapse_probe.py` on dev seed 8241, 400 steps:

| | Lancer | actor |
|---|---|---|
| return | −64 | −1273 |
| Remove / Restore | 33 / 64 | 23 / 28 |
| Analyse | 790 | 895 |
| worst per-tick reward | −3 | **−40** |
| ticks with negative reward | 60 / 399 | **153 / 399** |

First ~200 ticks are indistinguishable; divergence then compounds to −10 per
tick. The actor is *cautious*, not reckless: it spends its budget analysing
and under-remediates, and pays a large persistent penalty for the rest of
the episode.

Three consequences drive the shortlist:

1. **The useful signal arrives late** (a missed remediation bills you every
   subsequent tick), so credit must travel backwards from episode end.
2. **The baseline is already decent.** Any method must not be able to make
   things arbitrarily worse, which is what a free-running residual actor
   currently can.
3. **Budget is tiny** (4 iterations, ~800 rows). Sample-hungry methods are
   ruled out regardless of merit.

## Shortlist

### 1. Advantage-filtered / expert-anchored updates — recommended

**Literature.** Advantage-Filtered BC (`arXiv:2110.04698`) applies a binary
or exponential filter `f(Â)` to imitation loss so samples the critic
believes are *worse* than current policy are dropped. Residual Policy
Learning (`arXiv:1812.06298`) frames a baseline as a corrective term.
Nair et al. supply the expert action whenever its value exceeds the agent's.

**Why it fits.** This is the direct fix for "the actor overrides Lancer on
half of all decisions." Instead of every sampled override becoming gradient,
only overrides the critic predicts are *better* than Lancer's action
survive. It is cheap (we already have a central critic), it is a training-
time filter so eval policy is unchanged, and it structurally bounds the
override rate — which is precisely the quantity that went wrong.

**Honest risk.** Filtering by critic estimate is only as good as the critic.
With ~800 rows and 4 iterations the critic is weak, so this could filter
correctly on almost nothing. It is cheap to find out.

### 2. RUDDER — return decomposition

**Literature.** `arXiv:1806.07857` (NeurIPS 2019). An LSTM predicts the
episode return from the whole state-action sequence; per-timestep
contributions become redistributed rewards.

**Why it fits.** It is the canonical answer to "the penalty arrives late."
Our −40/tick tail is exactly the delayed-reward structure RUDDER targets,
and it reports large speedups under delay.

**Honest risk.** Needs many complete episodes to fit the LSTM, and we have
very few. The paper notes it is ineffective without genuine delay and can
introduce spurious signals. Likely too data-hungry for our budget, but it is
the textbook fix and worth a feasibility note rather than a silent omission.

### 3. Potential-based reward shaping

**Literature.** Ng, Harada & Russell, ICML 1999. Shaping is policy-invariant
**iff** potential-based, `F = γΦ(s') − Φ(s)`; non-potential shaping can
change the optimal policy.

**Why it fits.** Our own F3 write-up records that we violated exactly this
with ad-hoc `{clear +3, confirm +1, vandalism −3}` bonuses and got bonus
farming. If we touch the reward at all to fix remediation timing, this is the
only defensible form.

**Honest risk.** Not an algorithm change; it is a prerequisite for any
reward edit. Also needs Evaluation review per AGENTS.md, since it alters the
scored objective.

### 4. TAR² — temporal-agent reward redistribution (lower priority)

`arXiv:2502.04864`. Joint temporal + agent credit with potential-based
shaping; beats AREL and STAS on SMACLite/GRF.

**Why lower.** It addresses *agent* credit (which Blue host to fix). Our
failure is temporal, not agent-attribution — one shared team signal, so
per-agent decomposition is less urgent than fixing the time axis.

## Explicitly not recommended

- **Anything off-policy or replay-based** (SAC/TD3/CQL/IQL variants). Ruled
  out by the budget; sample-hungry and our buffer is one iteration wide.
- **Switching the ML risk model (gradient boosting etc.).** Premature. The
  ablation could not separate the arms because the policy failure dominates.
  Improving a feature the policy currently misuses is not the bottleneck.
- **GNN / attention architectures.** No representation evidence that features
  are the constraint; adds cost we cannot afford at this budget.

## Recommended order

1. Instrument `agree` rate and override-only advantage as **tracked
   metrics** — currently invisible, which is why this took a collapse to
   find. Cheap, no training.
2. Advantage-filtered updates (shortlist 1). ~1 day, existing critic.
3. Measure override rate and remediation counts against Lancer.
4. Only then consider reward-side changes (shortlist 3), with Evaluation
   review.

## Success criteria, pre-registered

Stated before running, because we have a history of reading noise as signal.

- **Primary:** mean paired return vs Lancer on ≥32 fresh dev seeds, with the
  unit of replication being the *training seed* (n≥3), t-interval on df=n−1.
  Eval seeds are shared across training runs and are a blocked factor.
- **Secondary (the actual defect):** Restore/Remove counts must not fall
  below Lancer's, and worst-case per-tick reward must not exceed Lancer's by
  more than 2×. A policy that wins on mean while under-remediating is a
  failure, not a success.
- **Report every run.** No best-of selection.
- Reserved seeds 7809–8200 remain untouched.

## Open question for Evaluation

Redistributing or shaping the reward changes the scored objective. Under
AGENTS.md that needs a proposal in `docs/coordination/` with Evaluation
sign-off, not a unilateral Blue change. Shortlist 1 needs no such approval
because the reward and the eval protocol are untouched.