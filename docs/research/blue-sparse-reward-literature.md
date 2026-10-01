# Literature alignment: why Blue PPO degrades our BC policy

Status: research note, 2026-10-01. Author: Blue role. Sources are cited by
name/DOI/arXiv id so the claims are checkable; nothing here is measured by
us. Our own measurements live in `docs/status/blue-session.md`.

**Two measured follow-ups that change the conclusions below** (details in
`docs/status/blue-session.md`):

- We implemented the F1 fix (critic-only warmup) and it did **not** help.
- The evaluation noise floor is larger than the effects we have been
  ranking. So F1 is not, on our data, a demonstrated cause of our losses.

The literature mapping still stands as a map of the field; treat our own
causal claims as unproven.

Scope: three failures we observed, mapped to the published literature that
names and fixes them. Conclusion up front — **all three are known failure
modes, two of our mitigations were weaker than the published standard, and
one of our shaping choices was a textbook violation.**

## F1. PPO fine-tuning degraded the BC policy in 4/4 runs

Measurements: `docs/status/blue-session.md` (4 fine-tunes from one BC init,
all monotonically worse in native return; monotone in lr).

| Phenomenon | Reference | Published fix | What we did |
|---|---|---|---|
| Forgetting of Pre-trained Capabilities (FPC) — "state coverage gap" and "imperfect cloning gap"; "often catastrophic" | arXiv 2402.02868 (Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem) | Continual-learning retention: BC regularisation, EWC, knowledge sourcing. Explicitly: "For NetHack and **imperfect cloning gap** case, **where the agent is initialized to closely mimic the expert**, it might be sufficient to prevent forgetting on states visited online by the fine-tuned policy" | Only lr reduction |
| Fine-tuning is stability–plasticity; stability-anchor regime selection | arXiv/blog twni2016 (RL Fine-Tuning Is a Stability–Plasticity Problem), 3780 fine-tuning runs, 21 datasets | Regime test on J(π0) vs J(dataset). In the "superior" regime (our case: BC is better than anything RL finds) use **π0-centric stability**. Ranked: strong = freeze backbone / adapters / residual policies / **KL to π0** / **online data warm-up with π0**; **weak = small learning rate** | Small learning rate — the *weakest* item on their list |
| Value-function divergence at onset of fine-tuning causes unlearning without offline data retention | WSRL, ICLR 2025 (arXiv 2412.07762) | **Warm-up phase: K steps of rollouts from the FROZEN pretrained policy** to recalibrate Q before any update (K=5000) | No warm-up at all; critic randomly initialised |
| Pretraining a Q-function on a narrow policy is actively harmful; Q collapses toward V | arXiv (Do You Really Need to Pretrain Q-Functions for Online RL Fine-Tuning?) | IPE: train N policy variants on the same data, roll out each, so the critic sees action coverage. "in several tasks it actually hurts online learning relative to fine-tuning on top of a **randomly-initialized** Q-function" | Random critic + single narrow BC policy = the exact bad case |
| Forgetting is predicted by KL divergence between base and fine-tuned policy on the new task | RL's Razor, arXiv 2509.04259 | Measure KL(π0, π_ft) as a diagnostic; on-policy updates are KL-minimal by construction | Not measured |
| On-policy data is what prevents forgetting (not KL penalties, not advantage estimation) | Retaining by Doing, arXiv 2510.18874 | ~on-policy data suffices; SFT on initial-policy data alone is not enough | N/A (we are on-policy) |

**Actionable:** F1 is a *diagnosable, fixable* problem, not a wall. The
strongest published fixes we have not tried: (a) frozen-policy online
warm-up before any gradient step (WSRL / stability-plasticity #1 result);
(b) explicit BC/EWC retention term during fine-tuning (FPC, and the paper
names our exact regime); (c) KL(π0,π) as a standing diagnostic.

## F2. Unweighted BC collapsed to the majority action (always Sleep)

| Phenomenon | Reference | Published fix | What we did |
|---|---|---|---|
| Imbalanced demonstrations → policies biased to the most represented behaviour; proven formally for BC | arXiv 2508.06319 / Springer Autonomous Robots, "Towards balanced behaviour cloning from imbalanced datasets" | Reweight or resample. Unbiased-parameter sampling is `q(s,a) = p(s,a)/ρ_i` (**divide** by sub-policy frequency). Strongest variant is meta-gradient rebalancing | **Inverse-sqrt** class weights + zeroing loss on busy ticks |
| Re-sampling alone is ineffective for long-tail; tail tasks fail ~4x more | Beyond the Majority (long-tail IL for manipulation) | Transfer from head to tail (Approaching-Phase Augmentation) | n/a |
| Mode collapse from class imbalance in BC, same failure mode | Discriminator-weighted BC, arXiv 2510.01479; Zuo 2024 | Density-ratio / discriminator weighting against a vetted clean reference set | n/a |

**Actionable:** our BC fix was directionally right (it took non-sleep
accuracy 0.3% → 12%) but is a weaker weighting than the `p/ρ` formulation
in the paper. Revisitable if BC fidelity ever becomes the bottleneck. Note
the ceiling: the round-robin sweep cursor is unobservable, so exact-id
fidelity is capped and only *semantic* fidelity is reachable.

## F3. Shaped rewards made native return worse (bonus farming)

| Phenomenon | Reference | Published standard | What we did |
|---|---|---|---|
| **Non-potential-based shaping can change the optimal policy.** F is policy-invariant **iff** it is potential-based: `F(s,a,s') = γΦ(s') − Φ(s)` (sufficiency *and* necessity). The paper's own examples are bonus-farming agents (bicycle circles; soccer "vibrating" next to the ball) | **Ng, Harada & Russell, ICML 1999**, "Policy invariance under reward transformations" | Use potential-based shaping only. Anything else risks a suboptimal policy for the original MDP | **Violation.** Ad-hoc {clear +3, confirm +1, vandalism −3} bonuses are not potential-based, so policy invariance was not guaranteed — bonus farming was a predictable outcome, not bad luck |
| Proxy reward optimisation should be an auxiliary, not the objective | "Defining and Characterizing Reward Hacking", NeurIPS 2022 | Reward functions used as *auxiliaries to policy learning*, not as specifications to optimise | We optimised the shaped reward directly and graded natively — the grading discipline was right, the shaping was unsound |
| Structural (not regression-based) policy preservation for credit assignment | TAR², arXiv 2502.04864 (SMACLite + GRF) | Decouple credit modelling from normalisation so the guarantee holds "regardless of model accuracy"; explicitly criticises STAS/AREL as theoretically brittle | Not attempted |
| Hand-tuned reward terms mislead and can be a poisoning vector | CG-MARL, arXiv 2208.03002 | "reward-tuning is likely to introduce biases that mislead agents to learn unexpected cooperative behaviors… valuable to reward poisoning" | Confirmed empirically: shaped training return −435 → −68 while native got worse |

**Actionable:** F3 is our own methodological error, with a textbook fix.
If shaping is revisited, it must be potential-based, and we should verify
invariance rather than hope for it.

## F4. The root cause: sparse/delayed credit, and MARL makes it worse

Our MAPPO is stock EPyMARL (QMIX-family assumption: a **dense per-timestep
team reward**). TAR² states that assumption "fails in the challenging
episodic settings we address". CG-MARL defines the distinction sharply:
single-agent RL decomposes reward along *time*; MARL must decompose along
*agents* **first**, and only then along time — they call this 2nd-order vs
1st-order sparsity. Five Blue agents share one team signal whose payoff for
a correct Remove lands hundreds of ticks later, so we are on the hard side
of that gap.

Relevant lines of work, none of which is hand-shaped bonuses:

- **Reward redistribution** instead of shaping: RUDDER, STAS (Shapley +
  temporal attention, AAAI'24), AREL / Agent-Temporal Attention (AAMAS'22),
  ATA (AAMAS'22), TAR².
- **Influence-scope credit + focused exploration**, no prior knowledge or
  dense feedback required: ISA, arXiv 2505.08630 — reports classical MADRL
  "fails to learn" under sparse reward and beats HMASD by 56% on 2s_vs_1sc.
- **Hybrid neuro-symbolic**, the closest match to our pivot: CG-MARL
  reduces 2nd-order to 1st-order sparsity by injecting "fundamental
  cooperation knowledge" as a fixed interface, and learns the rest.

## Net conclusions for Blue

1. ~~F1 is fixable by a frozen-policy warm-up.~~ **Tried; no effect on our
   data.** The intervention is implemented, tested and off by default, but
   the 4000-step warmup scored -455 vs -393 without it, and behavioural
   KL(π0, π_ft) is only ~0.001 nats, so the two policies are nearly the
   same function. Forgetting is not what is costing us points.
2. **Our real problem is measurement, before it is optimisation.** With
   std 277 across seeds, every policy delta we have ranked is inside the
   noise. Fix the eval protocol (>=8 seeds, mean ± std) before running more
   experiments; further A/B on 3 seeds is wasted compute.
3. F3 is a genuine method error regardless of the noise: the shaping was
   not potential-based, so policy invariance was never guaranteed. Drop it
   or make it potential-based. (The size of its measured harm is subject to
   point 2.)
4. The capability gap is the thing worth attacking, and it is a
   *representation* gap: the round-robin sweep cursor is not in the
   observation, so no policy fine-tune can recover it. This is the case for
   the neuro-symbolic route, and CG-MARL is the closest precedent — fix the
   cooperation skeleton, learn the rest.
5. Keep the KL(π0, π_ft) measurement in the eval harness regardless. It is
   cheap and it is the quantity that would have caught a genuine forgetting
   event; its near-zero value here is itself the finding.
