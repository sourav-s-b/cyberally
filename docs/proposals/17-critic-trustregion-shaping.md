# Proposal 17: Critic audit → trust region → safe densification

- Status: proposed (P4/P5; audit runs parallel with 14/16)
- Date: 2026-10-02

## Description

Three layered bets on WHY learning fails, cheapest first: (a) the
divergence may be critic-side, not policy-side — audit critic choice per
Lyu et al. before any new algorithm; (b) HAPPO sequential trust-region
updates replace simultaneous MAPPO; (c) training-time-only densification
that is optimality-preserving BY CONSTRUCTION (dynamic PBRS) plus
episodic exploration (E3B/NovelD). Eval stays native-only throughout.

## Motivation / evidence

(a) Lyu et al. (E21–E23) prove state-only central critics bias gradients
under partial observability AND inflate actor variance — that is our
1190-dim critic (3.8 → 61.8) with a published mechanism attached. The
DAAC lesson (H28) adds a concrete audit item: if actor/critic share an
encoder, value overfitting inflates regression scores while generalizing
nowhere. A history-state or decentralized critic is days of config work;
HAPPO is weeks. Audit first: if a history critic stabilizes flat MAPPO,
the algorithm was never the problem.

(b) HAPPO/HATRPO (E13/E14) replace the simultaneous updates behind our
divergence with sequential per-agent trust regions (monotonic guarantee,
heterogeneous-agent variants for our unequal zones). Deploy only if the
audit leaves divergence unexplained.

(c) Shaping, done honestly this time. 09 tested ARBITRARY shaping
(clear=3/confirm=1/vandalism=-3, no gain) and the native-only rule stands.
Dynamic PBRS (B22 over B21) is a different claim: densification of the
form F = γΦ(s′,t′) − Φ(s,t) provably preserves the optimal policy, so the
proof obligation is explicit (exhibit Φ). Concrete Φ candidate: Bates 2025
(A25) — sparse healthy-host bonus beats dense scaffolds. Exploration: 07
rejected PREDICTION-ERROR curiosity (noisy-TV trap) with a revisit
condition (stealth/discovery bottleneck); E3B/NovelD are a different
family — episodic/count-based, stochastic-robust — and the revisit
condition is now arguably met (learned policies show coverage collapse,
finite red punishes poor sweeping). Frame as a new test, not relitigation.

## Experiment

1. Audit (days, parallel with 14): history-state critic; decentralized
   critic ablation; shared-vs-split encoder check. Kill criterion: audit
   changes nothing → critic is exonerated, proceed to HAPPO with a clear
   conscience.
2. HAPPO (2–3 weeks, conditional): sequential updates on the factorized
   actor; KKT KL allocation (E25) if zones improve unevenly.
3. Densification (1–2 weeks, parallel): dynamic-PBRS healthy-bonus Φ +
   E3B episodic bonus + NovelD boundary bonus, TRAINING-TIME ONLY, native
   reward the sole gate metric. Each bonus ablated separately — 09 taught
   us bundles prove nothing.
4. Gate (binding): stabilization bar = a non-diverged RL run (ties
   teacher); win bar = beats lancer_v2 held-out (-91.2). Shaping that
   improves training curves but not native held-out is REJECTED (09 rule).

## Pros

- The audit is the cheapest experiment on the board that could explain
  W1; it runs in parallel with 14/16 without blocking them.
- PBRS carries a proof instead of a hope — the first shaping proposal
  compatible with the native-only rule by construction.
- E3B/NovelD directly target W3 fixation (27-vs-67-host collapse) with
  mechanisms 07's rejected curiosity lacked.

## Cons

- If the audit exonerates the critic AND HAPPO still diverges, W1's cause
  is deeper (plasticity → 16-D0, or data → 14-verdict). This proposal
  spends weeks to possibly redirect, not to win.
- HAPPO sequential updates cost 5× actor passes per epoch; budget wall-
  clock honestly against pool throughput (~500 s/8 cells baselines).
- Intrinsic bonuses add hyperparameters (E3B covariance, NovelD gating)
  to an already noisy ±313-band regime — demand ≥8-seed ablations per
  bonus or drop it.

## Verdict

PROPOSED, P4/P5. Audit immediately (parallel); HAPPO conditional on
audit; densification as a parallel track with per-bonus ablations and the
09 rule enforced.
