# Blue update — 2026-10-05

Branch `blue/metric-repair-maxage` (25 commits ahead of
`origin/blue/mappo-training` at `2900711`). Sim venv 40 targeted passed,
train venv 58 passed. One real bug fixed this session
(`eval_parallel.py` had a broken import committed in `bb0b724`).

## Headline

**RL is still not a demonstrated gain** — Lancer remains the defender.
But for the first time a learned policy is *above* the heuristic on
average, and the mechanism is understood.

| variant (8 dev seeds, paired vs Lancer) | mean | 95% CI | wins | worst |
|---|---:|---|---:|---:|
| risk feature, train on all decisions | -15.75 | [-76.1, +44.6] | 5/8 | -188 |
| **risk feature, train on informative decisions only** | **+4.50** | [-22.9, +31.9] | 5/8 | **-44** |
| + proba-0.5 eval gate | -1.88 | [-14.4, +10.7] | 4/8 | -37 |

Pre-registered gate is mean >= +5 with the CI excluding zero. +4.50 does
not clear it, so **no claim** and no fresh-seed confirmation. Status:
promising lead under test.

## What produced the improvement

The deck's Stage-2 featurizer (learned P(compromised), AUC 0.68,
well calibrated: 83% precise at proba 0.8+, 30% real in the 0.4-0.6 band)
is useless as a *decision rule* — it collapsed coverage in2026-10-02 and
finished negative in every form tested. It is useful as a *training
signal*: recording a PPO decision only when the pick's proba >= 0.5 keeps
the agent from learning on the ~70% of decisions that are noise. KL rose
10x (0.015-0.020 vs 0.001-0.003), the worst seed went -188 -> -44, and
the mean moved +20.2.

## Everything tried this session, with verdicts

| experiment | result | verdict |
|---|---|---|
| 32-seed heuristic screen (5 orderings) | Lancer best; 3 rivals identical 32/32 | no headroom |
| detection-headroom mining | every arm misses ~35% identically | no headroom |
| response-rule ablations | no_restore catastrophic; no_verify -24.4 | rules load-bearing |
| agent-4 suspicion skill | +23.75 (8 seeds) -> +0.44 (32 seeds) | failed confirmation |
| mode-vs-regime matrix | no mode clears gate in any regime | router unjustified |
| MAPPO pilots 1-2 | -122.4 / -81.4 vs Lancer | frozen as failed |
| shield (margin 0.5-1.0) | +0.00, identical to Lancer | safe fallback works |
| risk-as-feature (eval only) | -15.75 | fails |
| risk-as-training-signal | **+4.50** | best so far, fails gate |

## Why RL kept failing (two measured causes)

1. **Sparse reward** — only 41% of steps carry nonzero native reward.
   Shaping exists (`SHAPING_DEFAULTS`) and is still unused.
2. **No allocation trade-off** — coverage is 1.000 under every policy
   despite 53% forced-lockout Sleep, so there is nothing to *allocate*.

Plus one structural fact: containment on suspicion is impossible in this
simulator (Remove/Restore legal only on CONFIRMED/VERIFY hosts).

## Corrections made to my own earlier claims

- Two diagnostic bugs caught in review: an always-False agreement counter
  (compared agent id to host string) and a proba average over all
  candidates rather than the chosen hosts. Both claims withdrawn.
- "No RL was ever run" — withdrawn. MAPPO, residual PPO and IQL all ran;
  only the FQE evaluator was invalid. IQL's null came from a
  non-canonical implementation.
- `+23.75` was an 8-seed artifact that failed at 32 seeds.
- The host universe is seed-dependent (57-96), not a fixed 66.

## Untested / next

1. **Train longer** — only 4 iterations; the curve was still improving and
   updates are now 10x larger. ~20 min. The fairest remaining shot at +5.
2. **Shaped reward** — implemented, tested, never used in training.
3. **Capacity constraint** — makes allocation real, but synthetic
   (Blue-imposed), so any gain is a coordination claim, not a
   threat-model claim.
4. RL-gain claim needs a fresh-seed confirmation before the reserved block
   (7809-8200) is spent.

## Honest status for the deck

Keep: Lancer backbone, MARL prototype, retraining narrative.
Narrow: any claim that RL improves results.
Label planned: LLM Red, eBPF telemetry, SIEM audit, sub-5s response.