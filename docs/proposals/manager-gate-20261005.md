# Pre-registered gate: learned Blue manager vs deployed Lancer

- Status: pre-registered 2026-10-05, no run attached. Any future claim of
  "at least five points better" must satisfy this gate on a frozen
  evaluation, or the claim is not made.
- Pool: `scenario-pool-v1` (old / new / mixed / switching splits defined
  there). Seeds: frozen prefix of 7809–8200 disjoint from consumed
  sub-ranges, recorded in the eval manifest. No training on the block.

## Primary comparison

Learned manager (over the identical skill set, Lancer fallback armed) vs
**deployed unguarded Lancer** — never vs a guarded or weaker baseline.

## Metrics (per-scenario tables, never pooled-mean-only)

Return, compromise duration, detection delay (onset-aware Step-2
definitions), recovery time, false-positive recovery, action count,
manager/shield intervention rate.

## Decision rules (evaluate in this order, from the handoff §4.2)

1. Regression — upper 95% bound < 0 → report evidence of regression.
2. Advance — unadjusted paired mean ≥ +5, 95% CI excludes zero, no failure
   check tripped → replication only, not deployment.
3. Minimum worthwhile gain unsupported — upper bound < +5 → state it.
4. Inconclusive — otherwise. Stop at the predefined compute budget. Do not
   enlarge n until an interval passes. Final evaluation stays separate from
   checkpoint selection.

An "at least five points better" claim additionally requires the lower
confidence bound above +5 after replication.

## Stop conditions

Fixed budget exhausted without clearing the gate, or manager only ties
Lancer → keep Lancer fallback, publish the limits. A documented limit is
a complete result. Training runs that predate the Step-3 headroom finding
(mappo_guide1/2, resid-pilot1, IQL, EPyMARL fine-tunes) cannot be
re-attached to this gate retroactively.
