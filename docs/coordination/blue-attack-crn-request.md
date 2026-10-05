# Proposal: blue-attack-crn-request (requested and recommended against)

- Status: draft (kept on record as a rejected-for-now trade-off, not a
  silent omission)
- Author role and branch: Blue, `blue/metric-repair-maxage`
- Producer and affected consumers: Producer Environment (simulator RNG /
  scenario coupling) with attacker semantics owned by Red; consumer Blue
  (variance reduction for paired contrasts). Evaluation: informed.
- Related issue/PR and exact dependency commits: motivation is handoff
  §2.2(d), recomputed 2026-10-04/05 from
  `guard-maxage-matrix-20261004-cells.jsonl` (256 cells, committed).

## Problem and current behavior

Blue's attack schedule couples to Blue's actions: initial compromise is
shared (`n_initially_compromised_episodes` identical across all 8 arms in
32/32 seeds) but attack evolution is not (`n_compromise_episodes` differs
in 32/32 seeds, mean spread 28.31 events; compromised host sets differ in
224/224 arm-seed comparisons). Mechanism:
`FiniteStateRedAgent._choose_host` selects by host-state priority, so Blue
remediation changes which hosts Red attacks next. This coupling inflates
the paired SD of return contrasts (observed 14–31 across arms).

## Proposed contract change

Attack-level common random numbers: an Environment-owned option to hold
the compromise scenario (onset schedule/hosts) fixed across Blue arms on
the same seed, with Red owning the attacker-semantics review.

- Before: no coupling control; seed count is the only variance lever.
- After (if accepted): opt-in CRN mode for paired Blue contrasts, default
  off, main runnable unchanged, no mask/reward/pending-action change.

## Required branch changes

| Role | Files/behavior to update | Dependency | Acceptance test |
|---|---|---|---|
| Environment | CRN scenario-coupling option in simulator setup | Red review | same-seed compromise sets identical across Blue arms with CRN on |
| Red | attacker-semantics review: CRN must not make Red non-adaptive in a way that misrepresents the threat | Environment draft | documented threat-fidelity statement |
| Blue | eval manifest records CRN on/off per contrast | both above | paired-SD comparison CRN on vs off |
| Evaluation | none unless adopted for suites | — | — |

## Merge and migration order

Not proposed for the current pilot. If ever revived: Environment drafts
behind a default-off flag → Red reviews threat fidelity → Blue pilots one
paired contrast → results decide adoption. No removal step.

## Validation and decision

**Recommendation: do not build this now**, on cost and uncertain benefit:

- Observed `r(return_diff, Δcompromise)` is +0.003 (maxage24), -0.265
  (strict), -0.300 (maxage48), -0.365 (maxage96), -0.188 (strict_oracle),
  -0.488 (maxage48_oracle), -0.381 (unguarded_oracle): r² 0.00–0.24.
  These describe 32 observations and do **not** bound what a different
  coupling design would buy; valid CUPED covariates explain ≤3.5% of SD.
- Simulator change owned by Environment + attacker semantics owned by Red
  for an uncertain variance gain, while seed count (this proposal's
  companion: 7809–8200) is available without any simulator change.

Kept on record so a future trade-off discussion has numbers instead of a
silent omission. Affected-role review: pending; do not mark accepted just
because this file exists.
