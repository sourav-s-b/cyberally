# Proposal: blue-eval-seed-block-7809-8200

- Status: GRANTED 2026-10-05 by project owner for Blue evaluation use
  (recorded by Blue; Evaluation/Environment to countersign in ledger)
- Author role and branch: Blue, `blue/metric-repair-maxage` (commits
  `8613bac` P0, `701e23f` MAPPO build)
- Producer and affected consumers: Producer Blue (pilot/eval manifests);
  consumers Evaluation (frozen-evaluation seed bookkeeping, held-out suite
  separation) and Environment (seed ledger). Red: none (block untouched).
- Related issue/PR and exact dependency commits: extends
  `blue-training-seeds.md` (implemented `cc9a5c2`); sizing method from
  `blue-handoff-20261004.md` §4.3; guard-matrix provenance
  `guard-maxage-matrix-20261004.json` + committed cells
  `guard-maxage-matrix-20261004-cells.jsonl`.

## Problem and current behavior

The frozen evaluation for the MAPPO-guide pilot must be sized from measured
variance (`n = ceil((2.80·s_Δ/(δ−δ₀))²)`), which at the provisional
`s_Δ = 18.7` needs up to ~110 fresh evaluation seeds. No free block is
currently assigned to Blue evaluation, and training/evaluation seeds must
stay separated (P4 pilot trains on dev seeds only).

Seed ledger today (verified from manifests):

| Block | State |
|---|---|
| 7629–7729 | consumed (dev: guard matrix + pilots) |
| 7801–7808 | consumed |
| 7809–8200 | **apparently unclaimed in every role ledger** |
| 8201–8216 | consumed |
| 8301–8400 | finite Red (do not touch) |
| 8401+ | ledger conflict (do not touch) |
| 8501+ | reserved / untouchable |

## Proposed contract change

Reserve **7809–8200 for Blue evaluation only**, additive:

- Before: no assigned Blue evaluation block; eval seeds picked ad hoc.
- After: `7809–8200` = Blue frozen-evaluation pool. Rules preserved:
  **no training on this block** (training stays on dev seeds), other blocks
  untouched, Evaluation keeps held-out/final suites disjoint from it.
- Ownership: shared seed allocation is a shared contract, so this needs
  affected-role review, not a unilateral grab. P0–P4 need no new seeds and
  are not blocked; only the frozen evaluation (P5) is gated on this.

## Required branch changes

| Role | Files/behavior to update | Dependency | Acceptance test |
|---|---|---|---|
| Blue | eval manifests record `eval_seeds` subset of 7809–8200; `freeze_eval(seed_list, n)` refuses training-block seeds | this lands | eval seed list disjoint from train cycle in manifest |
| Evaluation | frozen-eval bookkeeping reads the block; keeps final suites disjoint | this lands | fixture: no overlap between block, train seeds, final suites |
| Environment | ledger records the reservation | this lands | ledger lists 7809–8200 as Blue-eval |
| Red | none | — | confirm none exist |

## Merge and migration order

1. This lands as a documentation reservation; no code behavior changes.
2. Blue's frozen eval consumes a frozen prefix; Evaluation/Environment adopt
   the ledger entry.
3. If the block is not granted, Blue runs at reduced n, applies the §4.2
   decision rules to whatever interval results, and reports achieved power.
   A null at reduced n is not relabeled as absent headroom.

## Validation and decision

- GRANTED 2026-10-05 by project owner: 7809–8200 reserved for Blue
  evaluation only. No training on this block; other blocks untouched.
  Frozen evaluations consume a frozen prefix recorded in the eval manifest.
- No code changed; `git diff --check` clean.
