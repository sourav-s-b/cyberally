# Blue event timeline (reconstructed 2026-10-05)

`blue-session.md` is concatenated from several sessions: do not infer
experiment order from line order here. This file reconstructs order from
`git log` timestamps and manifest `created` fields. Times +0530.

## 2026-10-02 — factorized/distill, IQL/RvS, red variants, eval baselines
Manifests dated 2026-10-02 (times not all recorded; day-level only):
`fact-*` (distill ladder, attn-1 parity), `iql-*` (offline RL, nulls from
non-canonical IQL), `rvs-*`, `kl-fact-*` (KL-anchored MAPPO probes),
`red-{discovery,finite,random,sleep,verbose}` (finite much harsher),
`lancer-*`, `parity-pool`, `risk*`. MAPPO fine-tunes diverged; line closed.

## 2026-10-03 — research audit, drift/rerun manifests
`blue-rl-feasibility-audit` findings (invalid FQE regression, unread IQL
target Q, dropped final reward, mask mismatch). `drift-*`, `bc-v1mix-*`,
`iql-v2auto-*` (10-ckpt pool, -97.9 over 160 fresh-seed cells),
`review-followup-20261003.json`.

## 2026-10-04 — guard matrix + claim repair
- 13:03–13:08 Phase -1 packages; 15:05 Phase 0 audit/relabel, Phase 2
  ordering headroom +16 (later found to be a coverage artefact).
- 15:59–16:16 Phase 1 parity-gated rebuild, MAPPO smoke.
- 18:26 `e6b805d` residual PPO pilot (20 iters, -83.0; invalidated by
  defective metrics, never rerun).
- Guard-matrix run + `2900711` tie-break fix, guarded-oracle stop.
- Evening: handoff `blue-handoff-20261004.md` written (P0–P7, statistical
  design); six overstatements corrected after two external reviews.

## 2026-10-05 — build and test (this branch, `blue/metric-repair-maxage`)
- 10:00–10:04 P0 repair + MAPPO-guide build + harness commits.
- 10:04 P1 proposals (seed block 7809–8200, CRN request).
- 12:20 pilot-2 smoke note; 12:55 Phase A attribution (near-tie flips).
- 18:13–18:44 Phase B (finite red, paired -399.6) + Phase C shield
  (margin-gated fallback, paired +0.00) + margin sweep.
- 19:34 headroom hunt (no deployable ordering beats Lancer; guide2 frozen).
- 19:39 seed grant recorded + evidence freeze (`evidence-freeze-20261005`).

## Unrecoverable ordering
Within 2026-10-02, manifest times are day-level; relative order of the
`fact-*`/`iql-*`/`red-*` runs is not established. Nothing in the current
plan depends on it.
