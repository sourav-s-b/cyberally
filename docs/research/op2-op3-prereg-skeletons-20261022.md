# OP2/OP3 pre-registration skeletons (2026-10-22)

Owner: Blue. Branch `blue/contract-suite` (worktree `worktrees/contract`).
Status: **skeletons only — nothing frozen, no run started.** These are the
pre-registration drafts the user asked for before committing compute. Both JSON
skeletons live in `docs/proposals/manifests/` (committable):

- `op2-jsrl-guide-skeleton-20261022.json` — scripted-guide shrinking-horizon
  warm start (JSRL, arXiv 2204.02372; sensitivity arm kickstarting
  arXiv 1803.03835/2207.11584).
- `op3-awdc-logs-skeleton-20261022.json` — masking-aware offline extraction on
  collected logs (AWAC arXiv 2006.09359 family, ReBRAC arXiv 2305.09836
  protocol, CQL arXiv 2006.04779 critic only).

## Shared conventions (identical in both preregs)

Seed audit (authoritative, not a grep — `blue.training.seed_audit`):

- `--check 8433 8500` → **FREE, no recorded consumption** (68 seeds; 112 used
  seeds discovered from structured sources). This is the last audited-free
  window; any usage must be re-audited and recorded at freeze time.

Allocation if the frozen prereg is adopted (dev-gate split, no post-hoc
reassignment):

| purpose | seeds | count |
|---|---|---|
| training / behaviour-logging rollouts | 8433–8444 | 12 |
| dev evaluation (gate decides) | 8445–8464 | 20 |
| spare (unused budget growth) | 8465–8480 | 16 |
| confirm block (opens ONLY if the dev gate passes) | 8481–8500 | 20 |

Reason the confirm block is only opened at gate pass: the margin-bounded
experiment's confirm gate was *not* met and its fresh block was therefore
never spent — that pattern is inherited here deliberately.

Statistics rule: paired per-seed percentile bootstrap n=10000,
`rng_seed=20261022` (matches `guard-maxage` manifests); ReBRAC-style
multi-seed reporting (mean ± std over ≥10 eval seeds, never single-seed
claims — per `docs/research/web-research-20261022.md` OP5). The treatment is
always compared against the frozen guard heuristic **measured fresh on the
same window with the same wrapper** — historical numbers (lancer −151.79 /
guard −150.21 on 7665–7688) are recorded context, not control values.

Hard gates, both preregs:

- **G0** — cold-contract audit (`test_contract_cold.py`, 8 tests, native
  scenario, no mocks) green at the frozen commit. Currently green on this tip.
- **G1** — for OP2: guard re-measured on the dev window before any training.
  For OP3: data-contract check (below) before the logging batch runs.
- **G2** — no mid-run changes; a found defect aborts the run and is reported,
  never salvaged.

Reporting rules: negatives are reported as negatives (the margin experiment's
precedent); confirm blocks open only via a user-visible fresh `seed_audit`
check; no results outside the prereg metric list are part of any claim.

## OP2 — JSRL scripted-guide warm start (what it answers)

Question: does wrapping fresh PPO rollouts with a shrinking guide window from
the frozen guard fix the sparse-credit problem (41% nonzero rewards, first
nonzero ~step 135) that our native-reward RL runs suffered?

Key prereg choices:

- Guide = the existing frozen guard scripted policy; guide acts for the first
  `H_i = max(200 × 0.6^i, 10)` steps of each episode; PPO takes over the rest.
  Handover is complete by iteration 5 of 12.
- `mappo_guide.py` (existing central-critic harness) supplies the training
  path; fresh rollouts every iteration; KL cap 0.02; no shaping.
- Four arms: frozen guard control / JSRL treatment / no-guide ablation
  (isolates warm-start effect) / kickstarting sensitivity (secondary).
- Compute: CPU torch, ≤6 h cap. Fails soft: report truncation as outcome.

Risks recorded in the skeleton: the guard→residual mixture needs a guide
injection point in the rollout loop that does not exist yet (implementation
work, not design risk); native-reward RL runs previously collapsed to −30.95,
so the no-guide ablation doubles as the historical-collapse reproduction check.

## OP3 — offline extraction on logs (what it answers)

Question: can an offline masking-aware extractor (BC init + AWAC-style
advantage-weighted extraction, no fresh rollouts during updates) reach guard
within TAU = 1.0 on held-out seeds — the extraction-lottery question without
any training-side seed variance?

Data finding recorded 2026-10-22 (this is the skeleton's most important
content): **no transition logs under the current wrapper obs contract exist.**
`offline_logs_14{,b,strong,full}.npz` are 872-dim legacy-obs logs (24/60
episodes, behaviour: older logging runs); `bc_demos_rr(_v2).npz`
(10 episodes) have no rewards channel; only `scorer_rows*.npz` and
`gpt_harness/data.npz` (row-form labelling data, not transitions) are
current-era. Therefore OP3 **requires a fresh behaviour-logging pass** on the
current wrapper (60–120 episodes on the 12 training seeds, ~3 h cap) before
any offline step. Converting the 872-dim legacy logs is explicitly rejected
in the prereg (unproven; re-collection is cheaper and cleaner). Legacy IQL
numbers (−96.0 vs −85.1 held-out round-robin) stay historical context and are
not treated as a fresh baseline.

Key prereg choices:

- Actor: masking-aware registered policy class, BC init, then
  advantage-weighted regression using an offline-fitted conservative critic;
  no online stage (an online fine-tune would need its own prereg).
- Critic: CQL-style lower-bounded Q, fixed lr, no ensemble in the primary
  arm (ensemble is sensitivity-only).
- G2 sanity: BC-init deterministic eval must NOT exceed guard by more than
  +1 on dev seeds — if it does, the logging recipe leaks and the offline
  stage aborts before extraction (a strong-heuristic logging policy *should*
  evaluate slightly below guard when its own noise is held constant).
- Compute: CPU, caps 3 h logging + 4 h extraction, .venv (+torch for critic).

Risks recorded in the skeleton: 12 training seeds is a small logging pool
(revision lever: more episodes per seed, not more seeds — seeds stay pinned);
BC-extract instability at this scale is documented in the literature ledger
(2109.00137), which is exactly why the guard-relative TAU is +1.0 not 0, and
why confirm-gate reporting is per-RNG-seed (H2, secondary).

## What remains before a run (explicit freeze checklist)

1. User picks OP2 or OP3 (or both, sequentially — they do not share code paths;
   only the seed ledger, guard measurement and G0 gate).
2. Fill in source-file shas + wrapper `get_env_info()` hash at freeze time
   (placeholders currently absent by design; the margin pilot's
   `frozen_plan.json` is the reference format).
3. Commit the frozen JSON **before** the first rollout (artifacts stay
   gitignored; the manifest is the public record).
4. OP2 only: implement + test the guide-injection point
   (`mappo_guide` guide-window schedule) before freezing.
5. OP3 only: implement + test the logging pass (Blue-owned writer) so that
   OP1's cold-contract invariants apply to it too: mask-legal logged actions,
   Blue-visible obs only, per-episode provenance in the manifest.
6. Immediately before opening confirmation seeds: re-run
   `python -m blue.training.seed_audit --check 8481 8500` and record output.
