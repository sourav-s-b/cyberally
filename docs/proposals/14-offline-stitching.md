# Proposal 14: Offline stitching on heuristic logs (IQL-discrete + RvS baseline)

- Status: proposed (P1 — do first)
- Date: 2026-10-02

## Description

Learn to EXCEED the teacher from fixed logs, with no online RL. Two
rungs: (a) RvS return/goal-conditioned MLP as the cheap baseline; (b)
IQL-discrete (expectile value + advantage-weighted extraction) for
in-sample stitching of suboptimal fragments. Data already exists
(8101+ demos, regression traces). Train-venv only; no simulator change.

## Motivation / evidence

W4 is the demo ceiling: BC ties at best (21.8% match), attention clone
ties the teacher but cannot exceed it. IQL fits our exact failure mode:
our logs are narrow deterministic-teacher traces over 155 discrete
actions, where OOD Q-queries are the divergence mechanism — IQL never
queries OOD actions (in-sample expectile regression learns the upper
envelope V, AWR extracts the policy). Stitching is the missing capability:
combining seed A's good opening with seed B's good endgame, which no
per-step cloner can do. Survey: C15/C16, C26, and the Brandfonbrener
warning (C27) that return-conditioning alone cannot stitch.

## Experiment

RUNG 1 DONE (2026-10-02). RvS return-conditioned factorized GRU.

- Data: extended `blue_collect_bc.py` (--teacher/--mix, team rewards
  saved); 24 eps on 8101–8124, 8 each lancer_v2/RR/masked-random:
  returns -44…-304, bimodal (good -44…-123, weak -187…-304) — the
  conditioning signal exists. `results/offline_logs_14.npz` (gitignored).
- Model: `blue_rvs_pretrain.py` — factorized head + trailing RTG scalar
  (`rtg_dim=1` split in `FactorizedRNNAgent`; default path bit-identical).
  30 epochs: acc 35.9%, nonsleep 22.6% (in-sample, on par with BC).
- Eval: `RvSCheckpointPolicy` (`blue_eval_rvs.py`, registry `rvs_ckpt`,
  minimal `observe_step` hook in `run_episode` — no-op for heuristics).
  Pool `--target-return`, `--temporal-groups ages belief` to match logs.
- Target sensitivity (regression-8): target -50 → -112.6 ± 87.0 (tail
  -313 on 7705: demanding near-max returns destabilizes); target -80 →
  -87.1 ± 28.1 vs RR -93.5 ± 27.8 (parity, no tail). The policy LISTENS
  to conditioning — machinery works.
- Held-out 7801–7808 (target -80): rvs -94.0 ± 45.3 vs RR -85.1 ± 46.1,
  paired -8.9 (-60 +5 -4 +7 -8 +7 -7 -11); cf lancer_v2 -91.2 ± 48.5
  paired -6.1. Teacher parity with one tail episode — no exceed.
- Manifests: `rvs-20261002.json`, `rvs-t80-20261002.json`,
  `rvs-heldout-20261002.json`. Ckpt `results/models/rvs_14/` (local only).
- RUNG-1 VERDICT per the diagnostic: RvS TIES. Return-conditioning
  reproduces the teacher but cannot exceed it — the Brandfonbrener
  prediction holds. Stitching is confirmed as the missing piece.

## Pros

- Cheapest learn-to-exceed shot: no env change, data exists, runs in
  train-venv on existing stack.
- In-sample learning cannot diverge the way our MAPPO runs did — the
  failure mode under test is impotence (ties teacher), not collapse.
- Settles a strategic question (data vs algorithm ceiling) regardless of
  outcome.

## Cons

- Needs a Q-learning implementation in our stack: audit vendored EPyMARL
  algorithms (QMIX/QTRAN-adjacent code) for a discrete IQL before writing
  new learner code.
- Stitching is bounded by log diversity: 8 demos of a deterministic
  teacher may contain too few distinct good fragments. Mitigation: pool
  ALL teacher traces (demos + regression + red-variant runs) into one
  mixed log; multi-modality is IQL's home turf (C14).
- Expectile tau + AWR temperature are real hyperparameters; budget a
  small sweep, then stop — both-tie means stop, not sweep harder.

## Verdict

PROPOSED, P1, do first. Outcome routes the program: IQL wins → fine-tune
the IQL policy (16) instead of the attention ckpt; both tie → the ceiling
is data and online methods (16/17) are mandatory, not optional.

RUNG 2 ROUTING (2026-10-02): RvS tied — proceed to IQL-discrete
(expectile V tau 0.7–0.9, SARSA backup, AWR extraction over VALID actions
only, factorized head). Same held-out gate. If IQL also ties, the ceiling
is data (14-diagnostic, second clause) and 16/17 become mandatory.
