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

1. Log dataset spec: 872-dim obs + validity mask + action + native reward
   from lancer_v2/round-robin traces (8101+ demos plus regression-8
   traces). Masks are REQUIRED — AWR extraction must range over valid
   actions only, else it re-learns masked-action mass.
2. Rung 1 (days): RvS return-conditioned MLP on the factorized head.
   Tests whether return-conditioning alone unlocks anything.
3. Rung 2 (1–2 weeks): IQL-discrete — expectile V (tau ~0.7–0.9 sweep),
   SARSA-style backup, AWR policy extraction (temperature sweep). Start on
   the factorized head (21.8% extraction surface beats flat 12.2%).
4. Gate (binding): beat round-robin on HELD-OUT 7801–7808 (mean and paired
   diff), same bar that killed 01/13/riskx. Report vs attention-distill
   parity line (-93.9 ± 45.8) as the secondary comparator.
5. Diagnostic read: RvS ties + IQL beats → stitching was the missing
   piece. Both tie → logs lack improvable diversity → the ceiling is data,
   not algorithm; proceed to online exploration (16/17), do not tune
   expectiles further.

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
