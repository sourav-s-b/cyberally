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

## Rung 2 results: IQL-discrete (2026-10-02, PARTIAL — regression win, gate failed)

- Implementation: `blue_iql.py` — feedforward Q/V MLPs on single-step
  transitions (obs already carries ages/belief), expectile V (tau 0.7),
  MSE Q backup (gamma 0.99, target net), AWR extraction into a plain
  factorized GRU (keys match EPyMARL layout → pool via
  `mappo_ckpt_factorized`). `--awr-alpha` blends AWR with uniform BC.
- Diagnostics healthy: loss_v 0.15, loss_q 1.6; adv std 0.54, max 6.9;
  weights clip 0.03%.
- Pure AWR (alpha=1.0) collapses a seed (-826 on 7630, touches 66 vs 86
  hosts): mechanism CONFIRMED — AWR drops zero-advantage sweep actions and
  narrows coverage. Beta 0.3 only halves it (-430). The `--awr-alpha`
  coverage guardrail is the fix, not beta.
- Regression-8 (RR -93.5 ± 27.8): alpha=1.0 → -174.8±265; 0.7 → -127.5±95;
  0.3 → -136±152; **0.5 → -82.0±27.7, paired +11.5** (+21 +35 +1 +19 +68
  +16 -48 -20). FIRST learned policy to beat the teacher MEAN on
  regression, with teacher-matched variance.
- Data follow-ups (regression): 60-ep log (added 8125–8160, tails to
  -578) → -110.0±63.6; strong-40 (weak teacher filtered) → -100.4±26.3.
  Both LOSE to the 24-ep mix: counter-intuitive, weak-teacher tails seem
  to sharpen the envelope (or train-seed noise) — recorded, not concluded.
- Held-out (alpha=0.5 champion, target of record): **-96.0 ± 47.5 vs RR
  -85.1 ± 46.1, paired -10.9** (-25 -37 -16 -25 +17 +24 -2 -23). No
  catastrophe, two seed wins (-34 and +24 diffs), but uniform mild
  degradation elsewhere. GATE FAILED.
- Manifests: `iql/a05/a03/a07/full/strong/heldout-20261002.json`.
  Ckpts local-only (`results/models/iql_14*/`).
- RUNG-2 VERDICT: stitching works IN-DISTRIBUTION (regression +11.5, 5/8
  seed wins) but does not transfer — neither diagnostic clause fires
  cleanly. The update is vindicated, transfer is the gap. Route: 16
  (online improvement from the IQL ckpt, which now replaces attention as
  the best init) and 17 (critic audit may explain the uniform held-out
  degradation). Second held-out look budget: SPENT cautiously (one fair
  run); further tuning stays regression-side; pristine 8201+ untouched.

## Phase A follow-up: Q-instability, not undertraining (2026-10-02)

- Scaled-iters retrain (60-ep, 20k→50k, seed 0): DIVERGED — loss_q 4.08 →
  420.9, adv_max 1159, 6.9% clipped. More budget without stabilization
  fails (deadly triad on a fixed log). Undertraining hypothesis DEAD;
  the 20k run was early-stopped luck.
- Train-seed check (24-ep, seeds 1–2): loss_q 8.44 / 6.36 (vs 1.60 seed
  0), adv_max 176 / 27. Regression: s1 -105.6±90.4 (7640: -319 tail),
  s2 -108.1±31.6. loss_q predicts performance; seed 0's -82 was a lucky
  ticket. Single-seed training conclusions SOFTENED across rung 2 —
  including the 24-vs-60/strong data gaps (confounded with train seed).
- Gate: nothing beat -82 → Phase B second held-out look CANCELLED,
  one-look budget PRESERVED.
- Lesson: rung-2 IQL works only as an optimization lottery ticket here.
  A rung 3 needs value stabilization FIRST (reward /100 to RTG scale,
  LayerNorm, lower lr / harder targets) before bigger logs — that work is
  17-adjacent. Manifests `iql-s1/s2-20261002.json` (no pool for the
  diverged 50k ckpt — diagnostics condemned it).
