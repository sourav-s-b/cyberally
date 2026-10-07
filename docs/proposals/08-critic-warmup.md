# Proposal 08: Critic-only warmup (WSRL-style)

- Status: tested
- Date: 2026-10-01
- Commits: warmup implementation + `environment/patches/ppo-warmup.patch`;
  verdict folded into `3efd80f`.

## Description

Freeze the actor optimizer for the first N env steps of MAPPO fine-tuning
while the critic keeps training (WSRL-style online warmup), so a noisy
fresh critic does not drag a good BC-initialized actor off a cliff.
Config: `--warmup-steps`, `--warmup-critic-only` (default on).

## Motivation / evidence

WSRL (ICLR 2025, arXiv:2412.07762): frozen-policy rollouts + critic-only
updates (K=5000) fix early Q-collapse when fine-tuning from pretraining.
Stability-plasticity literature: low-lr fine-tuning alone is weak protection
against forgetting. Our case matched the setup: BC actor at 12% non-sleep
accuracy + fresh critic + sparse native reward.

## Experiment

- Implementation: `blue/blue_train_mappo.py` flags;
  `third_party/epymarl/src/learners/ppo_learner.py` skips the actor
  optimizer step during warmup (critic still trains). `third_party/` is
  gitignored, so the exact change is stored as
  `environment/patches/ppo-warmup.patch` + README; fresh clones must apply it.
- Regression: `blue/tests_blue/test_ppo_warmup.py` (5 tests:
  actor bit-identical during warmup, critic still moves, actor resumes after
  boundary, default-off, opt-out). Suites after change: train 90 passed /
  27 warnings; sim 81 passed / 5 skipped.
- Run (from `cage-challenge-4`, `../.venv-train/bin/python`):
  `blue_train_mappo.py --steps 400 --t-max 12000 --seed 7 --train-seeds 7629
  7630 7640 --save-interval 2000 --lr 1e-5 --temporal-groups ages belief
  --init-ckpt results/models/bc_rr_20261001T141208Z --warmup-steps 4000`
  (run `mappo_cc4_seed7_20261001T162853Z`). Checkpoint geometry requires
  matching `--temporal-groups`; without it, load fails with
  `size mismatch [64,872] vs [64,1025]`.
- Eval (400 steps, seeds 7629/7630/7640, native reward):

| variant | s7629 | s7630 | s7640 | mean |
|---|---|---|---|---|
| no warmup | -234 | -460 | -486 | -393 |
| warmup 4000 | -315 | -406 | -643 | -455 |
| BC init (same 3 seeds) | -256 | -379 | -211 | -282 |

- Mechanism diagnostics: relative actor-L2 drift warmup 0.0047 vs no-warmup
  0.0048; behavioral KL on 4000 teacher states 0.00163 vs 0.00087 nats
  (~zero). The policies are near-identical with and without warmup.

## Pros

- Clean, framework-local ablation of the "critic drags actor" hypothesis.
- Patch + tests make it reproducible despite the gitignored vendor tree.
- Default-off; zero risk to existing runs.

## Cons

- No effect: -455 vs -393 is inside seed noise, and the drift/KL diagnostics
  show warmup changed almost nothing — the failure is not critic-induced
  forgetting of this form.
- Adds a code path + patch-maintenance burden for zero gain.

## Verdict

Keep as a tested ablation only. Do not use for future fine-tunes unless a new
diagnostic specifically implicates early critic gradients.
