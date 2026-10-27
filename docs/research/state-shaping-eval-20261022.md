# State-shaping matrix: completed paired readout

**Date:** 2026-10-22. **Branches:** training on `gpt-blue` (state-reward
implementation, commits `8cf1f93`..`2f44db1`); checkouts in the main working
tree at `blue/llmred-fresh24`.

## Question

Can state-based training shaping (`detection_debt` + `confirm_debt`,
weight 0.05, cap 20; see `blue/harness/state_reward.py`) produce a harness
actor that beats Lancer on native return, where action-scored shaping and
plain residual PPO both failed?

## Design (frozen by `scripts/blue_state_shaping_run.sh`)

- Train: 32 iters x 4 fresh episodes x 400 steps, seeds 7706-7709 (frozen to
  main32), bonus 0.25, temp 0.5, max-age 80, ml_inputs both. Paired arms
  differ only in `--state-shaping off|both`.
- Held-out readout: seeds 7665-7688, native reward only at evaluation.
- Cells: finite (6 new models) and discovery (unshaped reused from main32;
  discovery/shaped was NOT trained - the matrix launched finite-only).
- Lancer baselines precomputed per attacker and cached.

## Provenance incidents during readout (both fixed)

1. The REUSE discovery/unshaped cells point at main32 checkpoints, but the
   script passed the pilot scorer (`94904879...`) to them. `load_hook`
   refused with checkpoint/scorer mismatch. Rerun with the main scorer
   `kaggle_main_artifacts/scorer.pkl` (`ec4026a3...`) as
   `eval/discovery/unshaped/s{0,1,2}-fix` (the failed first attempt left an
   empty `s*/` directory; do not pair those names).
2. Discovery/shaped cells are missing (never trained); the finite readout
   answers the predeclared question without them.

## Finite-Red results (primary, discriminative attacker)

Lancer: **-151.79** (24 seeds, worst -483).

| arm | per training seed paired-vs-Lancer | arm mean | 95% t CI (df=2) |
|---|---|---:|---|
| unshaped | -45.42, -13.96, -10.08 | **-23.15** | [-71.29, +24.99] |
| shaped | -45.92, -6.92, -56.79 | **-36.54** | [-101.70, +28.61] |

- Both CIs overlap zero; both point estimates are below Lancer.
- The unshaped per-seed values match
  `finite-red-native-gap-diagnosis-20261015.md` exactly (-45.42, -13.96,
  -10.08), which cross-validates this readout against the independent
  reproduction on `blue/finite-red-check`.
- End-of-training diagnostics are unchanged from the known pathology:
  sampled override 0.85, greedy agreement 0.34-0.46, and fixated sweeps
  (max per-host analyses 120-155 vs Lancer's ~40-60).

Decision: **state shaping does not rescue residual PPO under finite Red.**
No confirmation cohort, no LLM-Red progression from this pilot. The
reward-scale/personal-shaping hypothesis is closed; the failure signature
(override rate + under-remediation timing) points at policy-side
constraints, not reward scale.

## Discovery-Red results (regime-matched control)

Lancer (discovery, 24 seeds): -92.04. Reused main32 "both" checkpoints:

| s | paired-vs-Lancer | 95% CI |
|---|---:|---|
| 0 | -1.38 | [-62.2, +59.5] |
| 1 | +2.62 | [-55.5, +60.8] |
| 2 | -2.00 | [-52.9, +48.9] |
| arm | -0.25 | [-6.48, +5.98] |

This is a new negative fact: the main32 "both" arm, whose +5.19
[-1.46, +11.85] vs Lancer on the REUSED dev block 8221-8252 was the only
positive point estimate in the main32 table, is dead even with Lancer on
this fresh block. The +5.19 belongs to dev-block reuse, not to the policy.

## Tail view (descriptive only, 24 shared eval seeds, 3 training seeds)

| model | mean | p10 | worst | eps < -300 |
|---|---:|---:|---:|---:|
| lancer | -151.8 | -204.6 | -483 | 1 |
| unshaped/s0 | -197.2 | -304.3 | -488 | 3 |
| unshaped/s1 | -165.8 | -218.6 | -268 | 0 |
| unshaped/s2 | -161.9 | -238.7 | -362 | 1 |
| shaped/s0 | -197.7 | -294.6 | -451 | 3 |
| shaped/s1 | -158.7 | -229.4 | -242 | 0 |
| shaped/s2 | -208.6 | -332.8 | -450 | 4 |

Seed-level spread dominates: s1 avoids the bad tail in both arms while s0/s2
do not. Individual-seed tail winners exist but n=3 training seeds on one
held-out block cannot select on them; tail-first checkpoint selection would
need per-iteration snapshots, which the current trainer does not save
(only the latest `checkpoint.pt`). Add `--save-every-k` before any tail-
first training attempt.

## Next (pre-decided sequencing, after this evaluation)

1. Margin-bounded greedy override: unlock the frozen anchor pilot with a
   preregistered logit-margin deviation budget (e.g. <=3 overrides/episode,
   only when actor margin over the Lancer candidate exceeds a threshold).
2. Advantage-filtered expert-anchored updates (never-run shortlist item).
3. BC warm-start chain: fine-tune the residual actor from the BC checkpoint
   (the only policy family that has beaten Lancer on finite Red) instead of
   Lancer-init.
4. Rule-level follow-through fix (re-verify UNKNOWN in Op-A within N ticks)
   evaluated separately, then RL on top of the fixed rules.
