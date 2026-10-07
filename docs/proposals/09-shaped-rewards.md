# Proposal 09: Ad-hoc reward shaping

- Status: tested
- Date: 2026-09-30 / 2026-10-01
- Commits: `ab6a8d2` (shaped@5e-5 verdict), `1e5b888` (shaped@1e-5 verdict),
  corrected by noise-floor analysis in `3efd80f`.

## Description

Densify the sparse native reward with hand-authored shaping:
`clear=3` (host cleared), `confirm=1` (detection confirmed),
`vandalism=-3` (destructive/wasteful action). Train on shaped, select on
native. This shaping is NOT potential-based (Ng–Harada–Russell), so it can
change the optimal policy.

## Motivation / evidence

Sparse native reward (first ~135 of 400 steps carry ~0) starves PPO credit
assignment. Shaping is the standard quick fix; TERLA also trains on a shaped
surrogate (segment-health deltas) while evaluating native.

## Experiment

- Shaped fine-tunes from BC init at lr 5e-5 and 1e-5, evaluated on native
  reward. Result: no reliable improvement over BC init or native fine-tune;
  deltas inside the 3-seed noise band.
- A prior "4/4 fine-tunes degrade / net-destructive" claim was RETRACTED:
  the 8-seed BC eval (mean -393.4, std 276.9, 95% CI ≈ ±313 for a 3-seed
  mean) showed 3-seed comparisons cannot support it. Only two effects stand
  outside noise: lr=1e-3 divergence, and Sleep-level collapse.
- TERLA-style principled shaping (per-agent segment-health delta, OT x2) was
  NOT tested here — see proposal 04 for why it stays deferred (it trains on
  privileged red-session counts).

## Pros

- Cheap to implement; confirmed the training loop propagates shaped reward
  correctly (plumbing works).
- Negative result killed a tempting path early.

## Cons

- Ad-hoc (non-potential) shaping risks teaching the wrong objective; observed
  effect indistinguishable from noise.
- Added a confound to early fine-tune comparisons before the noise floor was
  understood.

## Verdict

Native-reward-only rule stands for training and selection. Shaping may return
only as potential-based shaping with a proof of policy invariance, or as a
training-only auxiliary with native-reward selection — both need explicit
approval, never silent adoption.
