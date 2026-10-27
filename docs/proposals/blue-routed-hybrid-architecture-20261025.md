# Situation-routed SetActor + attention Blue policy

Status: frozen for a descriptive longer training/evaluation run, 2026-10-25.

## Hypothesis and architecture

SetActor pools candidate evidence with mean/max context. Candidate attention
models interactions among candidate hosts. A global learned gate can use the
current candidate set and Lancer score distribution to blend the two score
residuals according to the observed situation.

For each decision state, the router encodes masked mean and max candidate
features, mean and standard deviation of normalized Lancer scores, and the log
of the candidate count. A sigmoid gate `g(s)` mixes the two branches:

`residual(s, host) = (1 - g(s)) * SetActor(s, host) + g(s) * Attention(s, host)`

`g=0` routes fully to SetActor and `g=1` routes fully to candidate attention.
The gate is shared across candidates in one decision, so it represents a
situation-level route. Existing masked action distribution, strict Lancer
teacher mixture, reward, scorer, MAPPO update, and Red implementations remain
unchanged. Both branch heads and the gate head initialize to zero, giving an
initial gate of 0.5 and zero residual over the existing policy mixture.

## Longer run protocol

- Two independent policy/training RNGs (`--seed 0,1`), 6 PPO iterations each.
- Four 400-step episodes per iteration, alternating Discovery and Finite Red;
  24 training episodes per RNG, 48 total.
- Training environment seeds: RNG 0 uses 8253–8276; RNG 1 uses 8277–8300.
- Evaluation seeds: 8401–8416, 8 per Red regime, held out from training and
  common across the two policy RNGs.
- Same frozen pilot controls: strict Lancer and an untrained hybrid using the
  same stochastic teacher-mixture distribution.
- Primary reports: native return, paired learned-minus-Lancer return within
  each Red regime, and route-gate mean/std/range over learned free decisions.
  Coverage and decision counts remain in the raw episode reports.
- Seeds were audited free with `blue.training.seed_audit`; no confirm block is
  opened. This 2-RNG result is descriptive and cannot establish a winner.

## Rationale and limits

The previous 2-iteration/2-evaluation-seed pilot was too small and showed
different weakness by Red regime. This run gives each trained model three times
the PPO iterations, six times the training episodes, and four times the eval
seeds per regime. It tests whether an adaptive mixture can reduce the Finite
weakness without losing Discovery performance. A route-gate mean alone does not
prove useful routing; compare its per-seed spread with the returns and follow
with a larger multi-seed confirmation only if the effect is replicated.
