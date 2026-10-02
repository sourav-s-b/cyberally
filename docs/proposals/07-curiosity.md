# Proposal 07: Curiosity (ICM/RND) for sparse reward

- Status: rejected
- Date: 2026-10-02

## Description

Add intrinsic curiosity (ICM forward/inverse-model loss, a la Mindrake) to
PPO to manufacture dense signal from sparse native reward.

## Motivation / evidence

Mindrake CAGE-1: PPO+ICM won (-30.07); "substantial improvements over
vanilla PPO … and improved upon our attempts to create a more dense reward
function manually"; nearly 2x on b-line discovery. Retained for B-line
(fast, sparse attacker) in CAGE-2, dropped for Meander.

## Experiment

Not run — rejected on diagnosis fit before spending the experiment.

## Pros

- Published winner on CAGE-1; beats hand-shaping where discovery is the
  bottleneck.

## Cons

- Solves discovery sparsity. Our discovery problem is already solved by
  sweeping: round-robin reaches -93.5, and every host gets visited. There is
  no undiscovered-state bonus left worth an ICM module.
- Adds forward/inverse models, hyperparameters, and non-stationarity to a
  pipeline whose binding constraint is the actor's output shape (see 12),
  not exploration.

## Verdict

REJECTED. Revisit only if a future red (e.g. stealthy variant) makes
systematic discovery the bottleneck again — i.e. when sweeping stops
finding the attacker.
