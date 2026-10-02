# Proposal 02: Host-then-command factorized actor

- Status: proposed (design only, no code)
- Date: 2026-10-02

## Description

Replace the single 155-way actor softmax with a factorized head: 51-way host
selector `pi(host|s)` (+ global Sleep/Monitor gate), then conditional command
head `pi(cmd|host,s)` over Analyse/Remove/Restore. Per-slot validity masks
apply inside the host selector; padding slots masked out. Train within the
existing EPyMARL MAPPO loop (joint log-prob bookkeeping for PPO).

## Motivation / evidence

- SR-DRL on CAGE-2 (Nyberg & Johnson, arXiv:2407.05775): `pi(host) =
  softmax(phi1(h))` then `pi(cmd|h_i)`; zero-shot across 7 topology variants
  where flat MLPs cannot even run (-113 ± 63 vs untrained MLP -2104).
- GACD (arXiv:2506.22706): same host->action family with two PPO nets; on
  par with non-OT GNNs (its OT machinery rejected, its factorization kept).
- Directly tests the program's core diagnosis (see 12): "which host" becomes
  an architectural argmax instead of a learned dense-bottleneck comparison.
- BC-distillable: round-robin traces give (state, host, cmd) triples for
  supervised head init before any RL.

## Experiment

Not yet run. Planned: implement head in EPyMARL actor; BC-distill from
round-robin traces; short MAPPO at 1e-5; >= 8-seed native eval vs hybrid
(01) and teacher (-93.5).

## Pros

- Keeps PPO/MAPPO and all existing infra; change is localized to the actor
  (+ critic pooling).
- Preserves masking semantics; no new features needed (see 12).
- Each half is independently checkable (host-selection accuracy vs teacher;
  command accuracy given host).

## Cons

- Sequential sampling complicates PPO log-probs, entropy accounting, and
  EPyMARL's single-softmax assumptions — moderate plumbing risk.
- Host/command miscoordination during early training (good host, wrong cmd)
  can look like failure of the idea rather than of optimization; needs the
  BC-distill stage to be taken seriously.
- Effort M with no guaranteed gain over the far cheaper hybrid (01).

## Verdict

BUILD SECOND, after 01 sets the bar. If the hybrid already approaches the
lancer band, this proposal must clear a higher bar to justify its complexity.
