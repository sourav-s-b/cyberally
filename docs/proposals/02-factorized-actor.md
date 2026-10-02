# Proposal 02: Host-then-command factorized actor

- Status: built — expressiveness VALIDATED (distill 21.8% vs 12.2%),
  MAPPO fine-tune DIVERGES (same disease, faster). Blocked on fine-tuning,
  not on the head.
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

## Experiment 1: built, distilled, fine-tuned, diverged (2026-10-02)

- Implementation `cage-challenge-4/blue_factorized_agent.py`
  (`FactorizedRNNAgent`): same GRU recurrence over the flat obs, but the
  final layer is a shared 51-slot encoder + host selector + conditional
  command head + global Sleep/Monitor head, combined as
  flat[2+3s+c] = host[s] + cmd[s,c]. Output stays plain 155-way logits, so
  masking, sampling, PPO, and agent.th save/load work UNCHANGED — zero
  vendor edits (runtime registration in `blue_train_mappo.py --agent`,
  `blue_bc_pretrain.py --agent`, `GreedyCheckpointPolicy(agent_type=...)`).
  Slot width derives from input shape (17, or 16 for the root-drop
  ablation). Critic untouched.
- Tests `tests_blue/test_factorized_agent.py` (5 tests, train venv;
  `importorskip` guard so the sim suite skips cleanly): shapes incl.
  821-dim geometry, host+cmd composition structure, grad flow into all
  heads, save/load roundtrip.
- Vendored `third_party/epymarl/.../rnn_agent.py` found with a 6-char
  insertion (`fc2(h)openco`) making the whole torch side unimportable.
  Isolated typo (repo-wide sweep clean); prior torch runs prove it was
  intact 2026-10-01. Deleted the 6 chars (restores upstream content; no
  patch file — nothing functional changed). Verified by the suites below.
- BC-distill (`results/models/bc_rr_20261002T082810Z`, 30 epochs, same
  demos/weights as flat BC): **nonsleep-acc 21.8% vs flat 12.2%** —
  the architecture expresses the teacher nearly 2x better under pure
  supervision. Architecture hypothesis VALIDATED.
- MAPPO fine-tune (t-max 12000, lr 1e-5, ages+belief, init distill):
  greedy eval `fact-mappo-20261002` (spawn path, 16 cells, 156 s):

| ckpt | 7629 | 7630 | 7640 | mean(3) |
|---|---|---|---|---|
| distilled (0) | -131 | -393 | -364 | -296 (flat BC: -282) |
| 2793 | -19 | -566 | -197 | -261 |
| 7581 | -741 | -683 | -325 | -583 |
| 12369 (8 seeds) | | | | ~-1770 |

  Flat through ~2793, collapse 2793->7581, worse after. Same fine-tune
  divergence as the flat head, faster. (The in-training probe's -75 was a
  single-episode seed lottery — cf. -19 on 7629 at ckpt 2793 — not evidence
  of health. Single-episode probes are noise; documented for future runs.)
- Verdict: the HEAD is vindicated, the FINE-TUNE is the blocker. Do not
  train this head with vanilla MAPPO again without stabilization:
  candidates are KL-to-teacher regularization, lower lr + entropy schedule,
  behavior-constrained updates, or deploying the distilled greedy policy
  with no RL at all (its -296 ~= flat BC -282; whether 21.8% accuracy
  converts to return is still open).

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

Update 2026-10-02: built and tested (see Experiment 1). The complexity was
lower than feared (combined-logit trick, no plumbing risk materialized)
and expressiveness doubled — but vanilla MAPPO fine-tuning diverges on it.
Status is now BLOCKED ON FINE-TUNING, not on the head: next work here is
stabilized updates (KL-to-teacher, lr/entropy schedule, behavior
constraints), not more architecture.
