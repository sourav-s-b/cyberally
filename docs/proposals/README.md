# Blue proposals: decision trail

One file per proposal. Each file follows the template below so tested and
untested ideas are comparable. Numbers are native CC4 reward unless stated.
Evaluation standard: >= 8 seeds, mean ± std; 3-seed comparisons are inside the
±313 noise band and prove nothing (see `docs/research/`).

## Template

Each proposal file has: Status, Description, Motivation/evidence, Experiment
(config + results, if tested), Pros, Cons, Verdict.

Statuses: `tested` (ran, results recorded) · `drafted` (code exists,
unbenchmarked) · `proposed` (design only) · `deferred` (take pieces now, rest
later) · `parked` (valid but not now) · `rejected` (evidence against).

## Index

| # | Proposal | Status | Verdict |
|---|---|---|---|
| 01 | Hybrid rules + learned priority | drafted | BUILD FIRST |
| 02 | Host-then-command factorized actor | built+tested | head validated (distill 21.8%); vanilla MAPPO diverges; KL 0.1 slows only — fine-tune line closed |
| 03 | Entity-attention encoder | proposed | BUILD WITH 02 |
| 04 | TERLA-style semantic action collapse | deferred | take action-waiting audit only |
| 05 | Full GNN policy | parked | revisit only on generalization suite |
| 06 | Learned hierarchy master / bandit gate | rejected/queued | reject PPO master; bandit gate queued |
| 07 | Curiosity (ICM/RND) | rejected | solves a problem we do not have |
| 08 | Critic-only warmup (WSRL-style) | tested | no effect; keep as ablation only |
| 09 | Ad-hoc reward shaping | tested | no reliable gain; native-only rule stands |
| 10 | BC pretrain + MAPPO fine-tune | tested | teacher not beaten; fine-tune deltas inside noise |
| 11 | Balanced (inverse-sqrt) BC | tested | beats collapsed BC; still far from teacher |
| 12 | Representation check (stalest-first) | tested | cursor recoverable from obs; bottleneck is architectural |
| 13 | Factored-additive Q | rejected | misspecified for 1-of-155 exclusive choice |

Score-comparability rules (binding for every file here): our episodes are 400
steps, official CAGE-4 is 500 — compare per-step rates, never raw totals.
TERLA/H-MARL absolutes come from different builds — only relative deltas
transfer. Privileged simulator state never enters actors or masks; it may be a
*training target* or *eval label* only.
