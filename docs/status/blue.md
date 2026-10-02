# Blue program summary (general index)

Objective: a CAGE-4 Blue defender that beats the round-robin teacher
(-93.5 ± 27.8, 400-step native episodes) and holds up off-distribution.
Branch `blue/mappo-training` (pushed, tracks origin). Details live in
`docs/proposals/` (15 records + manifests), `docs/research/`, and the
chronological `blue-session.md`. This page is the short index — update it
at each meaningful handoff. (Prior foundation-era content superseded;
see git history.)

## What was tried, in order

1. **Literature** (`docs/research/`): sparse-reward/forgetting review
   (warmup tested: null), then architecture review — official CAGE-4
   analysis shows the podium was all sweep-and-remediate heuristics.
2. **Hybrid rules + ordering** (proposal 01, CLOSED): parity proven
   byte-identical; lancer constants hit -63.2 regression but failed
   held-out (-110.1 ± 130.4, incl. a -414 seed); v2 regularized to parity;
   snapshot risk scorer (AUC 0.67) collapsed coverage (-191); risk×recency
   hybrids failed the held-out gate. Fixed ordering cannot win robustly.
3. **Factorized actor + distill ladder** (proposals 02/10/11): host-then-
   command head (zero vendor edits) distills the teacher at 21.8% vs 12.2%
   flat; aux host-loss hurts; 1-layer slot attention reaches **teacher
   parity on held-out with no RL** (-93.9 ± 45.8 vs -85.1 ± 46.1).
   Every MAPPO fine-tune diverges (flat and factorized); KL-to-teacher
   (recorded vendor patch) only slows it. Fine-tune line closed absent a
   better distill.
4. **Generalization suite** (proposal 15, standing): `red_agent=` flag +
   pool support; finite red much harsher than discovery (-139 vs -59);
   lancer_v2 ≈ teacher on every red. Green rates/durations need
   Environment hooks (`docs/coordination/blue-red-variants.md`, draft).

## Standings (8-seed native means; manifests in `docs/proposals/manifests/`)

| policy | regression | held-out | note |
|---|---|---|---|
| round-robin (teacher) | -93.5 ± 27.8 | -85.1 ± 46.1 | the bar |
| hybrid_lancer_v2 | -75.2 ± 33.6 | -91.2 ± 48.5 | best fixed; no transfer edge |
| attn-1 distill (learned) | -85.5 ± 31.1 | -93.9 ± 45.8 | first learned parity, no RL |
| flat BC | -393.4 ± 276.9 | — | learned start point |
| MAPPO fine-tunes | diverge (-500 → -1710) | — | closed |
| risk / risk×recency | -191…-251 | — | closed (coverage collapse) |

## Open threads / next

- KL-anchored PPO from the attention ckpt is unlocked (stop-rules apply);
  pristine seeds 8201+ reserved for the final claim.
- Coordination proposal awaits Environment review (green rates, durations).
- Infra: process-pool eval + manifests; 2 vendored-learner patches recorded
  in `environment/patches/` (third_party itself is gitignored).
- Suites: sim 112 passed / 7 skipped; train 22+ passed. No privileged state
  in any actor/mask/scorer (audited); native reward is the only score.
