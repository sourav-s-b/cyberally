# Proposal 10: BC pretrain + MAPPO fine-tune

- Status: tested
- Date: 2026-09-30 / 2026-10-01

## Description

Behavior-clone the round-robin teacher (`blue_collect_bc.py` ->
`blue_bc_pretrain.py`), then MAPPO-fine-tune the actor from the BC
checkpoint at low lr (1e-5), optionally with warmup (08) or shaping (09).

## Motivation / evidence

Standard recipe: imitate a strong heuristic, then let RL improve on it.
Pretraining literature (FPC, WSRL) says cloning + fine-tuning can work if
coverage and stability are handled.

## Experiment

- Pipeline: round-robin demos -> BC (inverse-sqrt reweighting, see 11) ->
  checkpoint `results/models/bc_rr_20261001T141208Z` -> MAPPO (`steps 400`,
  `t-max 12000`, `lr 1e-5`, `temporal-groups ages belief`).
- Infra fixes along the way: BC quad optimizer-lr inheritance bug fixed
  (checkpoint stamped lr no longer leaks into fine-tune); checkpoint geometry
  requires matching temporal groups (`[64,872]` vs `[64,1025]` mismatch
  otherwise).
- 8-seed BC-init eval (7629, 7630, 7640, 7701–7705, 400 steps, native):
  -256, -379, -211, -323, -134, -324, -1023, -497
  -> mean -393.4, std 276.9. Same-teacher 3-seed mean was -282: the teacher
  itself is seed-sensitive, so no fine-tune delta measured on 3 seeds means
  anything (95% CI ≈ ±313).
- Fine-tune outcomes: genuine 1e-5 runs never beat the teacher; their deltas
  sit inside noise. Only hard failures reproduce: lr=1e-3 diverges,
  Sleep-collapse policies score at Sleep level. Warmup (08) changed nothing
  (drift 0.0047 vs 0.0048, KL ~0.001 nats).

## Pros

- Produces the best learned starting point we have (BC init).
- Surfaced real infra bugs (opt-lr leak, geometry mismatch) and forced the
  noise-floor analysis that now governs all claims.

## Cons

- RL has never improved on the teacher: BC 12% non-sleep accuracy (see 11)
  plus a flat head that cannot express argmax (see 12) means fine-tuning
  starts from a policy that cannot represent the teacher's core operation.
- Every "RL vs teacher" comparison before the 8-seed eval was underpowered;
  old "net-destructive" language is retracted.

## Verdict

BC init stays as the learned starting point. Further fine-tuning is paused
until the actor can represent the teacher (proposals 02/03) — more PPO on
the flat head is not expected to help, by measurement, not by hunch.
