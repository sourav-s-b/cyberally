# Proposal 11: Balanced (inverse-sqrt) BC

- Status: tested
- Date: 2026-09-30

## Description

Reweight BC training batches by inverse action frequency (inverse-sqrt
variant) instead of uniform sampling, so rare non-Sleep teacher actions are
not drowned by the dominant Sleep class (~69% of agent-ticks have only Sleep
legal).

## Motivation / evidence

Imbalanced-BC literature (arXiv:2508.06319; `q(s,a) = p/rho` corrections):
naive BC on skewed demonstrators collapses to the majority class. Our
round-robin traces are exactly such a skewed set.

## Experiment

- Unweighted BC: collapses to near-always-Sleep (policy scores at Sleep
  level; exact-action accuracy on non-sleep actions ~0).
- Inverse-sqrt reweighted BC: ~12.16% non-sleep exact-action accuracy.
  Still low in absolute terms, but a real policy instead of a collapse, and
  the checkpoint all fine-tunes start from.

## Pros

- One-line-class fix that turned a collapsed policy into a trainable init.
- Matches published imbalanced-BC findings; no architecture change needed.

## Cons

- 12% shows reweighting mitigates the symptom, not the cause: the flat head
  still cannot express per-slot argmax (see 12), so most of the remaining
  error is architectural, not distributional.
- Reweighting distorts the action prior; a downstream RL stage must unlearn
  that bias (acceptable here since RL never got going for other reasons).

## Verdict

Keep inverse-sqrt as the BC default. Do not expect further reweighting
variants to close the teacher gap — that needs proposals 01–03.
