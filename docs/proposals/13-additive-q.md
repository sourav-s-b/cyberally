# Proposal 13: Factored-additive Q decomposition

- Status: rejected (theory mismatch)
- Date: 2026-10-02
- Source: "Leveraging Factored Action Spaces for Efficient Offline RL"
  (NeurIPS 2022).

## Description

Decompose `Q(s, slot, cmd) = q_slot(s, slot) + q_cmd(s, cmd)` (+ state
baseline) to replace 155 heads with 51 + 3, cutting sample complexity via
variance reduction (smaller Rademacher complexity, Prop. 5).

## Motivation / evidence

Sepsis-simulator + MIMIC-III offline RL: factored Q beats joint Q, especially
with limited data; with abundant data the bias dominates and joint catches
up. Formal zero-bias conditions exist (§3/App. B).

## Experiment

Not run — rejected on structural mismatch before spending the experiment.

## Pros

- Principled variance reduction where data/coverage is limited (our regime).
- Could still serve as a critic-side baseline lens for credit assignment.

## Cons

- The paper factors *simultaneously executed* sub-actions. Our action is an
  *exclusive* 1-of-155 choice; Restore-vs-Analyse on the same host interact
  strongly, so the omitted interaction term is the whole decision —
  misspecified exactly where it matters.
- Breaks EPyMARL's single-flat-softmax assumption for speculative gain.

## Verdict

REJECTED as an architecture. Keep the paper as a lens only: if proposal 02
needs variance reduction, use its hierarchical/sequential reformulation
(slot selector + conditional command head — which is 02, not this), or a
factored critic baseline.
