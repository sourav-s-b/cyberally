# Lancer audit (Phase 0, 2026-10-04)

Question: does local `hybrid_lancer_v2` implement the published CAGE-4
Lancer? Checked against `blue/policies/hybrid.py`, `blue/core/wrapper.py`
(action space), `blue/core/baselines.py`.

## Findings

1. Decoys ABSENT — structurally. The wrapper action space is Sleep /
   Monitor / Analyse / Remove / Restore per host (`ACTION_TEMPLATES`,
   `wrapper.py`). No Deploy/decoy action exists, and `Deploy` appears
   nowhere in `blue/`. Published lancer's decoy deployment cannot be
   expressed here. Not an implementation gap — an environment gap.
2. Priority dynamics PRESENT. `LancerPriority`: init value, touch-decay
   on completed actions, detect-boost on CONFIRMED transitions,
   novelty-boost on view deltas, sticky suspicion bonus with
   fruitless-decay. This maps to "priorities change with actions and
   observations". Magnitudes unevaluated — that is Phase 2's job.
3. Response logic LOCAL, not published. Rules 1-3 (remediate CONFIRMED
   with Remove→Restore escalation, verify oldest-first, sweep the
   rest) are our own evidence-gating design around the local tracker
   states. Reasonable, but not a reproduction claim.
4. Monitor equivalence ASSUMED. Local policy never issues Monitor;
   relies on the wrapper merging unsolicited Monitor observations every
   tick. Same event stream only if the sim delivers it — true here by
   construction (`wrapper.py` merge), fragile to env change.

## Relabel decision

Report `hybrid_lancer_v2` as "lancer-style priority sweep with
evidence-gated remediation (no-decoy local variant)". Do not call it
the published Lancer. History (manifests, session) keeps old names;
new claims use the relabeled name.
