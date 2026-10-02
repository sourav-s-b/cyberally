# Proposal 04: TERLA-style semantic action collapse

- Status: deferred (two pieces adopted as ideas; collapse itself not built)
- Date: 2026-10-02
- Source: Dudman & Bull, CAMLIS 2025 (PMLR 299:87-109).

## Description

Full TERLA: per-observation heterogeneous graph (Mission/Subnet/Host nodes,
2 IDS bits per host) -> 2 HGT layers + sum pool -> 70-dim latent -> PPO over
**5 semantic actions** (Sleep / Analyse-least-compromised /
Remove-most-compromised / Restore-most-compromised / Decoy-most-compromised),
resolved to concrete hosts by IDS-rank targeting (first match). Train on
shaped per-agent segment-health deltas (OT hosts x2); evaluate on native
shared reward. Plus "action waiting": withhold observations while an action
is pending, keep streaming rewards.

## Motivation / evidence

Directly on CC4 with RLlib PPO, ~1M steps. Relative results (their build;
absolutes NOT comparable to ours): sleep -6650 < shaping-only -6300 <
random -5150 < vanilla PPO -2825 ~= separate TERLA -2773 < single shared
TERLA -2048. TERLA acts on only 6-7% of steps vs PPO ~33% (mostly Restore,
phase-adaptive) — large efficiency gain at equal performance, and one shared
topology-agnostic net redeployed on all 5 segments wins outright.

## Experiment

Not run by us. Adopted pieces: (a) action-waiting audit — our wrapper
already fails explicitly on overdue unresolved actions instead of silently
unmasking, which is the same honesty property; keep it. (b) IDS-ranked
targeting as a design idea for proposal 01's priority scorer.

## Pros

- Attacks our hardest problem (which-of-153) by collapsing it to what-class
  (5 actions); targeting heuristic does the rest.
- Action waiting fixes a real MDP-corruption trap (silent Sleep conversion).
- Shared topology-agnostic policy is the only published result beating
  per-segment specialists on CC4.

## Cons

- Drops Monitor and Block/AllowTrafficZone — caps the strategic ceiling
  (UC won partly through firewall play).
- First-match IDS targeting is crude; no prioritization info exists in CC4
  obs, so ties break arbitrarily.
- Trains on shaped reward computed from **privileged red-session counts**:
  acceptable only as training-only auxiliary with native-reward selection,
  needs explicit approval — conflicts with a strict native-only reading.
- EPyMARL port friction (paper is RLlib + in-model graph build); effort M
  for encoder + wrapper + reward-parity work.
- Absolute scores from a different build; only relative deltas transfer.

## Verdict

DEFERRED. Revisit only if proposals 01–03 stall: the collapse is powerful
but lossy, and its training reward needs a rules-committee decision first.
