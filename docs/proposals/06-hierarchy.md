# Proposal 06: Learned hierarchy master / bandit mode gate

- Status: rejected (PPO master); queued (bandit gate, S effort)
- Date: 2026-10-02

## Description

Two-level policy: low level = per-mode sub-policies (Sweep/Verify/Remediate,
a la H-MARL's Investigate/Recover/ControlTraffic); high level = learned
master picking the mode per step. Cheaper variant: contextual bandit or
heuristic gate on the same mode choice, low level frozen.

## Motivation / evidence

- Mindrake CAGE-1 winners (HPPO + curiosity, -30.07): per-red PPO
  sub-agents + per-step PPO selector; train-sub-freeze-train-master
  pipeline. But the hierarchy solved "which attacker?" — a problem CC4 does
  not have (single FiniteStateRed).
- Mindrake CAGE-2 (3rd): RL controller -57.05 vs heuristic 4-step classifier
  -57.29 vs bandit -57.46 — learned gating is noise when the choice is
  fingerprintable.
- H-MARL on CC4 (AAMAS'25): expert-rule master -129.53 beats flat IPPO
  -181.62; learned Meta master ~= -181.62 (trains 3-5x faster, same score);
  joint co-training (Collective) failed — freeze the low level or else.
- Our fixed CONFIRMED/VERIFY rules already ARE the expert master, and the
  learned priority scorer (01) already IS the learned low level. The
  transferable half of HPPO is already in the plan.

## Experiment

Not run. Queued micro-experiment only: bandit/heuristic sweep-vs-remediate
mode gate over frozen hybrid, episode-return reward (~15k episodes a la
Mindrake bandit), >= 8 seeds.

## Pros

- Bandit gate is S effort and directly tests "does any learned high level
  add anything over rules" — consistent with Mindrake's own null result.
- H-MARL code is open (`adityavs14/Hierarchical-MARL`) if a full master is
  ever justified (e.g. firewall mode with real tradeoffs, where a fixed rule
  is unfavorable — their ControlTraffic case).

## Cons

- Full PPO master: non-stationarity if co-trained, sparse high-level credit
  (Restore penalty vs delayed Impact), 5-agent coordination overhead — for an
  expected gain of ~0 where modes are rule-separable.
- Per-red-expert framing does not apply to CC4's single red.

## Verdict

REJECT the PPO master. QUEUE the bandit gate behind proposal 01: run it only
if the hybrid plateaus and mode choice looks like the binding constraint.
