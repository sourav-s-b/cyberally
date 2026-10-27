# Proposal: Blue harness feature and ML artifact contract v1

- Status: draft, local prototype; affected-role review pending before merge.
- Author role and branch: Blue / `gpt-blue` in isolated worktree.
- Producer: `blue/harness`; consumers: new Blue PPO trainer, loader, diagnostic
  evaluator; Environment reviews optional runtime, Evaluation reviews metrics.
- Exact dependency: `243a6fd` (full base revision recorded in role handoff).
- No issue/PR sent and no teammate review requested on the user's behalf.

## Problem and current behavior

Legacy residual investigation can repeatedly select one host, postponing
detection elsewhere. Old feature paths conflate absent telemetry with zeros
and update deltas at model-query frequency. Aggregate repair counts cannot
attribute native penalties to hosts/events.

## Proposed additive contract

`blue-harness-v1` feature names are fixed in `blue/harness/features.py`.
Missing tree inputs are NaN; their neural counterparts have explicit per-value
availability masks and training-only fitted scaling. Ages use completed actions
in simulation ticks normalized by the configured horizon. Public mission phase
is derived from the unchanged CC4 schedule, never simulator truth.

The trusted local model bundle records feature version/order, trained HGB,
separate-episode isotonic calibrator, benign-only Isolation Forest, training
normalization, benign score reference, dataset hashes/splits and runtime version.
Risk is calibrated supervised compromise probability; novelty is a separate
benign-reference percentile. Neither is observed verification or proof of clean.

Defaults: intervention threshold 80 ticks; RL residual amplitude 0.25;
ML heuristic risk bonus 0.25, novelty bonus 0. No risk-threshold masks. Actual
simulator-valid masks and pending-action semantics remain unchanged. Urgent
remediation and verification precede intervention/learner. Overshoot is logged.
Native rewards, termination and truncation are unchanged; V(s) is not an
action-comparison oracle. Privileged labels/events remain diagnostic files.

## Required branch changes and rollout

| Role | Change | Acceptance |
|---|---|---|
| Blue | opt-in features/scorer/policy/trainer/loader | finite neural inputs; episode-reset and candidate-order invariance; no label leakage; train/eval parity |
| Environment | review optional ML pins | isolated runtime works; no shared dependency replacement |
| Evaluation | review grouped metrics/event accounting | disjoint episode splits; native reward sum matches diagnostic events; all runs retained |

Review producer and diagnostic semantics, then migrate opt-in consumers.
Legacy checkpoints remain on their original path; new feature width requires
new training and a matching scorer hash. No rename/migration of simulator
agents, hosts or shared telemetry schemas. No external deployment is proposed.

## Validation and decision

Focused suite: 47 passed before one additional recorder test; updated harness
suite: 10 passed. Live two-episode PPO update and reload/evaluation completed.
Mechanism checks reproduced Lancer and original actor returns on 8241/8229.
Guarded original actor: -66/-113 versus original -1273/-1034. These are
diagnostic episodes, not held-out performance evidence. New ML heuristic did
not beat coverage-only heuristic on them. See role handoff for artifacts and
exact commands. Long training and broad performance claims remain untested.
