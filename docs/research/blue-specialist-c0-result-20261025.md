# Specialist C0 result: no second specialist selected

**Date:** 2026-10-25
**Frozen source:** `blue/finite-red-check` commit
`0dffc4afdc24964c605d3232525ae4e689c73f6e`
**Configuration:** `blue/configs/specialist-c0-audit-20261018.json`
**Development block:** Finite Red, seeds 8287–8300, one 400-tick episode per
seed.

## Result

All 14 episodes completed. The frozen C0 screen found:

| Candidate branch | Observed | Follow-up threshold | Result |
|---|---:|---:|---|
| Coverage misses | 0 episodes | 4 | Does not qualify |
| Failed recovery with multiple eligible choices | 0 episodes | 4 | Does not qualify |
| Unresolved verification with multiple eligible hosts | 0 events in 0 episodes | 32 events in 7 episodes | Does not qualify |

The policy investigated every owned host in all 14 episodes. The branch action
counts were 8,536 scorer investigations, 1,604 age-guard investigations, 1,040
recovery decisions, 1,658 verification decisions, and 15,092 pending waits.
These are workload counts, not evidence that a branch needs a learned policy.

**Decision:** no second specialist is selected from this study. It does not
prove that the rules are optimal or rule out a failure under Discovery or a
different development block. It says this frozen Finite screen did not find the
predefined repeated failure plus actionable alternatives needed to justify
recovery, verification, or coverage training.

## Training consequence

Keep the current response, verification, coverage, and safety rules fixed. The
next RL experiment targets the existing investigation hook, where learned
policies already have a concrete role and finite-Red BC has a confirmed gain.
Train that hook on a balanced Finite/Discovery episode mixture, without giving
the Red identity to the actor. Evaluate both regimes separately. Do not claim a
general Red improvement unless the frozen cross-regime gate passes.

The raw report and traces were written outside Git under
`/home/sourav/projects/cyberally/worktrees/margin/blue/results/blue-specialist-c0-audit-20261018/`.
The report records the config/source hashes and confirms that no privileged
state was used in the policy or trace.
