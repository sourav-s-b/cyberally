# Blue handoff

**Latest detailed session state and resume prompt:** [blue-session.md](blue-session.md).
Read [docs/current-state.md](../current-state.md) first — it holds the measured
facts and the list of traps. Update this file at each meaningful Blue handoff;
this page is its short index.

- Owner: Sourav
- First branch: `blue/foundation`
- Status: BLUE-01 implemented, tested and **merged into `main`**. Both
  `main` and `blue/foundation` are at `f8a9deb`. Action-space proposal drafted.
- Completed: repaired joint reset/seeding, host capacity (raises instead of
  truncating), agent-wide pending actions, passive observation merging,
  validity/evidence mask separation, `VERIFY` state after remediation, episode
  boundaries, and a single-agent facade over the joint implementation. Added a
  live regression suite. Documented per-agent host bounds (17 / 51).
- Validation: isolated Python 3.12.10 venv with NumPy 1.26.4, Gym 0.26.2,
  NetworkX 3.2.1, pytest 8.0.0. **20 focused tests pass.** Three 75-step
  masked-random episodes pass. Measured directly this session: 44 ms/joint step;
  true-state Red reaches 34 sessions by step 100 at seed 7629 while Blue's own
  view shows 0 the whole episode; 41 % of steps carry reward and the first ~135
  do not; 69 % of agent-ticks are forced to Sleep by the agent-wide pending
  lockout. See `docs/current-state.md` for the tables.
- Blockers: none for branching — Environment, Red and Evaluation may branch from
  `main`, which now carries the fixed wrapper and `tests_blue/`. Open items: no
  pinned runtime (torch absent, venv is 3.12 not the 3.10 target), and the
  action-space proposal is unaccepted. `TrueStateWrapper` is broken against this
  CAGE4 snapshot, so the privileged-label path for Evaluation must be built from
  `get_true_state(info).data`. Upstream `test_blue_actions.py` cannot collect
  because its conftest imports Ray.
- Dependency commits/PRs: bootstrap `e605ef3`; BLUE-01 fixes `7b76fc2` and
  `58b6cc7`; documentation and measured-state commit `f8a9deb`, fast-forwarded
  into `main`. No Environment, Red or Evaluation branch or PR exists as of
  2026-09-30.
- Contract changes: **drafted, not accepted** —
  [`docs/coordination/blue-action-space.md`](../coordination/blue-action-space.md)
  proposes per-agent host bounds (17 for agents 0–3, 51 for agent 4), which
  moves obs to 170/510, actions to 53/155 and critic state from 2550 to 1190.
  Today 111–126 of 155 action slots are permanently masked for agents 0–3.
  Also unresolved: how a `VERIFY` state resolves from Blue-visible evidence,
  and whole-agent primary-session loss, where native Monitor dereferences
  session 0 and needs a simulator-level decision from Environment.
- Next: (1) get the BLUE-01 PR reviewed and merged; (2) hand Environment the
  per-agent bounds so their configs are not written against 155/510/2550;
  (3) BLUE-02 Sleep/random/masked-random/round-robin heuristic baselines and
  reproduce the historical step-200 numbers on seed 7629; (4) only then wire
  EPyMARL's real constructor, tensor-action and lifecycle interfaces, behind a
  one-rollout-plus-one-optimizer-update smoke test.
- Artifacts: **no trained policy, no checkpoint, no EPyMARL registration.**
  Historical baseline numbers (Sleep 60, masked random 48 compromised hosts at
  step 200) remain unreproduced and must not be reported as fresh results.
