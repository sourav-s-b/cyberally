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
  do not; forced Sleep is 100 % pending-action lockout (676/676 ticks, zero
  session losses, seed-7629 breakdown) and policy-dependent — 68.7 %
  masked-random vs 54.4 % round-robin — so it is a throughput ceiling, not an
  unfixable blocker. See `docs/current-state.md` for the tables.
- Blockers: none for branching — Environment, Red and Evaluation may branch from
  `main`, which now carries the fixed wrapper and `tests_blue/`. Open items: the
  action-space proposal is unaccepted. Torch unblocked 2026-09-30 (Blue-drafted,
  needs Environment review): uv-built `.venv-train/` holds torch 2.14.1+cpu on
  system Python 3.12 with sim pins identical to `.venv`;
  see `environment/requirements-train.txt`. uv cannot supply 3.10 here (App
  Control blocks it), so the 3.10 target needs an IT-approved install.
   EPyMARL `cbc38c09` vendored at `third_party/epymarl/` (gitignored, .pth into
   `.venv-train`; `smaclite` dead import stubbed, src unmodified). First real
   MAPPO run completed 2026-09-30 in their EpisodeRunner (see blue-session.md):
   loop proven, policy not yet a defender — 8-episode checkpoint scores like
   Sleep. Round-robin remains the bar.
- foundation-v2 landed on `blue` (unpushed): `per_agent_bounds=True` gives
  17/17/17/17/51 hosts, 53/155 actions, 170/510 obs, 1190 critic state, with
  identical host indexing and trajectories; default mode unchanged and still
  required for stock EPyMARL. Proposal step 1 of 3; consumers unmigrated. `TrueStateWrapper` is broken against this
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
- Next: (1) BLUE-04 explicit temporal/belief features — the 12k multi-seed
  run (2026-09-30, 31 episodes, clean seed cycling) scores Sleep-level to
  seed-brittle (final: -407/-2536/-2824 vs round-robin -85/-105/-123) with a
  diverging critic (3.8→61.8); volume alone will not win, and no longer
  same-config run before features + stabilization (lower lr, real
  test_interval). Seed-cycling + Linux rebuild committed as `cc9a5c2`
  (47+1/42 green); coordination note drafted at
  `docs/coordination/blue-training-seeds.md`; (2) hand Environment the
  per-agent bounds plus the EPyMARL vendor recipe (`third_party/`, .pth,
  smaclite stub) so the training venv is reproducible; (3) PR `blue/mappo-training`
  once the policy beats round-robin or a longer run is worth reviewing. PR #1
  (`blue` → `main`) is merged.
- BLUE-02 measured (seed 7629, fixed wrapper, hosts-with-Red-session @ step 200):
  Sleep 64 total / 54 root (return -60); masked-random 33 / 25 (-102);
  round-robin heuristic 38 / 11 (-65). Historical broken-wrapper targets
  (Sleep 60/43, random 48/35) are superseded — recomputed here, do not compare
  across wrapper versions. Masked-random used a single policy seed (0).
  Extended to seeds 7629/7630/7640 via `evaluate_policies()`: round-robin
  halves root compromise vs masked-random on every seed (25→11, 19→13, 15→9).
- Artifacts: first MAPPO checkpoints under `cage-challenge-4/results/` (gitignored)
  with per-run manifest.json + stats.json; `blue_train_mappo.py` (their runner,
  our config) and `blue_eval_mappo.py` (greedy checkpoint vs heuristics on equal
  seeds). No competitive policy yet — report the Sleep-level score honestly.
  Historical baseline numbers (Sleep 60, masked random 48 compromised hosts at
  step 200) remain unreproduced and must not be reported as fresh results.
