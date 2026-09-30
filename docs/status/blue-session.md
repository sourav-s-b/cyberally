# Blue session update

**Last updated:** 2026-09-30

**Owner:** Sourav

**Working branch:** `blue/foundation` at `f8a9deb`
**Base commit:** `e605ef3` (`chore: import CAGE4 prototype and establish team workflow`)
**Merged in:** `origin/main` at `f8a9deb` — `blue/foundation` was fast-forwarded
into `main`, so both refs are identical and Environment, Red and Evaluation can
branch from `main`.
**Current state:** BLUE-01 foundation is complete as `7b76fc2` plus `58b6cc7`,
and this session added the measured-behaviour tables and the action-space
contract proposal in `f8a9deb`. Next task is BLUE-02 heuristic baselines.

Read `docs/current-state.md` before this file. It carries the measured facts and
the trap list that everything below depends on.

## Goal

Build a reliable CAGE4 Blue defender, establish strong baselines, then train and
evaluate MAPPO and attacker adaptation. Read root `AGENTS.md`,
`docs/implementation-plan.md`, `docs/contracts.md`,
`cage-challenge-4/blue-agent-plan.md`, and the correction at the top of
`cage-challenge-4/handoff.md` before changing code.

## Current Blue changes

Committed in `7b76fc2` and `58b6cc7` on `blue/foundation`:

- Reworked `cage-challenge-4/cc4_epymarl_wrapper.py`: reset the shared simulator
  once, apply explicit seeds and advance the RNG when none is supplied, reject
  host overflow instead of truncating, track exactly one pending action per
  agent, merge observations for every visible host every tick, keep simulator
  validity masks separate from opt-in evidence masks, and distinguish native
  termination from horizon truncation.
- Capacity raised to the scenario-derived 51 hosts. At seed 7629 agent 4 sees 38
  hosts, so the current dimensions are 510 local observation features, 155
  discrete actions, 2550 concatenated state.
- `CC4BlueWrapper` is a single-agent facade over the same joint-step
  implementation rather than a parallel copy of the bookkeeping.
- Reworked `cage-challenge-4/blue_action_masking.py`: pending actions no longer
  overwrite belief, `Restore` stays reachable in validity mode, and successful
  remediation transitions to `VERIFY` instead of declaring the host clean.
- Added `cage-challenge-4/tests_blue/test_foundation.py`.
- Added live tests for lost target sessions (Analyse/Remove/Restore returning
  `FALSE` when the child session disappears mid-action, with the original target
  still attributed and no clean label inferred), overdue-action failure, reset
  clearing unresolved pending state, and passive Monitor event replacement with
  timestamp advancement.

Drafted this session, not committed to any contract yet:

- `docs/current-state.md` — measured facts, branch topology, trap list.
- `docs/coordination/blue-action-space.md` — per-agent host bounds proposal.

## Verification already completed

- From `cage-challenge-4`: `..\.venv\Scripts\python.exe -m pytest -q tests_blue`
  — **20 passed** (12.7 s).
- Three masked-random episodes (seeds 42, 7629, 7630), 75 steps each —
  completed; per-agent common rewards matched and action masks held.
- `.venv\Scripts\python.exe scripts\check_source.py` from repo root — 163
  tracked Python files parsed.
- Measured this session, masked-random policy, seed 7629 unless noted:
  - 44 ms per joint step, 0.19 s reset, ~18 s per 400-step episode.
  - True-state Red sessions via `get_true_state(info).data`, host-wide: 1 at t0,
    13 user + 21 root at step 100, 17 + 27 at step 200, 18 + 15 at step 300.
  - `host_ground_truth()` over `blue_agent_4`'s own observation view: 0 for every
    host at every one of 399 steps, across seeds 7629, 7630 and 7640. Note the
    true-state count is host-wide and the Blue-view count covers agent 4's hosts
    only, so they are not a like-for-like ratio.
  - 162 of 399 steps carry nonzero reward; the first ~135 do not. Mission phases
    are `(134, 133, 133)`; phase 0 is pre-planning.
  - 1374 of 1995 agent-ticks had exactly one legal action (Sleep) — 69 %.
  - Live nonzero features within a single agent's observation: 22–39 %.
  - Per-agent host counts across seeds 7629–7660: agents 0–3 hold 7–14 hosts
    with exactly 1 router each; agent 4 holds 29–38 with 3 routers. Scenario
    bound is 17 per subnet, 51 for agent 4.
  - `CybORG/Agents/Wrappers/TrueStateWrapper.py:62` raises `AttributeError`
    against this snapshot because `get_true_state(info)` returns an
    `Observation`, not a dict; `CybORG.Agents.Wrappers` also imports Ray.
- Not done: no MAPPO training, no EPyMARL integration, no heuristic benchmark,
  no historical baseline reproduction, no upstream test-suite run.

## Next steps

1. Python 3.10 and the venv rebuild with Environment (ENV-01), splitting a
   simulator-only profile from a training profile that adds torch. torch is
   absent today, so no training is possible. This is now the top blocker.
2. Ask Environment to confirm the per-agent host bounds of 17 and 51 before
   they write any config, and get Evaluation's agreement on the `VERIFY`
   resolution rule using Blue-visible evidence only.
3. Investigate whole-agent primary-session loss: native automatic Monitor
   dereferences session 0, so this needs a simulator-level decision.
4. BLUE-02: build Sleep, built-in random, masked-random and round-robin
   Analyse/Restore baselines; reproduce the step-200 seed-7629 numbers.
5. BLUE-03 only after step 4: implement EPyMARL's actual constructor,
   tensor-action, reward and lifecycle interfaces. No long run before a complete
   rollout-plus-optimizer-update smoke test.

## Resume prompt for a new conversation

> Continue the Blue defender work in `G:\Projects\cyberally`. Read `AGENTS.md`,
> `docs/current-state.md`, `docs/status/blue-session.md`, `docs/status/blue.md`,
> `docs/contracts.md`, `docs/implementation-plan.md`,
> `docs/coordination/blue-action-space.md`, `cage-challenge-4/blue-agent-plan.md`,
> and the correction at the top of `cage-challenge-4/handoff.md`. Preserve my
> work: inspect `git status --short --branch` and fetch origin first. BLUE-01 is
> `7b76fc2` plus `58b6cc7`, documented in `f8a9deb`, and **`blue/foundation` has
> been fast-forwarded into `main` — both refs are at `f8a9deb` and identical**;
> do not reset or rewrite shared history. Next task is Environment's Python 3.10
> runtime (ENV-01), then BLUE-02 heuristic baselines. Do not start MAPPO
> training before a complete optimizer-update smoke test, and never report an
> unreproduced number as a result.

## Update rule

At each meaningful Blue handoff, replace "Current Blue changes," "Verification"
and "Next steps" with the latest state. Include branch and base commit, whether
changes are staged/committed/pushed, exact commands and results, open contracts
and the next task. Update `docs/status/blue.md` as the short index. Never report
a run that did not actually complete.
