# Proposal 01: Hybrid rules + learned scan priority

- Status: drafted — parity proven, priority scorer not yet built
- Draft: `cage-challenge-4/blue_hybrid.py` (`HybridBluePolicy`)
- Date: 2026-10-01 / reviewed 2026-10-02

## Description

Split the policy in two. Fixed, unviolable rules handle what rules do best:
CONFIRMED -> Remove (escalate to Restore on re-detection), VERIFY ->
re-Analyse, oldest remediation first. A learned per-host priority scorer
(`priority_fn(env, agent, host) -> float`) decides only the scan order —
which host to Analyse next. `priority_fn=None` degrades to host order, so the
learnable part is measurable in isolation against round-robin (-93.5 ± 27.8).

## Motivation / evidence

- CAGE-4 winner family, all three (Kiely et al., AAAI 2025): UC's
  flag-driven remediate + Analyse-longest-stale; lancer's round-robin +
  per-host float priority (decay on touch, boost on Monitor) — the closest
  published analog of `priority_fn`; punch's analyse-restore round-robin,
  "much better than our RL agents".
- H-MARL on CC4 (Singh et al., AAMAS'25): expert-rule master (IOCs ->
  Recover) -129.53 beats flat IPPO -181.62 (+52). Rules-first wins measured.
- Our `MASK_RULES`/`BlueZoneTracker` already encode CONFIRMED/VERIFY states;
  the hybrid only adds the ordering head. Privileged labels allowed as
  *training targets* for the risk scorer, never as inputs (contracts).

## Experiment

PARITY PROVEN 2026-10-02 (this change: `priority_fn=None` now runs a cursor
round-robin identical to `RoundRobinBaseline`; the first-draft `cands[0]`
would have stuck on one host, so it was fixed before measuring).

- Regression: `cage-challenge-4/tests_blue/test_hybrid_parity.py` — 2 seeds,
  asserts equal native returns AND identical per-tick traces. Passes (69 s).
- Full manifest, 8 seeds x 400 steps, native reward (scratch
  `/tmp/opencode/parity_8seed.py`, results/ is gitignored so numbers live
  here):

| seed | round_robin | hybrid_none | traces identical |
|---|---|---|---|
| 7629 | -85 | -85 | yes (1995/1995) |
| 7630 | -105 | -105 | yes |
| 7640 | -123 | -123 | yes |
| 7701 | -85 | -85 | yes |
| 7702 | -135 | -135 | yes |
| 7703 | -64 | -64 | yes |
| 7704 | -53 | -53 | yes |
| 7705 | -98 | -98 | yes |

Mean -93.5 both. Sim suite after change: 82 passed, 5 skipped (was 81/5).

Remaining, not yet run: (1) lancer-style priority + punch-style
file-density>0.9 flag + UC-style persistent malicious-event flags, same
seeds; (2) tiny risk scorer on Blue-visible 17-feat vectors. Target: beat
-93.5, approach lancer's constant-size -71 band on equal footing.

## Pros

- Smallest change that can beat the teacher: keeps everything the teacher
  does right, learns only the ordering it does arbitrarily.
- Learning is safely bounded: the scorer can never trigger a wrong
  remediation (rules gate all Remove/Restore).
- Each piece is ablatable (rules alone, +priority, +flags, +scorer).
- Directly validated by 3/3 competition winners + H-MARL Expert.

## Cons

- Ceiling: if optimal play needs strategy the rules forbid (e.g. proactive
  firewall play per UC, decoy play per lancer), the hybrid cannot reach it
  without rule changes — rules become the new bottleneck.
- The risk scorer trains on sparse, delayed labels; small-model discipline
  needed to avoid re-learning round-robin with extra steps.
- Draft is uncommitted and has no tests yet.

## Verdict

BUILD FIRST. Concrete next task: parity proof, then priority + flags,
>= 8-seed native evals throughout.
