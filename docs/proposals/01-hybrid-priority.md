# Proposal 01: Hybrid rules + learned scan priority

- Status: drafted (code exists, untracked, unbenchmarked)
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

Not yet run. Planned: (1) parity proof `HybridBluePolicy(None)` ==
round-robin on >= 8 seeds; (2) lancer-style priority + punch-style
file-density>0.9 flag + UC-style persistent malicious-event flags, same
seeds; (3) tiny risk scorer on Blue-visible 17-feat vectors. Target: beat
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
