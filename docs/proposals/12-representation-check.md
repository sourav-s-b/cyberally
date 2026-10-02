# Proposal 12: Representation check (stalest-first equivalence)

- Status: tested (diagnostic, not a policy proposal)
- Date: 2026-10-01
- Commit: `faeee08` (corrected the false "cursor missing from obs" claim).

## Description

A prior doc claimed the round-robin cursor was absent from observations,
implying the teacher was unlearnable without new features. This diagnostic
tested that claim: implement `StalestFirstBaseline` (serve the legal host
maximizing time-since-last-analysis, i.e. exactly what the `ages` feature
encodes) and compare against cursor round-robin.

## Motivation / evidence

If the cursor carried unobservable information, no Blue-visible architecture
could match the teacher and the fix would be new features. If it is
recoverable, the blocker is architectural (how the policy compares slots).

## Experiment

- Original: scratch (not committed): `/tmp/opencode/claim_check.py` (since
  lost — scratch dir is not versioned).
- Reproduced 2026-10-02 with committed code: `StalestFirstBaseline` in
  `blue_baselines.py` (same remediation core, sweep serves the legal host
  with min `(analysed?, last_analysis, host_index)`), registry
  `stalest_first`, pool `stalest-20261002` (manifest committed),
  `blue_compare --baseline round_robin` MATCH.
- 8 seeds, 400 steps, native reward:

| policy | mean | std |
|---|---|---|
| round_robin (cursor) | -93.5 | 27.8 |
| stalest_first (from obs) | -93.5 | 27.8 |

- Action traces byte-identical: 1995 entries/episode, 0 mismatches on all
  8 of 7629/7630/7640/7701-7705 (stronger than the original 3-seed
  check). The cursor carries zero information the observation
  lacks. (Caveat: `ages` saturates at 1.0 for never-analysed and very stale
  hosts alike; equivalence held within 400-step episodes.)

## Pros

- Killed a false blocker with a one-hour experiment instead of a feature
  redesign.
- Redirected the program from "add inputs" to "fix the head" (02/03), which
  is the smaller, testable change.

## Cons

- Scratch-script risk retired 2026-10-02: the equality now rests on a
  committed manifest, not the lost `/tmp` script. As documentation risk:
  the old false claim still exists in earlier handoff text — this file and
  `faeee08` are the correction; do not cite the old claim.

## Verdict

Closed. Representation is sufficient; the teacher's advantage is fully
expressible in Blue-visible features. All further work targets architecture
and ordering, not inputs.
