# V2 risk opportunity audit on reused development episodes

This is a mechanism audit of two existing reference MAPPO checkpoints. It does
not estimate defender improvement or change the policy. The frozen settings are
in `blue/results/gpt_harness/v2_sensitivity/high_risk_audit/frozen_config.json`:
reference_s0 and reference_s1, episodes 8233/8236/8241/8229, 400-step horizon,
sampled policy RNG `episode*997`, original risk scorer, and a diagnostic
high-risk threshold of **0.8**. Final seeds 7809–8200 were not used.

The instrumented replays validated all eight saved native returns and every
ordered tick/agent/action/host request against the existing `cells.json` hashes
(399 steps and 1,995 requests each). The scorer and checkpoint hashes are
frozen; the loader checked source/runtime contracts. The original interactive
process was interrupted after two cells; the remaining cells were resumed
using the same replay implementation. Incomplete JSONL files were preserved
with `.interrupted` suffixes. `resume.json` records this explicitly.

## What Blue did when a high-risk host was available

Here an *opportunity* means a sweep decision where at least one legal candidate
had Blue-visible calibrated compromise risk >=0.8. Counts are repeated
decisions, not independent hosts or episodes. `selected` means the sampled
policy chose **a** high-risk candidate at that decision. The age guard and
urgent rules can act before the hook; those decisions are outside this table.

| Episode | seed 0: opportunities / selected | seed 1: opportunities / selected | Distinct high-risk hosts (s0/s1) | First skipped, later confirmed with >=20-tick wait (s0/s1) |
|---|---:|---:|---:|---:|
| 8233 | 223 / 39 | 232 / 48 | 33 / 33 | 11 / 11 |
| 8236 | 112 / 33 | 114 / 33 | 33 / 34 | 12 / 12 |
| 8241 | 177 / 32 | 177 / 32 | 37 / 37 | 17 / 17 |
| 8229 | 252 / 41 | 244 / 42 | 32 / 35 | 11 / 11 |

At a high-risk opportunity the Lancer teacher itself selected a high-risk
candidate 31–46 times per episode/checkpoint, close to the trained policy's
32–48. The learner is therefore not consistently moving the chosen host
toward a high risk candidate. This is an observed choice pattern, not proof
that risk ought to override the teacher: several high-risk hosts can compete,
and the scorer's calibration is based on few independent episodes.

Every high-risk candidate in this audit had Blue tracker belief `UNKNOWN`.
For 29–34 of the first high-risk opportunities per cell, the actor chose
another host. Among these, 11–17 distinct hosts per cell were later
Blue-confirmed by a completed Analyse after a wait of at least 20 ticks.
One example on reference_s0, episode 8233: operational B user host 1 first
appeared at >=0.8 on tick 22, but its next completed Analyse and first
CONFIRMED transition were at tick 131. Episode 8241 has a host with a
121-tick gap. These are concrete delays after a usable *score* was present.
The audit cannot show that an Analyse on the earlier tick would have exposed
the same evidence; that requires a one-choice mechanism check.

Privileged Red-session labels are written only to separate diagnostic JSONL
files. For 27–35 of 32–37 distinct high-risk hosts per cell, a Red-session
label already existed at the first high-risk sweep opportunity. Every
>=20-tick delayed, later-confirmed host in the table had such a label.
These labels describe simulator session state, not Blue-visible evidence or
guaranteed detectability, and they never entered the actor or scorer online.

## Why this is not a performance result

The complete native reward ledger is revealing. Episodes 8233 and 8241 had
action costs only. The non-cost penalties in 8236 (-35) and 8229 (-95) were
recorded on contractor-network hosts; none overlapped the Blue-owned
high-risk sweep candidates. Earlier host detection might change later Red
behavior and thus native return, but this trace does not attribute those
losses to any missed high-risk choice. Choosing more hosts can also incur
more action costs. The eight cells are four reused simulator episodes crossed
with two training seeds, not eight independent attack scenarios.

This supports a narrower conclusion than "ML should force remediation":
the risk input often flags a host that the sampled investigation policy does
not promptly inspect, and later investigation sometimes confirms it. The
hook can reorder only sweep Analyse targets; Remove/Restore remain forced by
the existing Blue-confirmed remediation rules.

The next bounded mechanism check should change **one** investigation request
at a first high-risk opportunity on an already-used episode where the host
was later confirmed. It should log completion/detection, native action cost,
penalty events and the first Red/Green RNG divergence. This is a diagnostic
intervention, not a new policy or performance claim. A general >=0.8 override
would be premature: many hosts cross the threshold and may overwhelm the
sweep. Any future priority rule should use observed event recency and an
explicit capacity bound, then be tested against guard-only and Lancer.

Artifacts: `blue/results/gpt_harness/v2_sensitivity/high_risk_audit/report.json`
and each cell's `.decisions.json`, `.blue.jsonl` and `.privileged.jsonl`.
Blue requests/completions and privileged session/penalty records are separate.
The scripts are `blue/training/harness_high_risk_audit.py` and
`blue/training/harness_high_risk_resume.py`. All eight reward ledgers sum to
the native return at each tick.
