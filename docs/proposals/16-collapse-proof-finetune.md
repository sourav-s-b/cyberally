# Proposal 16: Collapse-proof BC-to-RL fine-tuning (diagnose → JSRL+AWAC → PEX)

- Status: proposed (P2 — after 14 routes the init)
- Date: 2026-10-02

## Description

Replace KL-anchored PPO (measured: slows divergence only) with updates
that cannot drift, change the DATA distribution with a frozen guide, and
— first — test whether the collapse is plasticity loss rather than
algorithm choice. Ordered rungs, each with a kill criterion. Start from
the attention-distill ckpt (best init), or the 14-winner if IQL beats it.

## Motivation / evidence

W1 in full: flat MAPPO (-205 → -1709), factorized MAPPO, KL anchor —
three update rules, same collapse. The survey says our fixed-KL result is
the expected one: a fixed coefficient under-constrains early (collapse
happens in the first dip) and over-constrains late (no improvement).
Ordered by guarantee strength:

- D0 plasticity diagnostics (D24–D30): dormant-neuron rate, feature rank,
  primacy-bias resets. HOURS of work. If dormant rate explodes at the
  divergence point, the collapse is capacity loss and ReDo/periodic resets
  may fix it without any new algorithm. Run before blaming updates.
- JSRL (D21): frozen lancer_v2 guide owns early-episode steps with a
  SHRINKING horizon; RL only decides late episodes. Orthogonal to loss
  choice (works with any learner), zero sim change — the guide is already
  in our registry. Guide bias (learner never sees early states) is the
  known cost; the shrinking horizon is the designed antidote.
- AWAC (D14/C18): off-policy critic + implicit advantage-weighted actor
  constraint — the literature's direct answer to "SAC-from-BC dips"; no
  hard constraint coefficient to tune, unlike our KL.
- PEX (D22): frozen base + separate expandable online head with
  value-based selection — divergence becomes ARCHITECTURALLY impossible
  (worst case: selector never picks the online head). Most code, strongest
  guarantee; deploy if AWAC still dips.
- Fallback: kickstarting with DECAYING teacher weight + PBT schedule
  (D1) — the direct upgrade of our fixed-KL attempt, not a new idea.

## Experiment

1. D0 (days): instrument dormant-neuron fraction + feature rank during a
   short fine-tune; try ReDo recycling and last-layer resets. Kill: no
   plasticity signal → skip to rungs, do not build plasticity tooling.
2. JSRL+AWAC (1–2 weeks): lancer_v2 guide, initial guide horizon ~300/400
   shrinking on competence; AWAC learner from attention ckpt; existing
   stop-rules (halt on divergence past threshold; pristine 8201+ untouched).
3. PEX (2 weeks, conditional): frozen attention base + online factorized
   head + value selector. Kill AWAC first — PEX is insurance, not the
   opening.
4. Gate (binding): beat lancer_v2 on held-out (-91.2). A non-diverged run
   that merely ties is PROGRESS (first RL run that survives) but not a
   win — record as stabilization milestone, proceed to 17 for the
   improvement signal.

## Pros

- Every rung is a response to a MEASURED failure, not a guess: fixed-KL →
  decay schedule; BC dip → AWAC/PEX; early collapse → JSRL; unknown cause
  → D0 diagnostics first.
- JSRL and D0 need no new learner code; AWAC reuses the off-policy critic
  pattern our stack already understands.
- PEX bounds the downside: the program cannot lose the attention-parity
  policy to another divergence.

## Cons

- JSRL inherits guide blindness: if the improvement over lancer_v2 lives
  in early-episode decisions, the curriculum hides exactly those states
  until late. Watch per-phase returns, not just totals.
- AWAC's off-policy critic under 41%-dense reward needs reward-scale care
  (PopArt/normalization per E12); budget the audit.
- PEX doubles actor code and adds a selector to tune; only pay if AWAC
  dips.
- If 14 shows the ceiling is data, fine-tuning ANY ckpt harder is motion
  without progress — 14's verdict gates 16's ambition (stabilization vs
  win).

## Verdict

PROPOSED, P2. D0 + JSRL+AWAC first; PEX conditional; kickstarting-decay as
fallback. A surviving run reframes 17 from "fix learning" to "improve
learning".
