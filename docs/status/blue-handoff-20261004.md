# Blue handoff — metric repair, max-age guard evaluation, and the AdaptiveBlue pilot plan

- Date: 2026-10-04 (evening). Supersedes nothing; append new sessions elsewhere and
  retire this file once P0 lands.
- Branch at time of writing: `blue/mappo-training`, HEAD `2900711`.
- Working tree: **dirty, nothing committed**. 5 modified + 10 untracked paths (see
  "Uncommitted inventory"). There is **no commit** for any of the work below.
- Status: **no agent implementation, no training, no checkpoint exists yet.**
  The previous session produced instrumentation and a negative guard result.
  This document is the handoff for the next session, which starts at P0.
- Read first: `AGENTS.md`, `docs/current-state.md`, `docs/status/blue.md`, and
  `docs/proposals/manifests/guard-maxage-matrix-20261004.json`.

---

## 1. TL;DR for the next session

The previous session built a completion-based evaluation harness, discovered
several metric defects in the older harness, ran an 8-arm x 32-seed guard
matrix, and concluded "do not run the residual pilot." That conclusion was
**overstated in five specific ways**, two external reviews caught it, and the
corrections are now decided. The work to do next is:

1. Repair the wrong claims in the manifests and status docs (P0).
2. Write two coordination proposals (P1).
3. Package `AdaptiveBluePolicy` and fix one dead RNG parameter (P2).
4. Run five named baselines (P3).
5. Run **one** bounded unguarded residual-PPO pilot on development seeds, save
   the checkpoint and decision traces (P4).
6. Size a frozen evaluation from *measured* variance (P5), gated on a seed
   allocation that another role must approve.

**Do not re-plan any of this.** The statistical design, the decision rules, the
control set and the sample-size method are all settled below. If a number in
this document disagrees with a manifest, the manifest is stale and P0 fixes it.

---

## 2. What the evidence actually says

All numbers below were recomputed from `blue/results/matrix-a/cells.jsonl`
(256 cells = 8 arms x 32 seeds, 400 steps). That file is **gitignored** and
exists only on this machine, which is why P0 commits a compact copy.

### 2.1 Return and guard behaviour

| arm | mean return | coverage (mean) | never-investigated (mean) | mean of per-episode max age | global max age |
|---|---:|---:|---:|---:|---:|
| `unguarded_lancer` | -79.41 | 1.000 | 0.00 | 274.3 | 370 |
| `unguarded_oracle` | -63.19 | 0.284 | 54.97 | 371.0 | 397 |
| `strict_lancer` | -82.53 | 1.000 | 0.00 | 179.1 | 277 |
| `strict_oracle` | -85.66 | 1.000 | 0.00 | 172.9 | 324 |
| `maxage24_lancer` | -83.25 | 1.000 | 0.00 | 144.1 | 206 |
| `maxage48_lancer` | -82.75 | 1.000 | 0.00 | 148.2 | 217 |
| `maxage48_oracle` | -84.72 | 1.000 | 0.00 | 133.3 | 186 |
| `maxage96_lancer` | -80.69 | 1.000 | 0.00 | 159.8 | 238 |

Paired differences vs `unguarded_lancer` (2000-sample paired bootstrap):
`strict_lancer` -3.12 [-10.25, +4.06]; `maxage24` -3.84 [-11.88, +3.53];
`maxage48` -3.34 [-9.81, +3.00]; `maxage96` -1.28 [-6.66, +2.88];
`maxage48_oracle` -5.31 [-16.19, +5.12]; `unguarded_oracle` +16.22 [+6.66, +25.56].

Within-regime oracle-vs-lancer: unguarded +16.22 [+6.66, +25.56];
strict -3.12 [-12.16, +6.00]; max-age 48 -1.97 [-12.09, +7.22].

### 2.2 The four facts that drive the whole plan

**(a) `unguarded_lancer` already has coverage 1.000 and zero never-investigated
hosts in all 32 seeds.** Every guarded arm also has 1.000. So coverage was never
the binding constraint, and the max-age guard bought **freshness, not coverage**.
The guard family optimised an invariant that was already satisfied. This is why
coverage is demoted to a monitored metric in the pilot.

**(b) The host universe is seed-dependent: 57 to 96 defendable hosts, mean 76.3.**
Identical across arms for a given seed. Any sentence with a fixed "of 66" or
"of 87" denominator is wrong; 66 was one seed, 87 was another.

**(c) The oracle is not an upper bound, and its own age metrics are worse.**
`unguarded_oracle` has mean-of-maxima 371.0 versus the unguarded lancer's 274.3.
A policy that is worse on one axis does not dominate another. Combined with the
fact that `OracleOnset` is a greedy per-host onset-likelihood scorer that ignores
action cost and timing, treating it as a ceiling is simply invalid.

**(d) The attack schedule is not policy-invariant, but that coupling explains
little variance.** Initial compromise is shared: `n_initially_compromised_episodes`
is identical across all 8 arms in **32/32** seeds. Attack evolution is not:
`n_compromise_episodes` differs across arms in **32/32** seeds with mean spread
**28.31** events, and the compromised host *sets* differ in **224/224** arm-seed
comparisons (median 35 onset-tick mismatches among common hosts). Mechanism:
`FiniteStateRedAgent._choose_host` selects by host-state priority, so blue's
remediation changes which hosts red attacks next. But
`r(return_diff, Δcompromise)` is only -0.265 to -0.488 (r² 0.07-0.24; 0.09 for
the headline `maxage48` contrast).

---

## 3. Claims that are wrong and must be corrected in P0

These are the specific overstatements. Each is a required P0 edit.

| # | Wrong claim (in current manifests/docs) | Correct statement | Why |
|---|---|---|---|
| 1 | "the privileged oracle is the upper bound on ordering choices" | the tested privileged scorer improves unguarded return but does not demonstrate improvement under the tested guards | greedy, ignores cost/timing; its own age metrics are worse (§2.2c) |
| 2 | gate evidence "indistinguishable from zero ... no measured ceiling", verdict framed as absent headroom | no **detectable** benefit in the arms tested, at a design resolving only ~7-16; upper bounds +6.00/+7.22 do **not** exclude a +5 effect | a null at n=32 cannot establish absence |
| 3 | `learned - lancer` demoted as "confounded / not attributable" | it is the **primary deployment comparison**; the scheduler control is the secondary attribution contrast | "confounded" is true about attribution, not about validity; replacing the incumbent is a total-system question |
| 4 | "worst-case max age falls 274 -> 148" | means of per-episode maxima fall 274.3 -> 148.2; the global worst falls 370 -> 217 | 274 is a mean, not a maximum |
| 5 | "abandoning 55 of 66 hosts" and coverage 0.284 as a fixed-denominator claim | 54.97 of a mean 76.3, per-seed 57-96; the figure is an **association** with the return gain, not a demonstrated cause | the two numbers do reconcile per-seed; the fixed denominator and the causal reading were both wrong |
| 6 | CRN "can only reduce SD by 5%" | deferred on **cost and uncertain benefit**; the r² above describes these 32 observations, and is not a bound on a different coupling | a correlation in observed data does not bound a counterfactual design |

Also record, as a design finding rather than a correction: **`unguarded_lancer`
coverage was already 1.000**, so the guard's benefit was freshness (§2.2a).

---

## 4. Statistical design (settled — do not re-derive)

### 4.1 Primary comparison

**Unadjusted paired difference.** `Δᵢ = R_learned,i − R_lancer,i`, computed per
seed on a frozen seed list.

**Compromise counts are outcomes, never adjustment covariates.** They are
post-treatment: preventing compromises *is* the benefit under test, so adjusting
for them subtracts the treatment effect. CUPED requires a covariate unaffected by
treatment. Measured justification for rejecting the covariate route, from the
existing 32 seeds:

| contrast | r² vs `n_defendable_hosts` (valid) | r² vs `n_initially_compromised` (valid) | r² vs Δcompromise (**inadmissible**) |
|---|---:|---:|---:|
| `strict_lancer` | 0.043 | 0.007 | 0.070 |
| `maxage48_lancer` | 0.004 | 0.071 | 0.090 |
| `maxage96_lancer` | 0.005 | 0.000 | 0.133 |
| `maxage48_oracle` | 0.007 | 0.045 | 0.238 |

Note the trap this exposes: the only covariate that would have "worked" is the
inadmissible one, precisely because it absorbs the treatment effect. And the
valid covariates explain essentially nothing (r² ≤ 0.071), so **valid CUPED buys
≤ ~3.5% of SD here.** Combined with CRN being deferred, **seed count is the only
remaining variance lever.**

### 4.2 Decision rules (evaluate in this order)

Pilot advancement, evaluated on the frozen evaluation interval:

1. **Regression** — upper 95% bound < 0 → report evidence of regression.
2. **Advance to replication** — unadjusted paired mean ≥ +5, its 95% CI excludes
   zero, and no predefined failure check tripped.
3. **Minimum worthwhile gain unsupported** — upper bound < +5 → "the evaluation
   does not support the minimum worthwhile gain."
4. **Inconclusive** — otherwise. Stop at the predefined compute budget. **Do not
   automatically launch more tuning or variance-reduction work.**

Adoption, after replication and a frozen final evaluation: require the **lower
confidence bound > +5**.

These rules apply to whatever interval was obtained. **A smaller experiment is
not automatically inconclusive** — a reduced-n run can still show a large gain or
a clear regression. Do not write "any reduced-sample verdict is inconclusive."

### 4.3 Sample size

Use 80% power, two-sided α = 0.05:

```
n = ceil( (2.80 * s_delta / (delta - delta0))^2 )
```

where `s_delta` is the paired SD of `R_learned − R_lancer`, `delta` the assumed
true gain, `delta0` the tested threshold. **Do not use 1.96 alone** — that sizes
the *expected* lower bound at the threshold, i.e. ~50% power.

Grid at the provisional `s_delta = 18.7` (measured from `maxage48_lancer` vs
`unguarded_lancer`, **not** from a learner — treat as a budget proxy only):

| tested threshold | δ=+8 | +10 | +12 | +15 | +20 |
|---|---:|---:|---:|---:|---:|
| lower bound > 0 | 43 | 28 | 20 | 13 | 7 |
| lower bound > +5 | 305 | 110 | 56 | 28 | 13 |

At the observed 6.8 s/cell, n=110 is ~12.5 min per arm.

**Accepted conservatism:** branch 2 requires the sample mean to reach +5 *and*
the interval to exclude zero. Both are functions of the sample mean, so it
reduces to `P(mean ≥ max(+5, 1.96·SE))`. At a true gain of exactly +5 the advance
probability is **0.500 at n=110** and **0.418 at n=43**; the binding constraint
flips from the sample-mean requirement to the interval requirement near n≈103.
So the pilot screen misses about half of genuinely worthwhile +5 recipes. That is
the intended price of conservative screening and must be stated in the
pre-registration rather than treated as a defect.

### 4.4 Freeze order (prevents optional stopping)

1. Estimate variance on **development** data with the frozen checkpoint.
2. **Freeze** checkpoint, evaluation seed list, sample size and stopping rules.
3. Only then read any paired difference.
4. Do not enlarge `n` until an interval passes. Final evaluation stays separate
   from checkpoint selection.

Implement this as a `freeze_eval(seed_list, n)` contract on the policy/eval
harness so it is structural, not a promise.

---

## 5. The pilot design

**Objective:** unmodified native episode return. Guard **off**. Coverage,
investigation age, missed incidents and remediation are **monitored, not
penalised** — coverage below 1.0 is recorded, not scored as failure.

**Training:** ≤ 80 episodes, development seeds only, one fixed recipe. Stop on
implementation failure or predefined severe degradation.

**Controls (P3), five distinct behaviours, all on the 32 dev seeds:**

| name | definition |
|---|---|
| `lancer` | deployed baseline, current `LancerValues`, deterministic |
| `sleep` | no investigation action |
| `lancer_rules_uniform_investigation` | lancer's decoy and urgent-response rules preserved on every step; only the sweep-stage investigation-host choice is uniform over legal candidates |
| `lancer_rules_zero_residual_sampling` | same response rules; sweep-stage choice sampled from `softmax(base_scores)` |
| `random_legal_actions` | uniform over all legal **defensive** actions; include only if project rules require it beyond the first three |

These are genuinely different and must not be collapsed. Two traps:

- **Zero residual is not uniform.** `blue/training/residual.py:103` computes
  `logits = (base + bonus*tanh(r)) * mask + (1-mask)*-1e9`, and `masked_choice`
  softmaxes exactly those logits. With `r ≡ 0` the distribution is
  `softmax(base_scores)` — lancer's own decoy/urgency values, *not* uniform. Only
  equal base scores give uniform.
- **Candidates are per-agent, not ~76.** `blue/policies/ordered.py:293`:
  `[h for h in env.hostnames[agent] if _legal(mask, env, agent, h, "Analyse")]`.
  76.3 is the mean whole-network defendable count; an actor's legal candidate set
  is its own zone's hosts minus already-served ones, and it shrinks over the
  episode.

`lancer_rules_zero_residual_sampling` with greedy argmax is a **parity
verification check**, not a sixth baseline: it must reproduce `lancer` exactly.

**Follow-on, gated on branch 2 only:** two further training seeds (replication),
then the retraining demonstration (Blue₀ → changed development attacker → Blue₁,
checking improvement and regressions).

---

## 6. Phases, with rationale

### P0 — claim repair and commit (branch `blue/metric-repair-maxage`)

Apply all six corrections in §3 to
`docs/proposals/manifests/guard-maxage-matrix-20261004.json`,
`docs/proposals/manifests/guard-maxage-20261004.json`,
`docs/status/blue.md` and `docs/status/blue-session.md`. Add the §2.2 facts as
supporting evidence, including that coverage was already 1.000 unguarded.

Also commit the 256 raw cells as a compact JSONL (arm, seed, return, metrics,
guard stats, trace hash, config, commit hash, seed list) under a small tracked
path, plus a pointer from the manifest. Rationale: `blue/results/` is gitignored,
so the only durable record of the run currently lives on one disk. Project rules
require committed artifact manifests and reproducibility metadata.

No behaviour changes in P0, so no test re-run is required beyond a docs diff
review.

### P1 — coordination proposals (`docs/coordination/`)

Use the existing `TEMPLATE.md` structure and **extend**
`docs/coordination/blue-training-seeds.md` rather than duplicating it.

1. **Seed block 7809–8200 for Blue evaluation only.** Consumed today: 7629–7729,
   7801–7808, 8201–8216. Reserved by others: 8301–8400 (finite Red),
   8401+ (ledger conflict), 8501+ (untouchable). Block 7809–8200 appears
   unclaimed in every role ledger. State the separation rules preserved: no
   training on it, other blocks untouched. **Why needed:** n=110 requires ~110
   fresh eval seeds and there is no other free block.
2. **Attack-level CRN request to Environment/Red** — recorded as *requested and
   recommended against*, on cost and uncertain benefit, with the §2.2(d) coupling
   facts as motivation. Rationale for keeping it on record: the decision is a
   trade-off someone else may disagree with, and a rejected-for-now request with
   numbers is better than a silent omission.

### P2 — `AdaptiveBluePolicy`

Baseline mode, learned mode, checkpoint loading, reset, decision logging. Guard
off but selectable. **`OrderedPolicy` must stay byte-identical** so the
deployment comparison remains apples-to-apples. Enable `record_decisions=True`.

Also two fixes found while preparing this plan:

- **Dead RNG parameter.** `masked_choice(logits, mask, sample=True,
  seed_rng=None)` at `blue/training/residual.py:62` accepts `seed_rng` and never
  uses it; `th.multinomial(probs, 1)` at line 70 reads the global torch RNG. Its
  only sampling caller is `blue/training/residual_pilot.py:107`. Fix minimally:
  thread an explicit `torch.Generator` through, add a repeatability test. Global
  seeding already exists at `residual_pilot.py:184` and **no previous run is
  invalidated** by this — it is a reproducibility improvement, not a redesign.
- **`freeze_eval(seed_list, n)`** per §4.4.

### P3 — five baselines (§5), noise floor recorded before any training.

### P4 — bounded pilot, dev seeds, ≤ 80 episodes, checkpoint and traces saved.

### P5 — variance estimation, then sized frozen evaluation. Gated on P1.1.

Size `n` from the *measured* learned-vs-lancer paired SD. Report power to detect
|Δ| ≥ +5 and ≥ +10 at the `n` actually used. If the seed block is not granted,
run at reduced n, apply §4.2 to whatever interval results, and report the power
achieved. Do not relabel a null as absent headroom.

### P6 / P7 — replication, then retraining demo. Only on branch 2.

---

## 7. Cross-role dependencies and blockers

- **Seed allocation (P1.1) blocks P5.** No other role has claimed 7809–8200, but
  shared seed allocation is a shared contract, so it needs a proposal, not a
  unilateral grab. P0–P4 need no new seeds and are not blocked.
- **Attack-level CRN (P1.2)** is a simulator change owned by Environment, with
  attacker semantics owned by Red. Not needed for this pilot.
- **Pre-existing seed-ledger conflict**, previously noted and still open:
  `blue.md` said "8201+ reserved" although 8201–8216 are consumed;
  `blue-session.md` claimed 8200–8420 free although 8301–8400 belong to finite
  Red. Both files now disclose the conflict rather than quietly disagreeing.
  P1.1 should resolve it as part of the block request.

---

## 8. Risks and things not to do

- Do **not** reinstate "oracle = upper bound", "no headroom", or "do not build the
  agent". Those are the specific overstatements this plan exists to undo.
- Do **not** adjust the primary return comparison for compromise counts.
- Do **not** size samples with 1.96 alone.
- Do **not** enlarge `n` until an interval passes.
- Do **not** declare reduced-n automatically inconclusive.
- Do **not** describe zero-residual sampling as uniform, or quote ~76 as a
  candidate count.
- Do **not** automatically launch more tuning or variance-reduction work after an
  inconclusive result.
- Do **not** commit model weights, logs, `blue/results/`, virtualenvs or
  credentials. Small configs and manifests only.
- Two external reviews contributed the corrections in §3 and §4. If a future
  analysis contradicts them, re-derive from the raw cells rather than trusting
  either the old docs or this summary.

---

## 9. Validation state at handoff

Run on the current dirty tree; re-run after P2 touches `residual.py`.

- Sim venv `.venv/bin/python -m pytest blue/tests_blue`: **178 passed, 11
  skipped** in 579 s. The skips are torch-gated modules, skipped as whole files
  (`--collect-only` confirms); no `pyflakes`/`ruff` in either venv.
- Train venv `.venv-train/bin/python -m pytest blue/tests_blue`: **227 passed,
  1 skipped** in 653 s; the skip is the pre-existing `test_rvs.py:73` "rvs_14
  ckpt not trained".
- Targeted: 23 metric-semantics, 30 recorder/metric live, 32 max-age guard, 17
  ordering parity, 16 residual audit, 4 legacy-ordering gate.
- `py_compile` clean on every touched file; `git diff --check` clean.
- No linter is installed in either venv. Unused imports were removed by hand
  after an `ast` scan; do not assume a lint gate exists in CI.

### Uncommitted inventory

Modified: `blue/analysis/ordering.py`, `blue/policies/ordered.py`,
`blue/training/residual_pilot.py`, `docs/status/blue-session.md`,
`docs/status/blue.md`.

Untracked: `blue/analysis/compare.py`, `blue/analysis/metrics.py`,
`blue/analysis/recorder.py`, `blue/tests_blue/test_legacy_ordering_gate.py`,
`blue/tests_blue/test_max_age_guard.py`,
`blue/tests_blue/test_metric_semantics.py`,
`blue/tests_blue/test_recorder_live.py`, `blue/tests_blue/test_residual_audit.py`,
`docs/proposals/manifests/guard-maxage-20261004.json`,
`docs/proposals/manifests/guard-maxage-matrix-20261004.json`,
plus this file.

## 10. Resume prompt

> Continue Blue on branch `blue/mappo-training` (HEAD `2900711`, dirty tree,
> nothing committed). Read `AGENTS.md`, `docs/status/blue-handoff-20261004.md`,
> then `docs/status/blue.md`. Start at **P0**: create branch
> `blue/metric-repair-maxage` and apply the six claim corrections in §3 of the
> handoff to `guard-maxage-20261004.json`,
> `guard-maxage-matrix-20261004.json`, `docs/status/blue.md` and
> `docs/status/blue-session.md`, using the verified numbers in §2. Commit the 256
> matrix cells as a compact tracked JSONL with config, commit hash and seed list,
> then commit P0 with a specific change title. Then P1 (extend
> `docs/coordination/blue-training-seeds.md` with the 7809–8200 block request and
> file the CRN request to Environment/Red as recommended-against with numbers),
> P2 (`AdaptiveBluePolicy` with `OrderedPolicy` byte-identical, guard off by
> default, decision logging on, an explicit `torch.Generator` threaded through
> `masked_choice`, and a `freeze_eval(seed_list, n)` contract), P3 (the five named
> baselines on the 32 dev seeds, recording the noise floor), and P4 (one bounded
> unguarded residual-PPO pilot, ≤ 80 episodes, development seeds, original
> reward, coverage and age monitored but not penalised, saving checkpoint and
> decision traces). Primary analysis is the unadjusted paired difference against
> deployed lancer; compromise counts are outcomes, not covariates. Apply the §4.2
> decision rules to whatever interval the frozen evaluation produces, and do not
> enlarge `n` after seeing gains. Do not re-plan agreed work; do not run the
> guard matrix again.

## P0 addendum, 2026-10-05

- §2.2(d) coupling range was incomplete: the quoted "r -0.265 to -0.488"
  covered only a subset of contrasts. Full recomputation from `cells.jsonl`:
  `strict_lancer` -0.265, `maxage24_lancer` **+0.003**, `maxage48_lancer`
  -0.300, `maxage96_lancer` -0.365, `maxage48_oracle` -0.488,
  `strict_oracle` **-0.188**, `unguarded_oracle` -0.381 (r² 0.00–0.24).
  The substantive point stands (coupling explains little variance), with
  the corrected range recorded in `guard-maxage-matrix-20261004.json`
  `corrections_applied_20261005.coupling_r`.
- Training-history correction: RL **was** run in this project before P0 —
  EPyMARL MAPPO fine-tunes (checkpoints under `blue/results/models/mappo_*`,
  `.th` files), the 20-iter residual PPO pilot (`e6b805d`,
  `resid-pilot1-20261004.json`, invalidated by metric defects not by
  absence), and IQL offline-RL runs (`iql_*` checkpoint dirs; the stronger
  comparison is ten `iql-v2auto` checkpoints pooled at -97.9 over 160 cells
  = 10 ckpts x 16 fresh seeds, a null against round-robin). But "only the
  FQE evaluator was invalid" is also wrong: the feasibility audit
  (`docs/research/blue-rl-feasibility-audit-20261003.md`) found IQL never
  reads its target Q, the offline logs drop the final reward, and extraction
  ignores validity masks — so those checkpoints trained a non-canonical IQL
  and their null says little about IQL as a method. "RL was never run" is
  withdrawn; the accurate status is that no run has yet produced a reliable
  learned defender that improves on lancer. Build continues in
  `blue/training/mappo_guide.py` (expert-guided shared-actor MAPPO for
  investigation scheduling, central-V critic).