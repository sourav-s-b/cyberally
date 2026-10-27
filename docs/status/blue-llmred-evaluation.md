# Blue against LLM Red: first evaluation of the learned agent

Date: 2026-10-07. Author: Blue role. Branch `opencode/blue-diagnosis`.
Attacker implementation and evidence: Red, merged in `main` at `01140fe`
(`red/blue-lancer-comparison`, `7634cb3`).

This is the first time any Blue policy was measured against Red's
planner-driven attacker. It changes the interpretation of every earlier
number in this project.

## Setup

- Attacker: `CC4LLMEnv` -> bounded LLM strategy planner + persistent tactical
  executor. No DiscoveryFSRed fallback.
- Model: `llama3.1:8b`, digest
  `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`, which
  is **identical to the digest in Red's report**. Local ollama is 0.40.0 vs
  Red's 0.35.1; LLM inference is not bit-reproducible across runtimes.
- Planner: 2 plans per agent, interval 20, global request budget 36 — Red's
  own `--compare` parameters.
- Blue observation contract unchanged: no actor, mask, risk model or scorer
  sees anything new.
- Seeds 7629/7630/7631 — the same three Red used, so the comparison aligns.
  These are spent development seeds, not a fresh evaluation block.
- Reserved final block 7809-8200 untouched; `llmred_eval.py` refuses those
  seeds in code.
- Actor: `real_s0` from the risk-input ablation, one checkpoint, one
  training seed. `actor_guard` is the same weights with the existing strict
  coverage guard enabled (`--sweep-guard strict`).

Reproduction of Red's baseline, same seeds:

| policy | ours | Red | delta |
|---|---:|---:|---:|
| LLM planner | -670.7 | -680.0 | +9.3 |
| DiscoveryFSRed (scripted) | -55.0 | -55.0 | 0.0 |
| tactical executor only | -113.3 | -113.3 | 0.0 |

The two non-LLM arms match Red to the decimal, which validates the local
environment. The LLM arm differs by 1.4%, consistent with inference
nondeterminism.

## Results

Native Blue return, 400-step horizon, mean over 3 seeds (lower is worse for
Blue):

| arm | 7629 | 7630 | 7631 | mean | rem | conf | top-1 pick share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lancer | -704 | -687 | -621 | **-670.7** | 53 | 20 | 3.2% |
| RL actor (unguarded) | -1309 | -1458 | -1234 | **-1333.7** | 39 | 17 | 10.1% |
| RL actor + coverage guard | -604 | -432 | -637 | **-557.7** | 67 | 24 | 2.8% |

Paired against Lancer, per seed, with a t interval on df=2 (n=3 training-
independent units):

| contrast | per-seed | mean | 95% CI |
|---|---|---:|---|
| actor - Lancer | -605, -771, -613 | **-663.0** | [-895.6, -430.4] |
| actor_guard - Lancer | +100, +255, -16 | **+113.0** | [-224.8, +450.8] |
| actor_guard - actor | +705, +1026, +597 | **+776.0** | [+221.6, +1330.4] |

## What this means

**1. The unguarded learned actor is catastrophically worse than the
heuristic.** -663 against Lancer, interval excluding zero, 3/3 seeds. Against
DiscoveryFSRed this same agent scored about -69 versus Lancer's -55 and read
as "roughly at parity, slightly worse". Against an attacker that concentrates
it is **twice as bad as doing nothing clever**. The coverage investigation
(`docs/research/rl-algorithm-change-shortlist.md`,
`blue/training/trace_episode.py`) found the mechanism: agent 3 spent 162 of
200 ticks re-analysing one router, confirmed nothing, and issued zero
remediation. That was invisible against the weak attacker because the weak
attacker never punished it.

**2. The coverage guard is the load-bearing component.** +776 over the same
weights, interval excluding zero, 3/3 seeds, with top-1 pick concentration
falling 10.1% -> 2.8% and remediation rising 39 -> 67. We very nearly shipped
the unguarded agent. This retroactively justifies making coverage discipline
mandatory rather than an option.

**3. A first, unestablished win over Lancer.** +113 in favour of the guarded
actor, but the interval [-225, +451] straddles zero and only 2 of 3 seeds are
positive. **This is not evidence that RL beats the heuristic.** It is a
reason to measure it properly.

## Honest limits

- n=3 seeds, one actor checkpoint, one training seed. The guard-vs-actor
  contrast is solid; the guard-vs-Lancer contrast is not.
- Seeds 7629-7631 are spent development seeds. Red used them, which is why
  the baselines align, but a fresh block is needed for any claim.
- LLM inference is nondeterministic; exact numbers will move on rerun.
- Only the ungarded training regime has checkpoints. Every model we have was
  trained while the sweep was unguarded, so it is mismatched to the policy
  that actually works.

## Consequences for the programme

- **The `red_agent="discovery"` development regime is no longer a valid
  optimisation target.** It ranked ML inputs by ~7 points in a world where
  the attacker cannot punish a broken investigation policy. All Blue numbers
  measured under it must be treated as uninformative about defence.
- **The ML question is open again, and now matters.** Whether the risk score
  helps has never been tested against an attacker that concentrates.
- **Retraining with the guard active is now a prerequisite**, not an option.

## Reproduce

```bash
# 1. model must be the pinned digest
ollama list   # expect llama3.1:8b @ 46e0c10c039e...
# 2. baseline reproduction against Red's numbers
.venv-train/bin/python -m red.main --compare --seeds 7629 7630 7631 \
    --steps 400 --plans-per-agent 2 --plan-interval 20 --max-requests 36 \
    --output runs/lancer-llmred
# 3. Blue policies under LLM Red
.venv-train/bin/python -m blue.training.llmred_eval \
    --actor <dir with actor.th> --seeds 7629 7630 7631 \
    --arms lancer actor actor_guard --max-requests 36 \
    --plans-per-agent 2 --plan-interval 20 \
    --out blue/results/llmred/eval_7629_7631.json
```

Note: ollama must have CUDA initialised. On this host a WSL restart was
required before it offloaded to GPU instead of exhausting system RAM
(7 GB total). Check with `ollama ps` — the PROCESSOR column must not read
100% CPU.

Results artifact: `blue/results/llmred/eval_7629_7631.json` (gitignored;
source hashes recorded inside it).
## Fresh 24-seed block (2026-10-13, branch `blue/llmred-fresh24`)

The 3-seed development block above could not resolve the guard-vs-Lancer
contrast. This section repeats that contrast on 24 seeds that were not used
for it: **7641-7664**, outside the reserved block 7809-8200, which the
evaluation script refuses in code. Same single checkpoint
(`blue/results/ablation_risk/real_s0`, one training seed), same guard
setting, same pinned attacker digest, 400-step horizon.

Native Blue return, paired on identical simulator seeds:

| arm | mean | sd | sem |
|---|---:|---:|---:|
| Lancer | -637.3 | 170.4 | 34.8 |
| RL actor + coverage guard | -516.7 | 176.1 | 36.0 |

| contrast | mean | sd of diffs | 95% CI (df=23) | t | seeds won |
|---|---:|---:|---|---:|---:|
| actor_guard - Lancer | **+120.6** | 282.3 | [+1.4, +239.8] | 2.09 | 16/24 |

Behavioural counters, mean over the 24 seeds:

| arm | remediation actions | confirmed hosts | top-1 pick share |
|---|---:|---:|---:|
| Lancer | 70.0 | 21.8 | 4.53% |
| actor_guard | 70.6 | 21.9 | 3.95% |

The two arms are behaviourally near-identical on this block: remediation
volume, confirmed-host count and pick concentration all match within noise.
Whatever return difference exists is therefore **not** explained by the
actor remediating more, as it was in the 3-seed block (39 -> 67). On 24
seeds the guard's concentration fix and the actor's contribution are no
longer separable by these counters.

### Reading

The sign is the same as the 3-seed block and the interval now excludes
zero, but only just: the lower bound is +1.4 against a per-seed spread of
282. Three of the 24 seeds carry the effect on their own (+744, +579,
+547) and two are severe losses (-502, -370). **This is a better-supported
result than the 3-seed version, not a settled one.** It remains one
checkpoint from one training seed, so it cannot separate "the guarded RL
actor is better" from "this particular initialisation is better".

Unlike the development regime, no parameter was selected after seeing these
seeds: the checkpoint, guard setting, horizon and planner parameters were all
fixed beforehand.

### Integrity caveat: the two halves of this table come from different code

The run crashed at 43 of 48 episodes writing no results file. Resume support
was added (`de37afe`) and the last 5 episodes were completed with it. The 43
earlier episodes were produced by an **uncommitted working-tree variant** of
`llmred_eval.py` that is not in any commit and did not survive the removal of
the `cyberally-opencode` worktree. Evidence: those rows carry a
`recovered_from` field that appears in no committed version of the file.

- Rows 0-42 carry `recovered_from`; rows 43-47 carry `distinct_analysed`.
- The primary metric (`return`) is produced by the same code path in the
  surviving version, but the exact producing code for rows 0-42 cannot be
  verified.
- Therefore rows 0-42 are **indicative only**. The defensible subset is the
  5 fresh episodes plus the separately-hashed 3-seed development block.
- A full clean re-run of all 48 episodes is the only way to remove this
  caveat. It costs roughly 50 minutes of LLM inference.

### Frozen artifact hashes (all gitignored, recorded here instead)

| Artifact | SHA-256 |
|---|---|
| `blue/results/llmred/eval_fresh24.json` | `e4a70983d68fbb5a367d49439a8f58d59db99868d65618634a545c306d22a6a9` |
| `blue/results/llmred/eval_fresh24.partial.jsonl` | `a85447539511453279fe03d8bb0a7934a351983f01529f73212b5a8338aed9a0` |

Source hashes at run time are recorded inside the JSON. Note that they
describe the *resumed* code, not the code that produced rows 0-42.

### What this does and does not settle

It does **not** license the claim that RL beats the heuristic. The
concentration failure the guard fixes is real and load-bearing (previous
section), but a single +120 against a 282 sd on 24 seeds of one model is
suggestive. Settling it needs the retrain with the guard active during
training, which is the next task; every checkpoint produced so far was
trained while the sweep was unguarded.
