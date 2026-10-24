# Blue event-time RL under Qwen Red (pre-run proposal)

**Status:** pre-registered design, 2026-10-24. Branch `blue/event-time-rl`.
Producer: Blue. Consumers: Blue evaluator and Red's existing local planner
adapter. No simulator, reward, action, observation, or shared schema change is
requested. The Red implementation remains at the pinned source revision; this
proposal changes only the Blue experiment's explicit local model setting.

## Why this experiment

Previous IQL/AWDC extraction frequently learned Sleep or failed to preserve
coverage. Unguarded PPO collapsed host coverage, while a strict coverage guard
prevented that failure. Earlier `actor_guard` gains were against an **unguarded**
Lancer, so they cannot identify an RL contribution. OP2 JSRL lost to its guard
control; GPT-Blue residual PPO has not established a multi-seed improvement.
These are historical results under different attackers and protocols.

The intervention is a narrow learned host selector inside the fixed strict
Lancer recovery/verification framework. The learner may rank valid Analyse
hosts only when the fixed policy has a free, nonurgent sweep choice. Its
decisions cannot bypass simulator validity or the strict coverage rule. The
primary control has the **same** guard and fixed rules, with Lancer host
ranking at the free choice. An unguarded Lancer remains a context arm.

The learning comparison keeps policy architecture, observations, exploration,
rollout budget, optimizer, and seed schedule equal. Arm A uses the existing
per-tick Monte Carlo/PPO targets. Arm B uses event-time semi-Markov targets:
if an agent's free decision occurs at tick `t` and its next free decision is
at `t+d`, its target is `sum_{k=0}^{d-1} gamma^k r_{t+k} + gamma^d V(h_{t+d})`;
terminal episodes omit the bootstrap. Both arms use only recent joint
Blue-visible history for the centralized critic; no Red or simulator truth is
an actor/critic input. Forced Sleep, recovery, and verification decisions
have no actor gradient. Native team reward is counted once per tick. The
event-time update uses the actual next free decision and logged duration,
including pending native actions.

This tests the causal mechanism suggested by [AACE's asynchronous credit
assignment](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1c153788756d35559c22d105d1182c30-Abstract-Conference.html)
within CAGE4's strong heuristic hierarchy; it does not assume that AACE's
results transfer. [CAGE4 hierarchical MARL](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_77.pdf)
supports retaining a competent expert mode. [SPIBB](https://proceedings.mlr.press/v97/laroche19a.html)
motivates a fixed fallback for unsafe/unsupported decisions. We make no
theorem-based safety claim for this implementation.

## New Red regime

- Local model `qwen2.5:3b`, Ollama digest
  `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`,
  Ollama `0.40.0`, 100% GPU at preflight. Red planner: 2 plans per agent,
  interval 20 ticks, 36 total LLM requests per episode; native horizon 400.
- This is a **new attacker regime**. All Llama 8B returns, paired variances,
  sample-size estimates, and rank claims are historical context, never Qwen
  baselines. The same Qwen tag/digest and Red source hashes are required for
  every arm. Any digest change starts a new version and fresh baselines.
- The 20-tick smoke at seed 7629 had 0 Blue return and is only a wiring check;
  it does not establish attacker strength. A 400-tick Qwen Lancer preflight on
  the same spent seed verifies nonzero planning and a meaningful native
  attack before the longer comparison. Record the Red request count, native
  reward, remediation, coverage, and wallclock. Do not select a Blue policy
  from the smoke.

## Stages and gates

1. **Mechanics.** Test exact strict-guard request/action parity with a no-op
   learner hook, legal host choices, row timing, pending-action duration,
   terminal/truncation bootstrapping, and zero gradient on forced choices.
   A one-episode dry run must emit transition counts and actor-gradient
   counts. Stop on any mismatch.
2. **Qwen headroom (development).** Evaluate `lancer`, `lancer_guard`, and
   historical `actor_guard` on identical development seeds 7641–7664,
   400 ticks, native reward. The old actor checkpoint is frozen at SHA256
   `f5afa909efeee600637e3cd01279ac8e1818b469133c752f770045f254267a4e`;
   the risk model at SHA256
   `64b3c7ae7775a1cd447de11bfd453fafa1097ea730b8ffffb7351dfd71473f93`.
   This is a diagnostic, not an RL success gate. If Qwen provides negligible
   attack pressure (near-zero native loss and no meaningful planning), do not
   claim a win; use Finite Red for the main causal training comparison and
   version any proposed attacker change separately.
3. **Pilot.** Two training RNG seeds per RL arm, equal native rollout ticks,
   with the guard enabled during collection and evaluation. Check episode
   boundaries and gradient counts again. Continue to the main run only if the
   event-time arm actually produces nonzero free-decision updates, never
   violates coverage/masks, and finishes within the measured compute budget.
   The pilot does not select between arms by return.
4. **Main.** At least five independent training RNG seeds per RL arm, fixed
   equal rollout budgets. Save all checkpoints; select each run's checkpoint
   by a predeclared development rule using native return versus strict
   Lancer, with lower-tail return breaking ties. Measure Qwen, Finite, Sleep,
   random, unguarded Lancer, strict Lancer, and both RL arms on separate
   paired evaluation seeds. Size the Qwen evaluation from a fresh Qwen pilot
   variance; never reuse Llama's paired SD. Use episode and training-run
   bootstrap units, not agents or ticks, for intervals.
5. **Final.** The reserved block 7809–8200 remains unopened until code,
   model, artifact hashes, checkpoint rule, and sample size are frozen.
   A win requires the event-time RL arm to improve mean native reward over
   **strict Lancer** with a paired 95% interval above zero, no material
   deterioration in the lower tail or host coverage, and the benefit to
   persist across training seeds. Distinguish Qwen and Finite results.

Qwen inference may vary even at the same simulator seed. Paired initial
seeds reduce scenario variance but do not make the Red trajectory identical.
Report inference/version details and any repeat-run variance. The fixed
fallback is a safeguard/control, while the endpoint claim is a trained RL
host selector outperforming its same-guard heuristic control.

## Reproducibility and rollout

The runnable implementation must save source commit and per-file hashes,
exact model digest, Red prompt/source hashes, Blue checkpoint hashes,
observations/action/preprocessing versions, training and evaluation seed
lists, hyperparameters, native rewards, selected checkpoint rule, and the
actual simulator action stream. Weights and runtime logs remain outside Git.
No historical Llama episode can be loaded from the Qwen cache: the evaluator
fingerprints model digest, planner/LLM config, Blue artifacts, and source.

Dependencies: inspect the GPT-Blue harness at exact commit `f32ba72` before
porting it into this isolated Blue branch. Do not merge Red code or shared
schemas. Any new evaluator field is additive; older result JSONs remain
historical and are not migrated. If a shared contract becomes necessary,
draft a separate coordination proposal for affected roles before merging.
