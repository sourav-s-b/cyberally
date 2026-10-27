# Non-PPO option after the finite-Red anchor pilot

Date: 2026-10-15. Branch: `blue/finite-red-check`, based on `2f44db1`.

## Recommendation

Screen a **JSRL-style guide curriculum with synchronous A2C** as the next
non-PPO learning method. This combines two separable choices:

- **A2C is the learner:** the existing local implementation uses one fresh,
  full-rollout policy-gradient update, without PPO's clipped likelihood ratio
  or replay. See `blue/harness/learning.py` and
  `blue/training/harness_redesign.py` (`--algorithm a2c`).
- **JSRL is the rollout curriculum:** the frozen Lancer + age-80 guard acts
  for a chosen initial prefix of each training episode; the A2C policy then
  controls eligible sweep choices for the suffix. Across training, reduce the
  guide prefix so the learner increasingly owns the episode. The JSRL paper
  defines this guide-policy roll-in followed by an exploration policy, and
  reports it as a generic wrapper for RL methods, with experiments showing
  sample-efficiency benefits in simulated robotics. This has not been tested
  in CAGE4 and gives no performance guarantee here.

This is preferable to changing only the optimizer because the last finite-Red
pilot found two different failure modes: current residual PPO often worsened
greedy defense, while fixed 0.9 categorical anchoring exactly reproduced the
guard on all 72 paired evaluation episodes. Guide-prefix training keeps the
Lancer-generated early trajectory without forcing the actor's deployed
candidate probabilities to stay near Lancer on every decision.

## Why not start with Discrete SAC

Discrete SAC is a valid off-policy algorithm for discrete action spaces and
could reuse old transitions. The project already documents the main cost:
replay rows must preserve each next decision's variable candidate list and
validity mask, while the simulator has multiple Blue agents with pending
actions and team rewards. The current harness is decision-hook based, so a
correct next-state/action-mask and reward-attribution contract is needed before
Q-learning from replay. This is a later candidate, not a one-line PPO swap.
The project's earlier IQL/CQL/SAC experiments and replay/OPE audit also mean
offline methods must not be presented as untried wins.

## Bounded screen protocol

1. Keep simulator, native reward, scorer, feature inputs, action rules and
   guard fixed. Do not expose privileged compromise labels to the learner.
2. Compare three arms: Lancer + guard; fresh-rollout A2C with no guide prefix;
   and the same A2C learner with a guide-prefix curriculum. A sensible pilot
   prefix set follows the measured three mission-phase lengths (0, about 134,
   about 267 ticks), with the schedule moving toward shorter guide prefixes.
   Freeze the exact sampling/annealing schedule before training.
3. Only learner-controlled decisions enter the actor loss. Guide actions do
   not receive actor gradients. Their rewards remain in the reported episode
   score, while each learned decision's return target starts at that decision
   and includes the remaining native-reward suffix. Preserve pending-action
   attribution, episode-end rewards and the native score exactly.
4. Match fresh environment episode budgets across A2C arms. Report A2C's
   optimizer-step count separately from rollout count because PPO takes
   multiple minibatch epochs. Use at least five training RNG seeds and a
   verified unused finite-Red evaluation block shared by both actors and all
   controls. Include Lancer, guard-only, Sleep and masked-random.
5. Primary metric: paired native-return differences, summarized by training
   RNG seed with the interval across training seeds. Also report lower-tail
   episodes, LWF/RIA penalties, repair requests/completions, coverage, guide
   fraction and actor action-change rate. A3C-style asynchronous workers are
   not needed; the local A2C implementation is synchronous and should remain
   so for reproducibility.
6. Require improvement over both Lancer and guard-only with the registered
   training-seed interval above zero and no material tail/coverage regression
   before considering a fresh confirmation cohort. Keep LLM Red deferred until
   that gate passes.

The earlier local MAPPO/A2C check trained only two iterations of two episodes
per run, evaluated on one reused episode, and matched the untrained controls.
That result establishes pipeline feasibility only. The bounded finite-Red
anchor pilot likewise is development evidence on a reused evaluation block;
its results are in `finite-red-anchor-pilot-20261015.md`.

## Primary references and project sources

- Uchendu et al., [Jump-Start Reinforcement Learning, ICML 2023](https://proceedings.mlr.press/v202/uchendu23a.html): guide-policy roll-in and learner handoff.
- Mnih et al., [Asynchronous Methods for Deep Reinforcement Learning, ICML 2016](https://arxiv.org/abs/1602.01783): actor-critic family reference. The project's `a2c` path is synchronous; it is not A3C.
- Christodoulou, [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207): discrete-action SAC, cited as a later replay-based alternative.
- Local implementation/evidence: `blue/harness/learning.py`, `blue/training/harness_redesign.py`, `docs/research/blue-harness-redesign-20261007.md`, and `docs/research/rl-algorithm-change-shortlist.md`.

## Finite-Red screen results, 2026-10-15

Completed the frozen screen in `blue/results/finite_red_jsrl_a2c_screen_v2/`:
three training RNG seeds per arm, eight iterations, four 400-tick episodes per
iteration, native reward, and finite Red. Plain A2C used no guide prefix;
guide-prefix A2C used `[267, 267, 134, 134, 67, 0, 0, 0]`. The held-out block
was evaluation seeds 9001–9032. Lancer, age-80 guard, Sleep and masked-random
controls used the same seeds. Each model/mode has 96 evaluation episodes
(three training seeds × 32 scenarios).

The paired unit is the mean evaluation-episode difference within each training
RNG seed; the reported 95% t intervals are across only three training seeds,
so these are exploratory:

| Policy / mode | Δ vs Lancer | 95% interval | Δ vs guard | 95% interval |
|---|---:|---:|---:|---:|
| A2C greedy | -11.49 | [-36.90, 13.92] | -14.68 | [-40.09, 10.73] |
| A2C sampled | +3.69 | [-12.74, 20.11] | +0.50 | [-15.93, 16.93] |
| Guide-prefix A2C greedy | -0.30 | [-25.36, 24.76] | -3.49 | [-28.55, 21.57] |
| Guide-prefix A2C sampled | +4.50 | [-13.72, 22.72] | +1.31 | [-16.91, 19.53] |

Guide-prefix minus plain A2C was +11.19 greedy (95% interval [-28.44,
50.82]) and +0.81 sampled ([-2.16, 3.79]). Every interval overlaps zero.
The guide curriculum therefore did not establish an improvement over Lancer,
guard-only, or plain A2C. Guide-prefix tail returns were also mixed: greedy
minimum -472 and p10 -252; sampled minimum -464 and p10 -228.5. Plain A2C
greedy minimum/p10 were -406/-259.5; sampled were -464/-235.

Across the same 32 seeds, control means were Lancer -169.22, guard -166.03,
masked-random -743.12 and Sleep -4402.84. These controls establish useful
context, but do not change the paired training-seed conclusion. The frozen
analysis sets `recommend_confirmation: false`. Per the screen protocol, add
finite-Red training seeds before making a superiority claim or considering
LLM Red. This three-seed result is not confirmatory.

Validation: all six manifests are `complete`; their configs, source commit and
recorded source hashes match the frozen plan, the frozen source files still
match their hashes, model/report hashes match, and every model and control
report contains the same 32 evaluation seeds. See `analysis.json` for the full
per-seed results. Existing partial v1 artifacts were retained and excluded.
