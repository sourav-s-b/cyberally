# Blue architecture comparison

Branch: `blue/architecture-tests` in the isolated `runs/blue-architecture-tests` clone. The root checkout and other agents' branches are untouched.

The project currently uses a small per-host MLP to rank investigations. This comparison changes only the actor architecture and its visible context; the existing remediation rules, coverage guard, ML risk scorer, finite Red, native reward, masks, critic and optimizer recipe are shared. Each arm gets newly collected on-policy episodes. The graph arm reads only Blue-observed process connections; a self-edge-only graph with identical parameters tests whether those connections actually help. The recurrent arm summarizes all owned hosts on every simulator tick and replays the whole causal sequence during training.

The four tests are: 64/128/256 hidden-unit MLP capacity; attention versus 128-unit MLP; GRU memory versus 128-unit MLP; observed-connection graph versus its self-edge control. All arms start with a zero residual head and the same protected heuristic mixture. This screen uses a deliberately small budget and reused development episodes. Model comparisons require a longer, separately frozen run before a superiority claim.

Predeclared screen: two independent training RNG seeds, eight PPO iterations with one full 400-step episode per iteration, and one policy sampling replica per evaluation episode. Training episodes cycle through 7706–7709; evaluation episodes are 8233, 8236, 8241 and 8229. The locked Python 3.11, NumPy 1.26.4, PyTorch 2.2.0 runtime and original risk scorer (SHA-256 `ec4026a3c81c2746253f306fd7add00a07f42fe13533b856fb6b4d01b8402fcd`) are read from the root checkout's ignored artifacts. The run writes exact configuration, source hashes, simulator digest, runtime, scorer hash, seed lists, checkpoints, all model results and paired contrasts under ignored `blue/results/architecture_tests/`.

The comparison unit is the training RNG seed. Evaluation episodes are shared blocks, then averaged within each trained model. With two training seeds, intervals are descriptive and have very low precision. Controls are Lancer, coverage guard, Sleep, masked random and the common untrained mixture. Report all models, including failures and extreme losses. Seeds 7809–8200 are excluded by the harness seed check.

Known limits: the user's newer finite-Red parity result is absent from the fetched source refs, so this experiment does not claim to reproduce that incumbent. The new trainer uses full-batch PPO for every arm; its numbers are not directly comparable to older minibatch trainer results. Visible process edges may be sparse. If so, the graph comparison measures that input limitation as well as the graph architecture.

Validation: 65 focused tests passed under the locked runtime, covering action masks, host permutation, visible-only edges, GRU replay and the existing harness. After the final configuration edit, 21 focused architecture/redesign tests passed. A 60-step, one-iteration run exercised every architecture with the original runtime and scorer; its scores are pipeline checks, not performance estimates. No external deployment or final-seed evaluation is part of this work.

## Completed local screen (2026-10-12)

Training source commit `d0700553ebf87567b9e08a57e524a2706b88a10f`; exact frozen manifest and all 14 checkpoints are under ignored `blue/results/architecture_tests/full400_8x1/`. The runner completed successfully: seven actor arms × two training RNG seeds, eight 400-step training episodes/model, four shared development evaluation episodes/model. Higher return is better. The two columns below are **separate trained models**, not eight independent training replications.

| Actor | Training RNG 0 | Training RNG 1 | Worst evaluation episode |
| --- | ---: | ---: | ---: |
| MLP 64 | −192.50 | −192.50 | −241 |
| MLP 128 | −192.50 | −192.50 | −241 |
| MLP 256 | −192.00 | −256.50 | −302 |
| Attention 128 | −224.75 | −202.00 | −370 |
| GRU 128 | −192.50 | −192.50 | −241 |
| Graph with visible connections 128 | −192.00 | −192.00 | −239 |
| Graph with self edges only 128 | −192.00 | −207.25 | −300 |

Shared evaluation controls on the same four episodes: Lancer −133.50, coverage guard −139.00, common untrained sampled policy −192.50, Sleep −5134.75, masked random −659.00. Every trained arm remains below Lancer and the guard. The graph arm is the best of these trained runs at −192.00, only 0.50 above its untrained starting return. This does not establish a useful gain from graph structure.

The four answers at this budget are limited but concrete. Increasing MLP width to 128 changes no mean return; 256 improves one training seed by 0.50 and harms the other by 64.00. Attention harms both relative to MLP 128 by 32.25 and 9.50. GRU ties MLP 128 exactly on both seeds. Visible graph connections tie the parameter-matched self-edge graph on seed 0 and improve seed 1 by 15.25, entirely because one episode (8241) scores −239 instead of −300. The graph did receive non-self Blue-visible edges during training and evaluation. Its result is an episode-specific mechanism hint, not evidence of a general topology benefit.

Full action request hashes show that the networks sometimes chose different investigations even when returns matched. The 256-unit MLP and attention each differ from the 128-unit MLP on all four evaluation episodes for both training seeds; GRU differs on one episode for seed 0 and none for seed 1. No model-selection decision follows from these four reused development episodes. The analysis postprocessor suppresses confidence intervals for fewer than three training seeds; its paired seed differences are descriptive. In particular, equal means across two seeds do **not** imply zero uncertainty over environments.

Targeted replay on development seed 8241 confirms the policy-influence mechanism. The MLP 128 chose a non-heuristic target in 108/594 sweep decisions; attention 128 in 80/543; visible graph 128 in 97/601. Yet the mean maximum per-decision change in action probability from the learned residual, evaluated on **the same visited states** against zero residual logits, was only 0.00044, 0.00072 and 0.00089 respectively. The mean maximum score shifts were 0.0138, 0.0206 and 0.0215. These are diagnostic replays, not additional independent evaluation episodes. They show that most non-heuristic choices at this setting come from the fixed sampling component rather than a strong learned reordering.

The trained policies inherit a fixed 20% sampled exploration component and a bounded residual bonus of 0.25. At this budget their architecture differences are small relative to the untrained policy's deficit against the controls. A next diagnostic should isolate policy execution and credit assignment using the frozen checkpoints, then test a revised execution/training recipe under a new manifest before paying for wider networks or a longer Kaggle run. This screen does not reproduce the user's later finite-Red incumbent checkpoint, which was not available in the fetched source refs.

Next task: diagnose the sampled policy's deficit against the guard and Lancer from the saved checkpoints, including actual learner overrides and whether the bounded residual can overcome base score gaps. If that has a testable fix, predeclare a separate longer run; use Kaggle shards for any substantial training expansion. PR review should check that all features are Blue-visible, the graph control is parameter matched, old policy actions are exactly replayed before PPO updates, and no seed or reward protocol changes were introduced. The training plan hashes the original committed runner; a later report-only change suppresses misleading two-seed t-intervals and records its own source hash separately.
