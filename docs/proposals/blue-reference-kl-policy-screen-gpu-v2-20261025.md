# Persistent-reference policy screen, CUDA rerun

**Status:** frozen before v2 training. This amends the execution environment and
seed block, not the policy question or hyperparameters in
`blue-reference-kl-policy-screen-20261025.md`.

## Why v2 exists

The first launcher used the shared `2.14.1+cpu` PyTorch build. It was stopped
after three episodes before any optimizer update or checkpoint was saved. That
attempt is marked aborted in ignored `blue/results/policy_reference_kl_v1/` and
its seeds `7730–7753` are treated as consumed. The RTX 3050 is available, so
v2 uses an isolated CUDA-enabled environment and places actor/critic inference
and optimization on CUDA. Simulator transitions and CybORG remain CPU work.

The policy comparison is unchanged: fixed 0.5 SetActor/attention PPO versus
the same PPO update plus persistent `0.1 * KL(reference || current)`. Both
retain `lambda=1.0`, PPO KL early stop 0.02, and the Lancer-mixture anchor.
No results from the aborted CPU attempt enter the v2 analysis.

## Seeds and run layout

- PPO and Reference-KL RNG 0: training environment seeds 7754–7761.
- PPO and Reference-KL RNG 1: training environment seeds 7762–7769.
- Held-out development seeds: 7770–7777, both Discovery and Finite Red.
- Each training cell: 6 iterations × 4 balanced 400-tick episodes.
- CUDA runtime: isolated `/home/sourav/projects/cyberally/.venv-train-cuda`,
  PyTorch CUDA 13.0 wheel on the RTX 3050; CPU simulator and one CPU thread.

Seed audit cleared 7754–7777 after recording the v1 abort and immediately
before this freeze. Two training RNGs remain a descriptive screen only. Output
is written to ignored `blue/results/policy_reference_kl_v2_gpu/`.
