"""Picklable policy specs for parallel evaluation.

``evaluate_policies`` takes lambdas, which cannot cross a process boundary.
This registry maps a policy name to an importable (module, class, kwargs)
triple plus optional overrides, resolved *inside* the worker via build().
Top-level imports are stdlib-only so the sim venv can load this module;
torch/EPyMARL imports happen lazily inside build(), in the worker that
actually instantiates the policy.
"""

import importlib

REGISTRY = {
    "sleep": ("blue_baselines", "SleepBaseline", {}),
    "masked_random": ("blue_baselines", "MaskedRandomBaseline", {"seed": 11}),
    "unmasked_random": ("blue_baselines", "UnmaskedRandomBaseline", {"seed": 11}),
    "round_robin": ("blue_baselines", "RoundRobinBaseline", {}),
    "suspicion_sweep": ("blue_baselines", "SuspicionSweepBaseline", {}),
    "hybrid_none": ("blue_hybrid", "HybridBluePolicy", {"priority_fn": None}),
    "hybrid_lancer": ("blue_hybrid", "HybridBluePolicy",
                      {"priority_fn": "lancer"}),
    # v2: suspicion bonus decays with fruitless re-analyses (0.5 each).
    # v1 ("hybrid_lancer", fruitless_decay=1.0) stays frozen for comparison.
    "hybrid_lancer_v2": ("blue_hybrid", "HybridBluePolicy",
                         {"priority_fn": "lancer",
                          "priority_kwargs": {"fruitless_decay": 0.5}}),
    # Learned risk scorer (model weights live outside git under results/).
    "hybrid_risk": ("blue_hybrid", "HybridBluePolicy",
                    {"priority_fn": "risk",
                     "priority_kwargs": {
                         "model_path": "results/risk_model_v2.pkl"}}),
}

TORCH_POLICIES = {"mappo_ckpt"}


def needs_torch(name):
    """True if this policy requires the train venv (torch + EPyMARL link)."""
    return name in TORCH_POLICIES


def _build_ckpt(ckpt_dir, hidden_dim=64, steps=400, **env_kwargs):
    """Greedy MAPPO-checkpoint policy; probe env sizes geometry from flags."""
    import cc4_epymarl_wrapper as wrapper
    from blue_eval_mappo import GreedyCheckpointPolicy
    probe = wrapper.CC4MARLEnv(steps=steps, **env_kwargs)
    env_info = probe.get_env_info()
    return GreedyCheckpointPolicy(ckpt_dir, hidden_dim=hidden_dim,
                                  env_info=env_info)


def build(name, **overrides):
    """Instantiate a fresh policy from its registry spec (call in worker)."""
    if name == "mappo_ckpt":
        if "ckpt_dir" not in overrides:
            raise ValueError("mappo_ckpt needs ckpt_dir=...")
        return _build_ckpt(**overrides)
    if name not in REGISTRY:
        raise KeyError(f"unknown policy {name!r}; known: {sorted(REGISTRY)}")
    module, cls, kwargs = REGISTRY[name]
    return getattr(importlib.import_module(module), cls)(**{**kwargs, **overrides})


def names():
    return sorted(REGISTRY) + sorted(TORCH_POLICIES)
