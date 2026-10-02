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
    # Representation check (proposal 12): committed version of the scratch
    # claim_check; pool manifest carries the equality evidence.
    "stalest_first": ("blue_baselines", "StalestFirstBaseline", {}),
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
    # KILLED 2026-10-02 (proposal 01 Experiment 4: coverage collapse).
    # Kept so the failure stays reproducible; do not deploy.
    "hybrid_risk": ("blue_hybrid", "HybridBluePolicy",
                    {"priority_fn": "risk",
                     "priority_kwargs": {
                         "model_path": "results/risk_model_v2.pkl"}}),
    # Risk x recency hybrids (proposal 01 phase 3): snapshot proba with
    # touch-decay dynamics. Gate: beat lancer_v2 on held-out, not regr.
    "hybrid_rx_decay": ("blue_hybrid", "HybridBluePolicy",
                        {"priority_fn": "risk_recency",
                         "priority_kwargs": {"mode": "decay"}}),
    "hybrid_rx_bonus": ("blue_hybrid", "HybridBluePolicy",
                        {"priority_fn": "risk_recency",
                         "priority_kwargs": {"mode": "bonus"}}),
    "hybrid_rx_gate": ("blue_hybrid", "HybridBluePolicy",
                       {"priority_fn": "risk_recency",
                        "priority_kwargs": {"mode": "gate"}}),
}

TORCH_POLICIES = {"mappo_ckpt", "mappo_ckpt_factorized"}
_CKPT_AGENT = {"mappo_ckpt": "rnn",
               "mappo_ckpt_factorized": "rnn_factorized"}


def needs_torch(name):
    """True if this policy requires the train venv (torch + EPyMARL link)."""
    return name in TORCH_POLICIES


def _build_ckpt(name, ckpt_dir, hidden_dim=64, steps=400, attn_layers=0,
                **env_kwargs):
    """Greedy MAPPO-checkpoint policy; probe env sizes geometry from flags."""
    import cc4_epymarl_wrapper as wrapper
    from blue_eval_mappo import GreedyCheckpointPolicy
    probe = wrapper.CC4MARLEnv(steps=steps, **env_kwargs)
    env_info = probe.get_env_info()
    return GreedyCheckpointPolicy(ckpt_dir, hidden_dim=hidden_dim,
                                  env_info=env_info,
                                  agent_type=_CKPT_AGENT[name],
                                  attn_layers=attn_layers)


def build(name, **overrides):
    """Instantiate a fresh policy from its registry spec (call in worker)."""
    if name in TORCH_POLICIES:
        if "ckpt_dir" not in overrides:
            raise ValueError(f"{name} needs ckpt_dir=...")
        return _build_ckpt(name, **overrides)
    if name not in REGISTRY:
        raise KeyError(f"unknown policy {name!r}; known: {sorted(REGISTRY)}")
    module, cls, kwargs = REGISTRY[name]
    return getattr(importlib.import_module(module), cls)(**{**kwargs, **overrides})


def names():
    return sorted(REGISTRY) + sorted(TORCH_POLICIES)