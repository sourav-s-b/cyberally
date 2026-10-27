"""Dashboard adapters for the main32 residual and gate-ablation actors."""

from __future__ import annotations

import json
from pathlib import Path

from blue.harness.policy import HarnessPolicy, StrictHarnessPolicy
from blue.harness.scoring import sha256


class Main32CheckpointPolicy:
    """Sampled main32 residual PPO actor behind the max-age harness guard."""

    def __init__(self, model_dir, scorer_path, policy_seed=0):
        from blue.training.harness_rl import load_hook

        hook, manifest = load_hook(
            model_dir, scorer_path, sample=True, sample_seed=policy_seed)
        config = manifest.get("config", {})
        self.policy = HarnessPolicy(
            hook, max_age=config.get("max_age", 80.0), record_requests=True)

    def reset(self):
        self.policy.reset()

    def select(self, env, agent):
        return self.policy.select(env, agent)

    @property
    def last_request(self):
        return self.policy.last_request


class ArchitectureCheckpointPolicy:
    """Sampled SetActor / fixed-blend / learned-gate actor, strict guard."""

    def __init__(self, model_dir, scorer_path, arch, policy_seed=0):
        from blue.harness.redesign import RedesignHook
        from blue.harness.scoring import MLScorer
        from blue.training.architecture_compare import actor_for

        directory = Path(model_dir)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError("gate-ablation checkpoint is incomplete")
        if manifest.get("config", {}).get("arch") != arch:
            raise ValueError("gate-ablation architecture does not match manifest")
        if manifest.get("actor_sha256") != sha256(directory / "actor.th"):
            raise ValueError("gate-ablation actor hash mismatch")
        if manifest.get("scorer_sha256") != sha256(scorer_path):
            raise ValueError("gate-ablation scorer hash mismatch")
        config = manifest["config"]
        actor = actor_for(arch, config["hidden"])
        import torch as th
        actor.load_state_dict(
            th.load(directory / "actor.th", map_location="cpu",
                    weights_only=True))
        actor.eval()
        hook = RedesignHook(
            actor, MLScorer(str(scorer_path)), config["bonus"],
            config["ml_inputs"], config["temp"], config["explore"],
            policy_seed, "sample")
        self.policy = StrictHarnessPolicy(hook, record_requests=True)

    def reset(self):
        self.policy.reset()

    def select(self, env, agent):
        return self.policy.select(env, agent)

    @property
    def last_request(self):
        return self.policy.last_request
