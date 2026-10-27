from __future__ import annotations

from pathlib import Path


class SetActorCheckpointPolicy:
    def __init__(self, actor_path, scorer_path, hidden_dim=64, bonus=0.25,
                 ml_inputs="risk", temperature=0.5, explore=0.2,
                 max_age=80.0, policy_seed=0, execution="sample"):
        import torch as th

        from blue.harness.policy import HarnessPolicy
        from blue.harness.redesign import RedesignHook, SetActor
        from blue.harness.scoring import MLScorer

        actor = SetActor(hidden_dim)
        state = th.load(Path(actor_path), map_location="cpu",
                        weights_only=True)
        actor.load_state_dict(state)
        actor.eval()
        hook = RedesignHook(
            actor, MLScorer(str(scorer_path)), bonus=bonus,
            ml_inputs=ml_inputs, temperature=temperature, explore=explore,
            policy_seed=policy_seed, execution=execution)
        self.policy = HarnessPolicy(hook, max_age=max_age,
                                    record_requests=True)

    def reset(self):
        self.policy.reset()

    def select(self, env, agent):
        return self.policy.select(env, agent)

    @property
    def last_request(self):
        return self.policy.last_request
