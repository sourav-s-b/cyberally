"""Evaluate a saved MAPPO actor against the heuristic baselines.

Loads upstream-EPyMARL `agent.th` (RNNAgent state dict) and runs it greedily
(masked argmax, carried GRU hidden state) through blue_baselines'
run_episode/evaluate_policies, so returns and compromise counts are directly
comparable with Sleep/random/round-robin on identical seeds and horizons.

Network geometry is re-derived here (obs + one-hot agent id) to match
BasicMAC's _build_inputs for obs_agent_id=True, obs_last_action=False;
mismatched shapes fail loudly instead of silently mis-scoring.
"""

import argparse
import os
from types import SimpleNamespace as SN

import numpy as np

import blue_baselines as baselines
import cc4_epymarl_wrapper as wrapper


def policy_dims(env_info):
    """(n_agents, obs_dim, n_actions) from a get_env_info() dict.

    Pure helper so geometry is unit-testable without torch or checkpoints:
    obs gains the one-hot agent id (as in BasicMAC._build_inputs with
    obs_agent_id=True, obs_last_action=False).
    """
    n_agents = env_info["n_agents"]
    return n_agents, env_info["obs_shape"] + n_agents, env_info["n_actions"]


class GreedyCheckpointPolicy:
    """Greedy masked-argmax policy from a saved MAPPO actor.

    Geometry derives from ``env_info`` (a ``get_env_info()`` dict), so
    temporal/bound variants size correctly; a checkpoint trained under
    different flags fails loudly on load_state_dict instead of
    silently mis-scoring. Pass the same temporal flags used in training
    both here and to the rollout env (see main()).
    """

    def __init__(self, ckpt_dir, hidden_dim=64, env_info=None):
        try:
            import torch as th
            from modules.agents import REGISTRY as agent_REGISTRY
        except ImportError as e:
            raise ImportError(
                "blue_eval_mappo needs the train venv (torch + EPyMARL .pth link): "
                f"{e}")
        self._th = th
        if env_info is None:
            probe = wrapper.CC4MARLEnv(steps=400)
            env_info = probe.get_env_info()
        self.n_agents, self.obs_dim, self.n_actions = policy_dims(env_info)
        # obs_dim already includes the one-hot id (see policy_dims); no last-act
        input_shape = self.obs_dim
        args = SN(hidden_dim=hidden_dim, n_actions=self.n_actions, use_rnn=True)
        self.agent = agent_REGISTRY["rnn"](input_shape, args)
        state = th.load(os.path.join(ckpt_dir, "agent.th"),
                        map_location="cpu", weights_only=True)
        missing, unexpected = self.agent.load_state_dict(state, strict=False), None
        assert not missing.missing_keys and not missing.unexpected_keys, missing
        self.agent.eval()
        self.reset()

    def reset(self):
        self.hidden = {
            a: self._th.zeros(1, self.agent.args.hidden_dim)
            for a in wrapper.BLUE_AGENTS
        }

    def select(self, env, agent):
        i = wrapper.BLUE_AGENTS.index(agent)
        obs = np.concatenate(
            [env.get_obs_agent(i), np.eye(self.n_agents)[i]]).astype(np.float32)
        with self._th.no_grad():
            logits, h = self.agent(self._th.from_numpy(obs).unsqueeze(0),
                                   self.hidden[agent])
        self.hidden[agent] = h
        masked = logits.squeeze(0).numpy()
        masked[env.get_avail_agent_actions(i) == 0] = -np.inf
        return int(np.argmax(masked))


def main():
    parser = argparse.ArgumentParser(description="Eval a MAPPO checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7629, 7630, 7640])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--temporal-groups", nargs="*",
                        default=["ages", "belief", "freshness", "mission"])
    parser.add_argument("--drop-root-session", action="store_true")
    cli = parser.parse_args()
    env_kwargs = {"temporal_features": tuple(cli.temporal_groups),
                  "include_root_session": not cli.drop_root_session}
    probe = wrapper.CC4MARLEnv(steps=cli.steps, **env_kwargs)
    env_info = probe.get_env_info()
    factories = {
        "mappo_ckpt": lambda: GreedyCheckpointPolicy(cli.ckpt,
                                                     env_info=env_info),
        "round_robin": baselines.RoundRobinBaseline,
        "masked_random": lambda: baselines.MaskedRandomBaseline(seed=11),
        "sleep": baselines.SleepBaseline,
    }
    results = baselines.evaluate_policies(
        factories, cli.seeds, steps=cli.steps, snapshot_steps=(200, cli.steps),
        **env_kwargs)
    for name, per_seed in results.items():
        for seed, run in per_seed.items():
            snaps = run["snapshots"]
            print(f"{name} seed={seed} return={run['cumulative_return']:.1f} "
                  f"root@200={snaps[200]['root']} total@200={snaps[200]['total']} "
                  f"root@end={run['final']['root']} total@end={run['final']['total']}")


if __name__ == "__main__":
    main()
