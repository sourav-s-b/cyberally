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
import torch as th

from modules.agents import REGISTRY as agent_REGISTRY

import blue_baselines as baselines
import cc4_epymarl_wrapper as wrapper


class GreedyCheckpointPolicy:
    """Greedy masked-argmax policy from a saved MAPPO actor."""

    def __init__(self, ckpt_dir, hidden_dim=64):
        self.n_agents = len(wrapper.BLUE_AGENTS)
        self.obs_dim = wrapper.DEFAULT_MAX_HOSTS * wrapper.VECTOR_LEN
        self.n_actions = 2 + 3 * wrapper.DEFAULT_MAX_HOSTS
        input_shape = self.obs_dim + self.n_agents  # + one-hot id, no last-act
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
            a: th.zeros(1, self.agent.args.hidden_dim) for a in wrapper.BLUE_AGENTS
        }

    def select(self, env, agent):
        i = wrapper.BLUE_AGENTS.index(agent)
        obs = np.concatenate(
            [env.get_obs_agent(i), np.eye(self.n_agents)[i]]).astype(np.float32)
        with th.no_grad():
            logits, h = self.agent(th.from_numpy(obs).unsqueeze(0), self.hidden[agent])
        self.hidden[agent] = h
        masked = logits.squeeze(0).numpy()
        masked[env.get_avail_agent_actions(i) == 0] = -np.inf
        return int(np.argmax(masked))


def main():
    parser = argparse.ArgumentParser(description="Eval a MAPPO checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7629, 7630, 7640])
    parser.add_argument("--steps", type=int, default=400)
    cli = parser.parse_args()
    factories = {
        "mappo_ckpt": lambda: GreedyCheckpointPolicy(cli.ckpt),
        "round_robin": baselines.RoundRobinBaseline,
        "masked_random": lambda: baselines.MaskedRandomBaseline(seed=11),
        "sleep": baselines.SleepBaseline,
    }
    results = baselines.evaluate_policies(
        factories, cli.seeds, steps=cli.steps, snapshot_steps=(200, cli.steps))
    for name, per_seed in results.items():
        for seed, run in per_seed.items():
            snaps = run["snapshots"]
            print(f"{name} seed={seed} return={run['cumulative_return']:.1f} "
                  f"root@200={snaps[200]['root']} total@200={snaps[200]['total']} "
                  f"root@end={run['final']['root']} total@end={run['final']['total']}")


if __name__ == "__main__":
    main()
