"""Eval a return-conditioned (RvS) checkpoint (proposal 14, rung 1).

Mirrors GreedyCheckpointPolicy, but the actor input is [obs+id, RTG/scale]
and the team RTG is tracked across the episode: reset(target_return)
initialises it, observe_step() decrements it by the team reward after each
joint step. run_episode() calls observe_step when present (see the hook in
blue_baselines.run_episode), so pool eval needs no special path.
"""

import os
from types import SimpleNamespace as SN

import numpy as np

import blue.core.wrapper as wrapper
from blue.policies.eval_mappo import policy_dims
from blue.training.rvs import RTG_SCALE


class RvSCheckpointPolicy:
    """Greedy masked-argmax from an RvS actor conditioned on target return."""

    def __init__(self, ckpt_dir, target_return=-50.0, hidden_dim=64,
                 env_info=None, agent_type="rnn_factorized", attn_layers=0):
        try:
            import torch as th
            from modules.agents import REGISTRY as agent_REGISTRY
        except ImportError as e:
            raise ImportError(
                "blue_eval_rvs needs the train venv (torch + EPyMARL link): "
                f"{e}")
        self._th = th
        if env_info is None:
            probe = wrapper.CC4MARLEnv(steps=400)
            env_info = probe.get_env_info()
        self.n_agents, self.obs_dim, self.n_actions = policy_dims(env_info)
        args = SN(hidden_dim=hidden_dim, n_actions=self.n_actions,
                  n_agents=self.n_agents, use_rnn=True,
                  attn_layers=attn_layers, rtg_dim=1)
        if agent_type == "rnn_factorized":
            from blue.policies import factorized
            agent_REGISTRY["rnn_factorized"] = (
                factorized.FactorizedRNNAgent)
        self.agent = agent_REGISTRY[agent_type](self.obs_dim + 1, args)
        state = th.load(os.path.join(ckpt_dir, "agent.th"),
                        map_location="cpu", weights_only=True)
        missing = self.agent.load_state_dict(state, strict=True)
        assert not missing.missing_keys and not missing.unexpected_keys, missing
        self.agent.eval()
        self.target_return = float(target_return)
        self.reset()

    def reset(self):
        self.hidden = {
            a: self._th.zeros(1, self.agent.args.hidden_dim)
            for a in wrapper.BLUE_AGENTS
        }
        self.rtg = self.target_return

    def observe_step(self, actions, rewards):
        """Decrement team RTG by the joint-step team reward (run_episode hook)."""
        self.rtg -= float(rewards[0])

    def select(self, env, agent):
        i = wrapper.BLUE_AGENTS.index(agent)
        obs = np.concatenate(
            [env.get_obs_agent(i), np.eye(self.n_agents)[i]]).astype(np.float32)
        rtg = np.array([self.rtg / RTG_SCALE], dtype=np.float32)
        with self._th.no_grad():
            logits, h = self.agent(
                self._th.from_numpy(np.concatenate([obs, rtg])).unsqueeze(0),
                self.hidden[agent])
        self.hidden[agent] = h
        masked = logits.squeeze(0).numpy()
        masked[env.get_avail_agent_actions(i) == 0] = -np.inf
        return int(np.argmax(masked))
