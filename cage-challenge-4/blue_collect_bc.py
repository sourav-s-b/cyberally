"""Collect behavior-cloning demonstrations from the round-robin heuristic.

Runs RoundRobinBaseline episodes and stores per-agent (obs+id, action, mask)
trajectories for supervised pre-training of the MAPPO actor. Observations use
the same geometry as training (temporal groups + one-hot agent id), so a
BC-trained agent.th drops straight into the standard checkpoint layout.
"""

import argparse
import json
import os

import numpy as np

import cc4_epymarl_wrapper as wrapper
from blue_baselines import MaskedRandomBaseline, RoundRobinBaseline

ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True}


def make_teacher(name, seed):
    if name == "rr":
        return RoundRobinBaseline()
    if name == "lancer_v2":
        from blue_hybrid import HybridBluePolicy
        return HybridBluePolicy(
            priority_fn="lancer",
            priority_kwargs={"fruitless_decay": 0.5})
    if name == "masked_random":
        return MaskedRandomBaseline(seed=seed)
    raise ValueError(f"unknown teacher {name!r}")


def collect_episode(seed, steps=400, teacher="rr"):
    env = wrapper.CC4MARLEnv(seed=seed, steps=steps, mask_mode="validity",
                               **ENV_KW)
    env.reset(seed=seed)
    policy = make_teacher(teacher, seed)
    policy.reset()
    n_agents = len(wrapper.BLUE_AGENTS)
    obs_dim = env.get_obs_size() + n_agents
    obs = np.zeros((n_agents, steps, obs_dim), dtype=np.float32)
    actions = np.zeros((n_agents, steps), dtype=np.int64)
    masks = np.zeros((n_agents, steps, env.n_actions), dtype=np.float32)
    rewards = np.zeros((steps,), dtype=np.float32)  # common team reward
    for tick in range(steps):
        acts = {}
        for i, agent in enumerate(wrapper.BLUE_AGENTS):
            obs[i, tick] = np.concatenate(
                [env.get_obs_agent(i), np.eye(n_agents)[i]]).astype(np.float32)
            masks[i, tick] = env.get_avail_agent_actions(i)
            acts[agent] = int(policy.select(env, agent))
            actions[i, tick] = acts[agent]
        _, step_rewards, terminated, truncated, _ = env.step(acts)
        rewards[tick] = float(step_rewards[0])
        if terminated or truncated:
            obs, actions, masks = (a[:, :tick + 1] for a in (obs, actions, masks))
            rewards = rewards[:tick + 1]
            break
    env.close()
    return {"obs": obs, "actions": actions, "masks": masks,
            "rewards": rewards, "seed": seed, "steps": int(tick + 1),
            "teacher": teacher, "return": float(rewards.sum())}


def main():
    parser = argparse.ArgumentParser(description="Collect BC demonstrations")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[7629, 7630, 7640, 7701, 7702, 7703,
                                 7704, 7705, 7706, 7707])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--teacher", default="rr",
                        choices=("rr", "lancer_v2", "masked_random"),
                        help="single teacher for all seeds; use --mix for a "
                             "multi-teacher log (proposal 14 needs return "
                             "diversity for conditioning)")
    parser.add_argument("--mix", type=str, default=None,
                        help="comma-separated teachers cycled over seeds, "
                             "e.g. 'lancer_v2,rr,masked_random'")
    parser.add_argument("--out", default="results/bc_demos_rr.npz")
    cli = parser.parse_args()

    teachers = cli.mix.split(",") if cli.mix else [cli.teacher]
    episodes = [collect_episode(s, steps=cli.steps,
                                teacher=teachers[k % len(teachers)])
                for k, s in enumerate(cli.seeds)]
    lengths = [e["steps"] for e in episodes]
    max_len = max(lengths)
    n_agents = episodes[0]["obs"].shape[0]
    obs_dim = episodes[0]["obs"].shape[2]
    n_actions = episodes[0]["masks"].shape[2]
    obs = np.zeros((len(episodes), n_agents, max_len, obs_dim), dtype=np.float32)
    actions = np.zeros((len(episodes), n_agents, max_len), dtype=np.int64)
    masks = np.zeros((len(episodes), n_agents, max_len, n_actions),
                     dtype=np.float32)
    rewards = np.zeros((len(episodes), max_len), dtype=np.float32)
    for i, e in enumerate(episodes):
        t = e["steps"]
        obs[i, :, :t] = e["obs"]
        actions[i, :, :t] = e["actions"]
        masks[i, :, :t] = e["masks"]
        rewards[i, :t] = e["rewards"]
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    np.savez_compressed(cli.out, obs=obs, actions=actions, masks=masks,
                        rewards=rewards, lengths=np.array(lengths))
    meta = {"seeds": cli.seeds, "teachers": [e["teacher"] for e in episodes],
            "returns": [e["return"] for e in episodes],
            "lengths": lengths, "obs_dim": obs_dim,
            "n_actions": n_actions, "temporal_features": list(ENV_KW["temporal_features"])}
    with open(os.path.splitext(cli.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    hist = {a: int((actions == a).sum()) for a in np.unique(actions)}
    print(f"wrote {cli.out}: {len(episodes)} eps, lens {lengths}")
    print("teacher action ids histogram:", hist)
    print("teacher returns:", [round(r, 1) for r in meta["returns"]])


if __name__ == "__main__":
    main()
