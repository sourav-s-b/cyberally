"""BLUE-03 smoke: rollout -> masked update -> save/load -> valid actions.

Not a trainer: one REINFORCE update of a shared masked actor head on a short
live episode, proving the loop plumbing (masks -> torch -> optimizer step ->
checkpoint -> mask-legal actions) before any EPyMARL/MAPPO work. Run with the
training venv: ``..\\.venv-train\\Scripts\\python.exe blue_train_smoke.py``.
Checkpoint goes to a tempfile, never into Git.
"""

import tempfile

import numpy as np
import torch
import torch.nn as nn

import cc4_epymarl_wrapper as wrapper

SEED = 7629
STEPS = 30


def masked_sample(logits, mask):
    masked = torch.where(mask.bool(), logits,
                         torch.full_like(logits, float("-inf")))
    return torch.distributions.Categorical(logits=masked).sample()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    actor = nn.Linear(wrapper.CC4MARLEnv(steps=STEPS).obs_size,
                      wrapper.CC4MARLEnv(steps=STEPS).n_actions)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)

    env = wrapper.CC4MARLEnv(seed=SEED, steps=STEPS, common_reward=True)
    obs, _ = env.reset(seed=SEED)
    logps, rewards = [], []
    tick = 0
    while True:
        actions = {}
        for i, agent in enumerate(wrapper.BLUE_AGENTS):
            logits = actor(torch.as_tensor(obs[i]))
            mask = torch.as_tensor(env.get_avail_agent_actions(i))
            dist = torch.distributions.Categorical(
                logits=torch.where(mask.bool(), logits,
                                   torch.full_like(logits, float("-inf"))))
            a = dist.sample()
            logps.append(dist.log_prob(a))
            actions[agent] = int(a)
        obs, reward, terminated, truncated, _ = env.step(actions)
        rewards.append(float(reward))
        tick += 1
        if terminated or truncated:
            break
    assert tick == STEPS - 1, tick  # native horizon end, not an early terminal

    returns = torch.tensor(rewards).flip(0).cumsum(0).flip(0)
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    n_agents = len(wrapper.BLUE_AGENTS)
    loss = -(torch.stack(logps).view(tick, n_agents)
             * returns.unsqueeze(1)).mean()
    opt.zero_grad()
    loss.backward()
    assert all(p.grad is not None for p in actor.parameters())
    opt.step()
    assert torch.isfinite(loss)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save({"state_dict": actor.state_dict(), "seed": SEED,
                "obs_size": env.obs_size, "n_actions": env.n_actions}, path)
    fresh = nn.Linear(env.obs_size, env.n_actions)
    ckpt = torch.load(path, weights_only=True)
    assert ckpt["obs_size"] == env.obs_size and ckpt["n_actions"] == env.n_actions
    fresh.load_state_dict(ckpt["state_dict"])

    env2 = wrapper.CC4MARLEnv(seed=SEED, steps=STEPS)
    obs, _ = env2.reset(seed=SEED)
    for _ in range(10):
        actions = {}
        for i, agent in enumerate(wrapper.BLUE_AGENTS):
            a = int(masked_sample(
                fresh(torch.as_tensor(obs[i])),
                torch.as_tensor(env2.get_avail_agent_actions(i))))
            assert env2.get_avail_agent_actions(i)[a] == 1
            actions[agent] = a
        obs, _, terminated, truncated, _ = env2.step(actions)
        if terminated or truncated:
            break
    print(f"smoke ok: ticks={tick} return={sum(rewards):.1f} "
          f"loss={loss.item():.4f} ckpt={path}")
    return loss.item()


if __name__ == "__main__":
    main()
