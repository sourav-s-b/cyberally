"""WSRL-style critic-only warmup tests.

Motivation: our BC policy scores -256/-379/-211 native, better than anything
PPO reached from scratch, yet all four PPO fine-tunes from that init degraded
it (docs/status/blue-session.md). The literature names the mechanism: at the
onset of on-policy fine-tuning the value function diverges because the critic
was never fit to on-policy data, and that divergence unlearns the pretrained
behaviour (WSRL, ICLR 2025). The published mitigation is a warmup phase that
runs rollouts from the FROZEN pretrained policy and fits the critic before any
actor update.

These tests pin the mechanism, not the reward. They use a synthetic scheme and
a real ReplayBuffer, so they do not need the simulator or a trained checkpoint.
"""

import numpy as np
import pytest


torch = pytest.importorskip("torch", reason="warmup learner imports torch")
pytest.importorskip("components.episode_buffer", reason="needs epymarl on path")


def _sys_path():
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = os.path.join(root, "third_party", "epymarl", "src")
    if src not in os.sys.path:
        os.sys.path.insert(0, src)
    return src


_sys_path()

from components.episode_buffer import ReplayBuffer  # noqa: E402
from components.transforms import OneHot  # noqa: E402
from controllers import REGISTRY as mac_REGISTRY  # noqa: E402
from learners.ppo_learner import PPOLearner  # noqa: E402
from types import SimpleNamespace as SN  # noqa: E402


N_AGENTS = 2
N_ACTIONS = 4
OBS_SHAPE = 6
STATE_SHAPE = 12
T_LEN = 5


def _args(**over):
    base = dict(
        n_agents=N_AGENTS, n_actions=N_ACTIONS, use_cuda=False,
        standardise_returns=False, standardise_rewards=False,
        common_reward=True, gamma=0.99, lr=0.01, grad_norm_clip=10,
        eps_clip=0.2, entropy_coef=0.001, epochs=2, q_nstep=2,
        add_value_last_step=True, target_update_interval_or_tau=0.01,
        learner_log_interval=10 ** 9, hidden_dim=8, use_rnn=True,
        obs_agent_id=True, obs_last_action=False, obs_individual_obs=False,
        agent_output_type="pi_logits", agent="rnn", name="test",
        action_selector="soft_policies", mask_before_softmax=True,
        mac="basic_mac", critic_type="cv_critic", learner="ppo_learner",
        warmup_steps=0, warmup_only_critic=True,
    )
    base.update(over)
    return SN(**base)


def _make_learner(logger=None, **over):
    scheme = {
        "state": {"vshape": STATE_SHAPE},
        "obs": {"vshape": OBS_SHAPE, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (N_ACTIONS,), "group": "agents",
                          "dtype": torch.int},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "reward": {"vshape": (1,)},
    }
    groups = {"agents": N_AGENTS}
    args = _args(**over)
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=N_ACTIONS)])}
    # ReplayBuffer mutates the scheme dict it is given (adds the reserved
    # "filled" field and the preprocessed "actions_onehot"), so snapshot the
    # pristine scheme first for later episode-batch construction.
    episode_scheme = {k: dict(v) for k, v in scheme.items()}
    buffer = ReplayBuffer(scheme, groups, 4, T_LEN + 1,
                          preprocess=preprocess, device="cpu")
    mac = mac_REGISTRY["basic_mac"](buffer.scheme, groups, args)
    if logger is None:
        class _L:
            def log_stat(self, *a, **k):
                pass
        logger = _L()
    return PPOLearner(mac, buffer.scheme, logger, args), buffer, args, episode_scheme


def _fill(buffer, episode_scheme, seed=0, n=2):
    """Insert n random episodes so buffer.can_sample(2) is true."""
    rng = np.random.RandomState(seed)
    for _ in range(n):
        batch = ReplayBuffer(episode_scheme, buffer.groups, 1, T_LEN + 1,
                             preprocess=buffer.preprocess, device="cpu")
        batch.update({
            "state": [rng.randn(STATE_SHAPE).astype(np.float32)],
            "obs": [rng.randn(N_AGENTS, OBS_SHAPE).astype(np.float32)],
            "avail_actions": [np.ones((N_AGENTS, N_ACTIONS), dtype=np.int64)],
        }, ts=0)
        for t in range(T_LEN):
            batch.update({
                "actions": rng.randint(0, N_ACTIONS,
                                       size=(N_AGENTS, 1)).tolist(),
                "reward": [(rng.randn(),)],
                "terminated": [[0]],
            }, ts=t + 1)
        last = rng.randint(0, N_ACTIONS, size=(N_AGENTS, 1)).tolist()
        batch.update({"actions": last}, ts=T_LEN)
        buffer.insert_episode_batch(batch)
    return buffer.sample(2)


def _flat_params(module):
    return torch.cat([p.detach().reshape(-1).clone() for p in module.parameters()])


def test_warmup_freezes_actor_but_still_trains_critic():
    """Core contract: during warmup the actor must be bit-identical while the
    critic actually moves. If either half regresses, the warmup is useless."""
    learner, buffer, _, escheme = _make_learner(warmup_steps=10 ** 6)
    batch = _fill(buffer, escheme)

    actor_before = _flat_params(learner.mac)
    critic_before = _flat_params(learner.critic)

    learner.train(batch, t_env=0, episode_num=0)

    actor_after = _flat_params(learner.mac)
    critic_after = _flat_params(learner.critic)

    assert torch.equal(actor_before, actor_after), "actor moved during warmup"
    assert not torch.equal(critic_before, critic_after), "critic did not train"


def test_actor_trains_again_after_warmup_boundary():
    """After t_env crosses warmup_steps the actor must resume updating, or the
    policy would stay frozen for the whole run."""
    learner, buffer, _, escheme = _make_learner(warmup_steps=10)
    batch = _fill(buffer, escheme)

    before = _flat_params(learner.mac)
    learner.train(batch, t_env=50, episode_num=0)
    after = _flat_params(learner.mac)
    assert not torch.equal(before, after), "actor never unfroze after warmup"


def test_warmup_disabled_by_default_is_a_no_op_flag():
    """warmup_steps=0 must never freeze the actor (back-compat with the four
    already-measured fine-tunes, which had no warmup)."""
    learner, buffer, _, escheme = _make_learner(warmup_steps=0)
    batch = _fill(buffer, escheme)
    before = _flat_params(learner.mac)
    learner.train(batch, t_env=0, episode_num=0)
    assert not torch.equal(before, _flat_params(learner.mac))


def test_opt_out_allows_actor_updates_during_warmup():
    """--no-warmup-critic-only restores full PPO inside the warmup window;
    this is the ablation arm that isolates the freeze as the causal factor."""
    learner, buffer, _, escheme = _make_learner(warmup_steps=10 ** 6,
                                                warmup_only_critic=False)
    batch = _fill(buffer, escheme)
    before = _flat_params(learner.mac)
    learner.train(batch, t_env=0, episode_num=0)
    assert not torch.equal(before, _flat_params(learner.mac))


def test_driver_config_defaults_are_off():
    """A run must be unchanged unless warmup is explicitly requested."""
    pytest.importorskip("blue_train_mappo", reason="train driver needs torch")
    from blue_train_mappo import build_config
    assert build_config()["warmup_steps"] == 0
    assert build_config()["warmup_only_critic"] is True
    cfg = build_config(warmup_steps=5000, warmup_critic_only=True)
    assert cfg["warmup_steps"] == 5000
