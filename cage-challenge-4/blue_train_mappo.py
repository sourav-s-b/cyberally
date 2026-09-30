"""First real MAPPO training driver: CC4 Blue inside EPyMARL's own runner.

All learning machinery is upstream EPyMARL (pinned commit in
../third_party/epymarl/PIN.txt): EpisodeRunner, ReplayBuffer, BasicMAC
(shared RNN actor), PPOLearner with central-V critic. This file contributes
only the config (values from EPyMARL's own mappo.yaml/default.yaml unless
noted), env registration, the smaclite dead-import stub, torch seeding, and
a run manifest. No training hyperparameter is hardcoded from stale docs:
network shapes come from the env's get_env_info() inside run_sequential.

Smoke defaults: 8 episodes x 100 steps on CPU, common team reward, seeds
cycled per episode from train_seeds (first reset takes the first entry;
EPyMARL's runner calls bare reset() every episode). Every applied seed is
recorded in the manifest's reset_seeds. Checkpoints + manifest land under
results/ (gitignored).
"""

import argparse
import json
import os
import random
import sys
import types
from datetime import datetime, timezone
from types import SimpleNamespace as SN

import numpy as np
import torch as th

# Tiny actor/critic on a 12-thread laptop: cap BLAS threads so torch does not
# oversubscribe the CPU while the (single-threaded) simulator steps.
th.set_num_threads(2)

try:
    import smaclite  # noqa
except ImportError:
    # EPyMARL's envs/__init__ imports smaclite_wrapper (SMAC-only, unused:
    # we register our own env below). smaclite ships only as from-source
    # GitHub, not PyPI, so stub the dead import instead of vendoring a
    # StarCraft env into a cyber venv. Any real use fails loudly.
    _stub = types.ModuleType("smaclite")
    _stub.__file__ = "<smaclite-stub: SMAC envs unsupported>"

    def _missing(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise ImportError(
            "smaclite is not installed: SMAC envs are unsupported here; "
            "install from https://github.com/uoe-agents/smaclite if needed"
        )

    _stub.__getattr__ = _missing
    sys.modules["smaclite"] = _stub

from envs import REGISTRY as env_REGISTRY  # noqa: E402
from run import run_sequential  # noqa: E402
from utils.logging import Logger, get_logger  # noqa: E402

import cc4_epymarl_wrapper as wrapper  # noqa: E402

EPYMARL_PIN = "cbc38c09588064eab978501d0f12c2cf58fa7fc2"

# run_sequential builds its own training env internally through the factory
# below, so stash the live instance to read back env.reset_seeds for the
# manifest after the run. Exactly one env exists at a time (the probe below
# is constructed directly and closed before training starts).
_train_env_ref = {}


def _make_train_env(**kw):
    env = wrapper.CC4MARLEnv(**kw)
    _train_env_ref["env"] = env
    return env


env_REGISTRY["cc4"] = _make_train_env


def build_config(steps=100, t_max=800, seed=7, results="results",
                 train_seeds=(7629, 7630, 7640), save_interval=2000):
    """MAPPO config; algorithm keys mirror EPyMARL's mappo.yaml."""
    return {
        "name": "mappo_cc4",
        "runner": "episode",
        "mac": "basic_mac",
        "env": "cc4",
        "env_args": {"seed": train_seeds[0], "steps": steps,
                     "mask_mode": "validity",
                     "seed_cycle": list(train_seeds)},
        "common_reward": True,
        "reward_scalarisation": "sum",
        "batch_size_run": 1,
        "batch_size": 2,
        "buffer_size": 4,
        "buffer_cpu_only": True,
        "test_nepisode": 1,
        "test_interval": t_max + 1,  # smoke: train only, one greedy probe below
        "test_greedy": True,
        "log_interval": 200,
        "runner_log_interval": 100,
        "learner_log_interval": 100,
        "t_max": t_max,
        "use_cuda": False,
        "gamma": 0.99,
        "lr": 0.0003,
        "grad_norm_clip": 10,
        "add_value_last_step": True,
        "agent": "rnn",
        "use_rnn": True,
        "hidden_dim": 64,
        "obs_agent_id": True,
        "obs_last_action": False,
        "obs_individual_obs": False,
        "agent_output_type": "pi_logits",
        "action_selector": "soft_policies",
        "mask_before_softmax": True,
        "learner": "ppo_learner",
        "critic_type": "cv_critic",
        "epochs": 4,
        "eps_clip": 0.2,
        "q_nstep": 5,
        "entropy_coef": 0.001,
        "standardise_returns": False,
        "standardise_rewards": True,
        "target_update_interval_or_tau": 0.01,
        "use_tensorboard": False,
        "use_wandb": False,
        "save_model": True,
        "save_model_interval": save_interval,  # periodic + first checkpoints
        "checkpoint_path": "",
        "evaluate": False,
        "render": False,
        "load_step": 0,
        "save_replay": False,
        "local_results_path": results,
        "seed": seed,
    }


def train(config=None):
    config = dict(config or build_config())
    token = "{}_seed{}_{}".format(
        config["name"], config["seed"],
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    run_dir = os.path.join(config["local_results_path"], token)
    os.makedirs(run_dir, exist_ok=True)

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])

    args = SN(**config)
    args.device = "cuda" if config["use_cuda"] and th.cuda.is_available() else "cpu"
    args.unique_token = token

    logger = Logger(get_logger())

    # One throwaway construction for manifest provenance; run_sequential
    # builds its own training env from the same env_args. Deleted before
    # training so only one simulator is ever alive (laptop memory safety).
    probe = wrapper.CC4MARLEnv(**config["env_args"])
    env_info = probe.get_env_info()
    probe.close()
    del probe
    manifest = {
        "token": token,
        "config": config,
        "env_info": {k: (list(v) if isinstance(v, np.ndarray) else v)
                     for k, v in env_info.items()},
        "wrapper_version": wrapper.WRAPPER_VERSION,
        "epymarl_pin": EPYMARL_PIN,
        "torch_version": th.__version__,
        "seeding": {
            "torch_numpy_random": config["seed"],
            "seed_cycle": config["env_args"]["seed_cycle"],
            "env_first_reset": config["env_args"]["seed_cycle"][0],
            "reset_seeds": None,  # filled post-run from the training env
        },
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    run_sequential(args=args, logger=logger)

    train_env = _train_env_ref.get("env")
    manifest["seeding"]["reset_seeds"] = (
        list(train_env.reset_seeds) if train_env is not None else None)
    manifest["seeding"]["n_episodes"] = len(manifest["seeding"]["reset_seeds"] or [])
    manifest["seeding"]["note"] = ("chronological per env.reset(), train and "
        "greedy test episodes interleaved; test episodes also consume cycle "
        "slots")
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    stats = {k: [(int(t), float(v)) for t, v in vals]
             for k, vals in logger.stats.items()}
    with open(os.path.join(run_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("run dir:", run_dir)
    return run_dir


if __name__ == "__main__":
    # Guarded for Windows spawn (future parallel runner) and import safety.
    parser = argparse.ArgumentParser(description="MAPPO Blue smoke/short runs")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--t-max", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-seeds", type=int, nargs="+",
                        default=[7629, 7630, 7640])
    parser.add_argument("--save-interval", type=int, default=2000)
    cli = parser.parse_args()
    train(build_config(steps=cli.steps, t_max=cli.t_max, seed=cli.seed,
                       train_seeds=tuple(cli.train_seeds),
                       save_interval=cli.save_interval))
