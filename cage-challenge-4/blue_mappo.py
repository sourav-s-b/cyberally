"""Train Blue using the pinned EPyMARL EpisodeRunner, BasicMAC and PPOLearner.

Fresh on-policy episode batches only. Actors see local Blue observations; the
central value critic sees concatenated Blue observations, never simulator truth.
Upstream EPyMARL remains unmodified and retains its own license in its checkout.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from types import SimpleNamespace

EPYMARL_COMMIT = "cbc38c09588064eab978501d0f12c2cf58fa7fc2"
ROOT = Path(__file__).resolve().parents[1]


def git_revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_options(options):
    if options.updates < 1 or options.episodes_per_update < 1:
        raise ValueError("updates and episodes-per-update must be positive")
    if options.steps < 3:
        raise ValueError("steps must be at least 3")
    if options.threads < 1:
        raise ValueError("threads must be positive")
    train_seeds = list(range(options.seed, options.seed +
                            options.updates * options.episodes_per_update))
    if set(train_seeds) & set(options.eval_seeds):
        raise ValueError("training and evaluation seeds must be disjoint")
    if len(set(options.eval_seeds)) != len(options.eval_seeds):
        raise ValueError("evaluation seeds must be unique")
    return train_seeds


class Metrics:
    def __init__(self):
        self.latest = {}

    def log_stat(self, key, value, step):
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError(f"Nonfinite learner metric: {key}={value}")
        self.latest[key] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epymarl-root", type=Path, default=ROOT / ".tools/epymarl")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" /
                        time.strftime("mappo-%Y%m%d-%H%M%S"))
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7629)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[90001, 90002])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--evaluate-manifest", type=Path,
                        help="Verify and evaluate a saved checkpoint without training")
    options = parser.parse_args(argv)
    train_seeds = validate_options(options)
    epymarl = options.epymarl_root.resolve()
    if git_revision(epymarl) != EPYMARL_COMMIT:
        raise ValueError(f"EPyMARL must be checked out at {EPYMARL_COMMIT}")
    dirty_framework = subprocess.check_output(
        ["git", "-C", str(epymarl), "status", "--porcelain"], text=True)
    if dirty_framework.strip():
        raise ValueError("EPyMARL checkout must be clean")
    sys.path.insert(0, str(epymarl / "src"))

    import numpy as np
    import torch
    import yaml
    from components.episode_buffer import EpisodeBatch
    from components.transforms import OneHot
    from controllers.basic_controller import BasicMAC
    from envs import REGISTRY
    from learners.ppo_learner import PPOLearner
    from runners.episode_runner import EpisodeRunner
    from cc4_epymarl_wrapper import CC4MARLEnv, WRAPPER_VERSION

    class CheckedEnv(CC4MARLEnv):
        """Fail on illegal policy actions instead of using wrapper fallback."""
        def step(self, actions):
            indices = np.asarray(actions, dtype=np.int64).reshape(-1)
            if len(indices) != self.n_agents:
                raise RuntimeError("Expected one action per Blue agent")
            for agent, action in enumerate(indices):
                mask = self.get_avail_agent_actions(agent)
                if action < 0 or action >= len(mask) or not mask[action]:
                    raise RuntimeError(f"Illegal action {action} for agent {agent}")
            return super().step(actions)

    REGISTRY["cage4_blue"] = CheckedEnv
    config = yaml.safe_load((epymarl / "src/config/default.yaml").read_text())
    config.update(yaml.safe_load((epymarl / "src/config/algs/mappo.yaml").read_text()))
    config.update(env="cage4_blue", runner="episode", batch_size_run=1,
                  batch_size=options.episodes_per_update, device="cpu",
                  use_cuda=False, seed=options.seed, learner_log_interval=1,
                  runner_log_interval=1, test_nepisode=1,
                  env_args={"seed": options.seed, "steps": options.steps,
                            "mask_mode": "validity", "per_agent_bounds": False})
    saved = None
    if options.evaluate_manifest:
        saved = json.loads(options.evaluate_manifest.read_text(encoding="utf-8"))
        if (saved["schema_version"] != "blue-mappo-smoke-v1" or
                saved["epymarl_commit"] != EPYMARL_COMMIT or
                saved["wrapper_version"] != WRAPPER_VERSION):
            raise ValueError("Incompatible policy manifest")
        if set(saved["train_seeds"]) & set(options.eval_seeds):
            raise ValueError("training and evaluation seeds must be disjoint")
        config = saved["config"]
        if (config["env"] != "cage4_blue" or not config["common_reward"] or
                config["env_args"]["per_agent_bounds"]):
            raise ValueError("Incompatible policy environment/configuration")
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if config_hash != saved["config_sha256"]:
            raise ValueError("Policy config hash mismatch")
        for relative, digest in saved["source_sha256"].items():
            # Trainer source may change; policy feature/action semantics must match.
            if Path(relative).name in ("cc4_epymarl_wrapper.py", "blue_obs_features.py",
                                      "blue_action_masking.py"):
                if sha256(ROOT / relative) != digest:
                    raise ValueError(f"Policy source mismatch: {relative}")
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.set_num_threads(options.threads)
    args = SimpleNamespace(**config)
    logger = Metrics()
    runner = EpisodeRunner(args, logger)
    info = runner.get_env_info()
    args.n_agents, args.n_actions = info["n_agents"], info["n_actions"]
    args.state_shape = info["state_shape"]
    scheme = {
        "state": {"vshape": info["state_shape"]},
        "obs": {"vshape": info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (info["n_actions"],), "group": "agents",
                          "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(args.n_actions)])}
    template = EpisodeBatch(scheme, groups, 1, info["episode_limit"] + 1,
                            preprocess=preprocess)
    mac = BasicMAC(template.scheme, groups, args)
    runner.setup(scheme, groups, preprocess, mac)
    learner = PPOLearner(mac, template.scheme, logger, args)

    def evaluate(policy):
        runner.mac = policy
        results = []
        for seed in options.eval_seeds:
            runner.env.seed(seed)
            # Separate evaluation RNG from training history; repeatable evaluation.
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            begin = time.perf_counter()
            episode = runner.run(test_mode=True)
            results.append({"seed": seed, "policy_seed": seed,
                            "joint_steps": runner.t,
                            "native_return": float(episode["reward"].sum()),
                            "seconds": time.perf_counter() - begin})
        runner.close_env()
        return results

    if saved is not None:
        for key in ("n_agents", "n_actions", "obs_shape", "state_shape", "episode_limit"):
            if info[key] != saved["environment_info"][key]:
                raise ValueError(f"Policy environment dimension mismatch: {key}")
        checkpoint = options.evaluate_manifest.resolve().parent / "checkpoint"
        expected_files = {"agent.th", "critic.th", "agent_opt.th", "critic_opt.th"}
        if set(saved["checkpoint_sha256"]) != expected_files:
            raise ValueError("Incomplete checkpoint manifest")
        for name, digest in saved["checkpoint_sha256"].items():
            if sha256(checkpoint / name) != digest:
                raise ValueError(f"Checkpoint hash mismatch: {name}")
        learner.load_models(str(checkpoint))
        result = {"policy_manifest": str(options.evaluate_manifest.resolve()),
                  "evaluation": evaluate(mac)}
        output = options.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        return result

    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    timings = {"rollout_seconds": 0.0, "update_seconds": 0.0}
    all_episode_returns = []
    seed_cursor = 0
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        for update in range(options.updates):
            # No replay buffer: all episodes collected from the current actor.
            episodes = []
            for _ in range(options.episodes_per_update):
                runner.env.seed(train_seeds[seed_cursor])
                seed_cursor += 1
                begin = time.perf_counter()
                episode = runner.run(test_mode=False)
                timings["rollout_seconds"] += time.perf_counter() - begin
                if runner.t != options.steps - 1:
                    raise RuntimeError(f"Unexpected episode length: {runner.t}")
                episodes.append(episode)
                all_episode_returns.append(float(episode["reward"].sum()))
            batch = EpisodeBatch(scheme, groups, len(episodes),
                                 episodes[0].max_seq_length, preprocess=preprocess)
            for index, episode in enumerate(episodes):
                for field in batch.data.transition_data:
                    batch.data.transition_data[field][index:index + 1].copy_(
                        episode.data.transition_data[field])
            batch = batch[:, :int(batch.max_t_filled())]
            actor_before = [p.detach().clone() for p in mac.parameters()]
            critic_before = [p.detach().clone() for p in learner.critic.parameters()]
            begin = time.perf_counter()
            learner.train(batch, runner.t_env, seed_cursor)
            timings["update_seconds"] += time.perf_counter() - begin
            for name, before, parameters in (
                ("actor", actor_before, list(mac.parameters())),
                ("critic", critic_before, list(learner.critic.parameters())),
            ):
                if not all(torch.isfinite(p).all() for p in parameters):
                    raise RuntimeError(f"Nonfinite {name} parameters")
                if not any(not torch.equal(old, new) for old, new in zip(before, parameters)):
                    raise RuntimeError(f"{name} parameters did not update")
            record = {"update": update + 1, "joint_steps": runner.t_env,
                      "episode_returns": all_episode_returns[-len(episodes):],
                      "metrics": dict(logger.latest), **timings}
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(json.dumps(record), flush=True)

    checkpoint = output / "checkpoint"
    checkpoint.mkdir()
    learner.save_models(str(checkpoint))
    fresh_mac = BasicMAC(template.scheme, groups, args)
    fresh_learner = PPOLearner(fresh_mac, template.scheme, logger, args)
    fresh_learner.load_models(str(checkpoint))
    # Upstream load_models does not synchronize old_mac; evaluation uses mac.
    for saved, loaded in zip(mac.parameters(), fresh_mac.parameters()):
        if not torch.equal(saved, loaded):
            raise RuntimeError("Checkpoint actor mismatch")
    for saved, loaded in zip(learner.critic.parameters(), fresh_learner.critic.parameters()):
        if not torch.equal(saved, loaded):
            raise RuntimeError("Checkpoint critic mismatch")
    evaluation = evaluate(fresh_mac)

    source_files = [Path(__file__), ROOT / "cage-challenge-4/cc4_epymarl_wrapper.py",
                    ROOT / "cage-challenge-4/blue_obs_features.py",
                    ROOT / "cage-challenge-4/blue_action_masking.py"]
    dependencies = {d.metadata["Name"]: d.version
                    for d in importlib.metadata.distributions()}
    manifest = {
        "schema_version": "blue-mappo-smoke-v1", "algorithm": "EPyMARL MAPPO",
        "source_commit": git_revision(ROOT),
        "source_status": subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        "epymarl_commit": EPYMARL_COMMIT, "python": sys.version,
        "platform": platform.platform(), "processor": platform.processor(),
        "torch_threads": options.threads,
        "dependencies": dependencies, "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "wrapper_version": WRAPPER_VERSION, "preprocessing": "host-vector-10-v1",
        "action_mapping": "Sleep, Monitor, sorted-host Analyse/Remove/Restore",
        "environment_info": info, "scenario": "native CAGE4 enterprise",
        "red_policy": "DiscoveryFSRed", "train_seeds": train_seeds,
        "evaluation_seeds": options.eval_seeds,
        "evaluation_action_selection": "upstream soft_policies stochastic",
        "checkpoint_sha256": {p.name: sha256(p) for p in checkpoint.iterdir()},
        "checkpoint_path": str(checkpoint), "joint_steps": runner.t_env,
        "train_episode_returns": all_episode_returns, "evaluation": evaluation,
        "metrics_sha256": sha256(output / "metrics.jsonl"),
        "timings": {**timings, "total_seconds": time.perf_counter() - started},
        "limitations": ["Smoke/short-run evidence, no claim of RL improvement",
                         "Exact training resume is not implemented; upstream save omits reward normalizer",
                         "Evaluation uses native returns only; full incident metrics belong to Evaluation"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MAPPO rollout/update/reload/evaluation passed: {output}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
