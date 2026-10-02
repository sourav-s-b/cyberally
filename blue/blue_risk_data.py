"""Collect per-host risk dataset: Blue-visible 17-feat rows + privileged labels.

Each row: 10 base features (host_to_vector on the merged Blue view) + ages
(2) + belief one-hot (5), i.e. the ("ages", "belief") actor geometry.
Label: 1 iff the host has a red session in TRUE state at that tick
(privileged; training target only -- never a policy input).
Meta: seed, tick, agent, host. Episodes run a sweep policy so the feature
distribution matches deployment.

Writes results/risk_data_<id>.npz + .json (gitignored artifacts).
"""
import argparse
import json
import os

import numpy as np

import cc4_epymarl_wrapper as wrapper
from blue_baselines import RoundRobinBaseline
from blue_obs_features import host_to_vector, host_to_temporal

GROUPS = ("ages", "belief")
N_BASE = 10 + 2 + 5
# View deltas since the previous tick (lancer's Monitor-novelty signal in
# feature form): d_unknown_files, d_max_density, d_n_conn, d_n_ext.
N_DELTAS = 4
N_FEATS = N_BASE + N_DELTAS
_SIG_IDX = (6, 7, 8, 9)  # positions inside the 10 base features


def true_compromised(env):
    """Privileged set of hostnames with a red session (eval/train label)."""
    controller = env.env
    true_state = controller.get_true_state(controller.INFO_DICT["True"]).data
    out = set()
    for host, host_obs in true_state.items():
        if not isinstance(host_obs, dict):
            continue
        sessions = host_obs.get("Sessions", []) or []
        if any("red" in str(s.get("agent", "")) for s in sessions):
            out.add(host)
    return out


def collect_episode(seed, steps=400, policy=None):
    env = wrapper.CC4MARLEnv(seed=seed, steps=steps, mask_mode="validity",
                             temporal_features=GROUPS,
                             include_root_session=True)
    env.reset(seed=seed)
    policy = policy or RoundRobinBaseline()
    policy.reset()
    rows, labels, meta = [], [], []
    prev_sig = {}
    ticks = 0
    for tick in range(1, steps + 1):
        acts = {agent: int(policy.select(env, agent))
                for agent in wrapper.BLUE_AGENTS}
        bad = true_compromised(env)
        for i, agent in enumerate(wrapper.BLUE_AGENTS):
            tracker = env.trackers[agent]
            busy = agent in env._awaiting
            for host in env.hostnames[agent]:
                row = host_to_vector(env.views[agent][host], agent,
                                     env.subnets[agent][host])
                row += host_to_temporal(tracker, host,
                                        env.observed_at[agent][host],
                                        env._tick, env.episode_limit, busy,
                                        GROUPS)
                sig = tuple(row[j] for j in _SIG_IDX)
                key = (agent, host)
                old = prev_sig.get(key, sig)
                row += [s - o for s, o in zip(sig, old)]
                prev_sig[key] = sig
                rows.append(row)
                labels.append(1 if host in bad else 0)
                meta.append((seed, tick, agent, host))
        _, _, terminated, truncated, _ = env.step(acts)
        ticks += 1
        if terminated or truncated:
            break
    return {"X": np.array(rows, dtype=np.float32),
            "y": np.array(labels, dtype=np.int64),
            "meta": meta, "seed": seed, "ticks": ticks}


def main():
    ap = argparse.ArgumentParser(description="Collect risk-model dataset")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[7901, 7902, 7903, 7904, 7905, 7906, 7907, 7908])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="results/risk_data_v1.npz")
    cli = ap.parse_args()
    Xs, ys, tallies = [], [], {}
    for s in cli.seeds:
        e = collect_episode(s, steps=cli.steps)
        Xs.append(e["X"])
        ys.append(e["y"])
        tallies[str(s)] = {"rows": len(e["y"]), "pos": int(e["y"].sum()),
                           "ticks": e["ticks"]}
        print(f"seed={s}: rows={len(e['y'])} pos={int(e['y'].sum())} "
              f"({e['y'].mean() * 100:.2f}%) ticks={e['ticks']}")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    assert X.shape[1] == N_FEATS, X.shape
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    np.savez_compressed(cli.out, X=X, y=y)
    meta = {"seeds": cli.seeds, "steps": cli.steps, "n_feats": N_FEATS,
            "groups": list(GROUPS), "include_root_session": True,
            "policy": "RoundRobinBaseline",
            "pos_rate": float(y.mean()), "n_rows": len(y), "tallies": tallies}
    with open(os.path.splitext(cli.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {cli.out}: {len(y)} rows, pos_rate={y.mean():.4f}")


if __name__ == "__main__":
    main()
