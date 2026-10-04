"""Phase 3: imitation-init the host scorer (offline, fixed budget).

Collect: run the reference OrderedPolicy (lancer config) and record, at
each sweep-decision tick, the candidate host feature rows + picked index.
Train: small MLP scorer (shared weights across hosts) with cross-entropy
over candidates, trajectory-blocked validation split (V1 hygiene only --
early stopping is NOT validated, fixed epoch budget regardless).
Eval: argmax agreement rate (scorer-argmax == heuristic pick) on held-out
episodes, reported ALONGSIDE the exact-bypass trace test (they are not
the same thing: the bypass is rules+reference scorer, agreement measures
the learned clone).

Usage (repo root):
  .../python -m blue.training.scorer --mode collect --seeds 7629 ... --out blue/results/scorer_rows.npz
  .../python -m blue.training.scorer --mode train --data blue/results/scorer_rows.npz --out blue/results/scorer_mlp.npz
  .../python -m blue.training.scorer --mode eval --data blue/results/scorer_rows.npz --model blue/results/scorer_mlp.npz
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blue.common import logutil


def host_rows(env, agent, cands):
    """10 base view features + 4 history features (all Blue-visible):
    analysis age, remediation age (normalized), CONFIRMED/VERIFY flags."""
    from blue.policies.hybrid import host_risk_features
    tracker = env.trackers[agent]
    tick = env._tick
    rows = []
    for h in cands:
        base = list(host_risk_features(env, agent, h))
        last_an = tracker.last_analysis.get(h)
        last_re = tracker.last_remediation.get(h)
        T = max(env.episode_limit if hasattr(env, "episode_limit") else 400,
                1)
        age_an = (tick - last_an) / T if last_an is not None else 1.0
        age_re = (tick - last_re) / T if last_re is not None else 1.0
        state = tracker.state.get(h)
        rows.append(base + [age_an, age_re,
                            1.0 if state == "CONFIRMED" else 0.0,
                            1.0 if state == "VERIFY" else 0.0])
    return rows


def cmd_collect(seeds, steps, out, **env_kwargs):
    import numpy as np
    from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
    from blue.core.baselines import action_index, decode_index
    from blue.policies.ordered import LancerValues, OrderedPolicy
    feats, picks, counts, eps = [], [], [], []
    prog = logutil.Progress(len(seeds))
    for n, seed in enumerate(seeds, 1):
        env = CC4MARLEnv(seed=seed, steps=steps, **env_kwargs)
        env.reset(seed=seed)
        policy = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5))
        policy.reset()
        ep = int(seed)
        for tick in range(1, steps + 1):
            actions = {a: int(policy.select(env, a)) for a in BLUE_AGENTS}
            # Record rows BEFORE stepping: candidate features and masks
            # must be the decision-time views the pick was made on.
            for agent in BLUE_AGENTS:
                idx = actions[agent]
                if idx < 2:
                    continue
                name, host = decode_index(env, agent, idx)
                if name != "Analyse" or host is None:
                    continue
                mask = env.get_avail_agent_actions(
                    BLUE_AGENTS.index(agent))
                cands = [h for h in env.hostnames[agent]
                         if bool(mask[action_index(env, agent, h,
                                                   "Analyse")])]
                if host not in cands:
                    continue
                feats.append(host_rows(env, agent, cands))
                picks.append(cands.index(host))
                counts.append(len(cands))
                eps.append(ep)
            _, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break
        print(f"collect {prog.line(n)} seed={seed} rows={len(feats)}",
              flush=True)
    max_c = max(counts)
    dim = len(feats[0][0])
    F = np.zeros((len(feats), max_c, dim), dtype=np.float32)
    M = np.zeros((len(feats), max_c), dtype=np.float32)
    for i, (rows, c) in enumerate(zip(feats, counts)):
        F[i, :c] = np.asarray(rows, dtype=np.float32)
        M[i, :c] = 1.0
    np.savez(out, feats=F, masks=M,
             picks=np.asarray(picks, dtype=np.int64),
             episodes=np.asarray(eps, dtype=np.int64))
    print(f"wrote {out}: {len(feats)} rows, max_cands {max_c}")


def _mlp(d_in, hidden=64):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))


def cmd_train(data, out, epochs=20, hidden=64, seed=0, val_frac=0.2):
    import numpy as np
    import torch as th
    from blue.common.val_split import blocked_split, EarlyStopper
    th.manual_seed(seed)
    np.random.seed(seed)
    d = np.load(data)
    F, M, P, E = d["feats"], d["masks"], d["picks"], d["episodes"]
    uniq = np.unique(E)
    tr_eps, va_eps = blocked_split(len(uniq), val_frac, seed)
    tr_eps, va_eps = set(uniq[tr_eps]), set(uniq[va_eps])
    tri = np.array([i for i, e in enumerate(E) if e in tr_eps])
    vai = np.array([i for i, e in enumerate(E) if e not in tr_eps])
    net = _mlp(F.shape[2], hidden)
    opt = th.optim.Adam(net.parameters(), lr=1e-3)
    stopper = EarlyStopper(patience=10**9)  # hygiene split only; fixed budget
    prog = logutil.Progress(epochs)
    hist = []
    for epoch in range(1, epochs + 1):
        net.train()
        perm = tri[np.random.permutation(len(tri))]
        tot, n = 0.0, 0
        for s in range(0, len(perm), 256):
            idx = th.from_numpy(perm[s:s + 256])
            scores = net(th.from_numpy(F[idx])).squeeze(-1)
            scores = scores + (1.0 - th.from_numpy(M[idx])) * -1e9
            loss = th.nn.functional.cross_entropy(
                scores, th.from_numpy(P[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            n += len(idx)
        net.eval()
        with th.no_grad():
            va = th.from_numpy(vai)
            vs = net(th.from_numpy(F[va])).squeeze(-1)
            vs = vs + (1.0 - th.from_numpy(M[va])) * -1e9
            vl = float(th.nn.functional.cross_entropy(
                vs, th.from_numpy(P[va])))
            vacc = float((vs.argmax(-1).numpy() == P[va]).mean())
        stopper.update(epoch, vl, net)
        hist.append({"epoch": epoch, "train_loss": tot / n,
                     "val_loss": vl, "val_agree": vacc})
        print(f"epoch {epoch:3d} loss {tot / n:.4f} val-loss {vl:.4f} "
              f"val-agree {vacc:.4f} {prog.line(epoch)}", flush=True)
    stopper.restore(net)
    th.save(net.state_dict(), out)
    with open(out + ".json", "w") as f:
        json.dump({"data": data, "epochs": epochs, "hidden": hidden,
                   "seed": seed, "val_eps": sorted(int(e) for e in va_eps),
                   "final_val_agree": vacc, "final_val_loss": vl,
                   "hist": hist}, f, indent=1)
    print(f"wrote {out} val-agree {vacc:.4f}")


def cmd_eval(data, model, hidden=64):
    import numpy as np
    import torch as th
    d = np.load(data)
    F, M, P = d["feats"], d["masks"], d["picks"]
    net = _mlp(F.shape[2], hidden)
    net.load_state_dict(th.load(model, map_location="cpu"))
    net.eval()
    agree, per_ep = [], {}
    with th.no_grad():
        for s in range(0, len(F), 1024):
            sc = net(th.from_numpy(F[s:s + 1024])).squeeze(-1)
            sc = sc + (1.0 - th.from_numpy(M[s:s + 1024])) * -1e9
            pred = sc.argmax(-1).numpy()
            agree.extend(list(pred == P[s:s + 1024]))
    agree = np.asarray(agree, dtype=float)
    print(f"agreement: {agree.mean():.4f} on {len(agree)} rows "
          f"(argmax clone vs heuristic pick; NOT a bypass test)")


def main():
    ap = argparse.ArgumentParser(description="Phase 3 scorer imitation")
    ap.add_argument("--mode", required=True,
                    choices=("collect", "train", "eval"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[7629, 7630])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--data", default="blue/results/scorer_rows.npz")
    ap.add_argument("--model", default="blue/results/scorer_mlp.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    cli = ap.parse_args()
    env_kwargs = {"temporal_features": ("ages", "belief"),
                  "include_root_session": True, "red_agent": "discovery"}
    if cli.mode == "collect":
        cmd_collect(cli.seeds, cli.steps,
                    cli.out or "blue/results/scorer_rows.npz", **env_kwargs)
    elif cli.mode == "train":
        cmd_train(cli.data, cli.out or "blue/results/scorer_mlp.npz",
                  epochs=cli.epochs)
    else:
        cmd_eval(cli.data, cli.model)


if __name__ == "__main__":
    main()
