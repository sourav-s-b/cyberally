"""Long-horizon unattended trainer for the risk-feature actor.

Purpose: run longer than a laptop session allows, and stop on EVIDENCE of
plateau rather than on a wall-clock guess. Every iteration:

1. fresh joint on-policy rollouts (dev seeds, filtered informative
   decisions -- see docs/plans/risk-actor-next.md caveat: this is a
   selected-experience update, not standard full-buffer PPO);
2. PPO update;
3. checkpoint (actor, critic, optimizer, RNG) so a session kill loses
   nothing;
4. append one JSON line to metrics.jsonl (never truncated);
5. periodic paired dev evaluation vs Lancer, which is the ONLY thing that
   decides continuation.

Stop conditions (first hit wins), all recorded:
- plateau: no improvement of the best dev paired mean by >= --min-gain for
  --patience consecutive evals;
- --max-iters or --max-hours.

Reporting rules that this file exists to enforce:
- a cheap mid-run subset can NEVER declare a gate pass;
- an actor snapshot is kept at EVERY eval, because the last actor is a
  random draw from a noisy curve and must not be the only thing we keep;
- the final full-seed evaluation is run on BOTH the best subset checkpoint
  and the last actor, so a lucky stopping point cannot hide a bad one;
- progress is printed continuously (per episode, per iteration, plus a
  heartbeat) because a silent log is indistinguishable from a hang.

Everything is dev-seed training/evaluation. The reserved evaluation block
7809-8200 is never touched here.

Usage (train venv, repo root):
  .venv-train/bin/python -m blue.training.remote_train --out blue/results/remote1 \
      --iters 200 --eval-every 5 --patience 4 --min-gain 1.0 --max-hours 8
"""

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blue.training import mappo_guide as mg
from blue.training import risk_actor as ra

EVAL_SEEDS = [7629, 7630, 7640, 7701, 7702, 7703, 7704, 7705]
T = 2.365  # t critical, df=7 (paired dev eval)


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())     # survive a hard kill


class Progress(object):
    """Heartbeat so a long silent stretch is never mistaken for a hang.

    The simulator spends ~25 s inside one episode with no output at all, and
    ~12 min between evals. Both look exactly like a crash from the outside.
    This prints a timestamped line every `every` seconds regardless of what
    the main thread is doing.
    """

    def __init__(self, every=60.0, t0=None):
        self.every = float(every)
        self.t0 = t0 if t0 is not None else time.time()
        self.state = {"phase": "starting", "iter": 0, "ep": 0,
                      "ep_total": 0, "rows": 0, "best": None}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    def set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def line(self, msg):
        el = time.time() - self.t0
        with self._lock:
            s = dict(self.state)
        rows = []
        for k in ("iter", "ep", "ep_total", "rows", "best"):
            if s.get(k) is not None:
                rows.append("%s=%s" % (k, s[k]))
        print("[%7.1fs] %s | %s" % (el, msg, " ".join(rows)), flush=True)

    def _loop(self):
        while not self._stop.wait(self.every):
            try:
                self.line("heartbeat")
            except Exception:
                pass

    def start(self):
        if self.every > 0 and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()


def evaluate(actor, seeds, steps, bonus, temp_eval=0.0):
    """Paired dev evaluation vs Lancer. Greedy (temp 0) at eval."""
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy
    risk = ra._risk()
    base, greedy = [], []
    for seed in seeds:
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
        rb = mg.run_team_episode(pol, seed, steps, **mg.ENV_KW)
        hook = ra.RiskGreedyHook(actor, bonus, risk)
        rg = mg.run_team_episode(pol, seed, steps, hook=hook, **mg.ENV_KW)
        base.append(rb["return"])
        greedy.append(rg["return"])
    d = [g - b for g, b in zip(greedy, base)]
    m = float(np.mean(d))
    se = float(np.std(d, ddof=1)) / (len(d) ** 0.5)
    return {"paired_mean": m, "sd": float(np.std(d, ddof=1)),
            "ci95": [m - T * se, m + T * se],
            "wins": int(sum(1 for x in d if x > 0)), "n": len(d),
            "lancer_mean": float(np.mean(base)),
            "actor_mean": float(np.mean(greedy))}


def save_ckpt(out, actor, critic, opt, gen_state, it, track=None):
    """Write the resume checkpoint atomically.

    `track` carries the plateau bookkeeping (best score, stale count, which
    iteration set it). Without it a resumed run forgets its own best, treats
    the first eval after resuming as an improvement, and can therefore never
    plateau.
    """
    import torch as th
    tmp = os.path.join(out, ".ckpt.tmp")
    os.makedirs(tmp, exist_ok=True)
    th.save(actor.state_dict(), os.path.join(tmp, "actor.th"))
    th.save(critic.state_dict(), os.path.join(tmp, "critic.th"))
    th.save(opt.state_dict(), os.path.join(tmp, "opt.th"))
    th.save(gen_state, os.path.join(tmp, "rng.th"))
    state = {"iter": it}
    state.update(track or {})
    with open(os.path.join(tmp, "state.json"), "w") as f:
        json.dump(state, f)
    old = os.path.join(out, "latest")
    trash = os.path.join(out, ".ckpt.trash")
    # Rename-then-replace: a kill between the two leaves `latest` intact
    # under a different name instead of deleting the only good checkpoint.
    if os.path.isdir(old):
        subprocess.run(["rm", "-rf", trash], check=False)
        os.rename(old, trash)
    os.replace(tmp, old)
    subprocess.run(["rm", "-rf", trash], check=False)
    with open(os.path.join(out, "LATEST"), "w") as f:
        f.write(str(it))


def save_snapshot(out, actor, it, ev):
    """Keep the actor at every eval so the best moment is recoverable.

    Only the actor weights: the critic and optimizer are not needed to
    re-score a checkpoint, and these files are small.
    """
    import torch as th
    d = os.path.join(out, "snapshots", "iter_%04d" % it)
    os.makedirs(d, exist_ok=True)
    th.save(actor.state_dict(), os.path.join(d, "actor.th"))
    with open(os.path.join(d, "eval.json"), "w") as f:
        json.dump({"iter": it, "eval": ev}, f, indent=1)
    return d


def load_actor(out, ckpt_dir, hidden):
    import torch as th
    a = ra.build_actor(hidden=hidden)
    a.load_state_dict(th.load(os.path.join(ckpt_dir, "actor.th"),
                              map_location="cpu", weights_only=False))
    return a


def main():
    ap = argparse.ArgumentParser(description="unattended long trainer")
    ap.add_argument("--out", default="blue/results/remote1")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--eps-per-iter", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, nargs="*",
                    default=list(EVAL_SEEDS))
    ap.add_argument("--eval-subset", type=int, default=4,
                    help="dev seeds used for the mid-run gate check")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--train-min-proba", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-gain", type=float, default=1.0)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--gate-min-seeds", type=int, default=8,
                    help="refuse to declare a gate pass on fewer seeds")
    ap.add_argument("--gate-min-steps", type=int, default=400,
                    help="refuse to declare a gate pass on shorter episodes")
    ap.add_argument("--min-rows", type=int, default=8,
                    help="skip the PPO update below this many usable rows")
    ap.add_argument("--log-every", type=float, default=60.0,
                    help="heartbeat seconds; 0 disables")
    ap.add_argument("--resume", action="store_true")
    cli = ap.parse_args()

    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy

    os.makedirs(cli.out, exist_ok=True)
    metrics_path = os.path.join(cli.out, "metrics.jsonl")
    actor = ra.build_actor(hidden=cli.hidden)
    critic = mg.build_central_critic(hidden=cli.hidden)
    opt = th.optim.Adam(list(actor.parameters())
                        + list(critic.parameters()), lr=cli.lr)
    start_it = 1
    best = -1e9
    best_it = 0
    best_ckpt = None
    stale = 0
    evals_seen = 0
    if cli.resume and os.path.exists(os.path.join(cli.out, "latest")):
        d = os.path.join(cli.out, "latest")
        actor.load_state_dict(th.load(os.path.join(d, "actor.th"),
                                      map_location="cpu",
                                      weights_only=False))
        critic.load_state_dict(th.load(os.path.join(d, "critic.th"),
                                       map_location="cpu",
                                       weights_only=False))
        opt.load_state_dict(th.load(os.path.join(d, "opt.th"),
                                    map_location="cpu", weights_only=False))
        rng = th.load(os.path.join(d, "rng.th"), map_location="cpu",
                      weights_only=False)
        th.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        with open(os.path.join(d, "state.json")) as f:
            st = json.load(f)
        start_it = int(st["iter"]) + 1
        best = float(st.get("best", -1e9))
        best_it = int(st.get("best_it", 0))
        stale = int(st.get("stale", 0))
        evals_seen = int(st.get("evals_seen", 0))
        print(f"resumed at iter {start_it} (best {best:+.2f} @ {best_it}, "
              f"stale {stale}/{cli.patience})", flush=True)

    ev_seeds = cli.eval_seeds[:max(1, cli.eval_subset)]
    manifest = {"algo": "risk-as-feature residual PPO, filtered buffer",
                "git_commit": git_rev(),
                "selected_experience_caveat": "buffer filtered to picks "
                "with risk proba >= --train-min-proba; NOT standard "
                "full-buffer PPO",
                "params": vars(cli),
                "eval_seeds": list(cli.eval_seeds),
                "subset_seeds": list(ev_seeds),
                "selection_caveat": "the mid-run subset seeds are a subset of "
                "eval_seeds, so 'best subset' selection is mildly optimistic; "
                "the last-actor full eval is reported alongside it",
                "reserved_block": "7809-8200 never used here"}
    with open(os.path.join(cli.out, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    risk = ra._risk()
    cycle = [7706, 7707, 7708, 7709]
    t0 = time.time()
    prog = Progress(every=cli.log_every, t0=t0).start()
    stop = None
    it = start_it - 1
    for it in range(start_it, cli.iters + 1):
        prog.set(phase="rollout", iter=it, ep=0, ep_total=cli.eps_per_iter)
        gen = th.Generator().manual_seed(cli.seed * 100003 + it)
        rec = ra.RiskRecorder(actor, cli.bonus, gen, temp=cli.temp,
                              risk=risk, min_proba=cli.train_min_proba)
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
        rets = []
        for e in range(cli.eps_per_iter):
            seed = cycle[(it * cli.eps_per_iter + e) % len(cycle)]
            prog.set(phase="episode", ep=e + 1)
            r = mg.run_team_episode(pol, seed, cli.steps, hook=rec,
                                    recorder=rec, **mg.ENV_KW)
            rets.append(r["return"])
            prog.set(rows=len(rec.rows))
            prog.line("iter %d ep %d/%d seed %d ret %+.1f rows %d/%d"
                      % (it, e + 1, cli.eps_per_iter, seed, r["return"],
                         len(rec.rows), rec.total))
        skipped = len(rec.rows) < cli.min_rows
        if skipped:
            # Skip the update rather than learn from a degenerate batch:
            # a 1-row update normalised advantages to NaN and poisoned the
            # actor weights irrecoverably (observed 2026-10-06).
            st = {"kl": 0.0, "pol": 0.0, "vf": 0.0, "ent": 0.0}
        else:
            prog.set(phase="update")
            prog.line("iter %d update on %d rows" % (it, len(rec.rows)))
            st = mg.ppo_central_update(actor, critic, opt, rec.rows,
                                       bonus=cli.bonus)
        row = {"iter": it, "train_return": float(np.mean(rets)),
               "trained_rows": len(rec.rows), "total_rows": rec.total,
               "skipped_update": bool(skipped),
               "kl": st["kl"], "elapsed_s": round(time.time() - t0, 1)}
        prog.line("iter %d done train %+.1f kl %.4f rows %d%s"
                  % (it, row["train_return"], st["kl"], len(rec.rows),
                     " SKIPPED" if skipped else ""))

        did_eval = (it % cli.eval_every == 0 or it == cli.iters)
        if did_eval:
            prog.set(phase="eval")
            prog.line("iter %d eval on %d seeds" % (it, len(ev_seeds)))
            actor.eval()
            ev = evaluate(actor, ev_seeds, cli.steps, cli.bonus)
            actor.train()
            row["eval"] = ev
            evals_seen += 1
            improved = ev["paired_mean"] > best + cli.min_gain
            if improved:
                best = max(best, ev["paired_mean"])
                best_it = it
                stale = 0
                snap = save_snapshot(cli.out, actor, it, ev)
                best_ckpt = os.path.join(snap, "actor.th")
                prog.line("iter %d new best %+.2f -> snapshot %s"
                          % (it, best, snap))
            else:
                stale += 1
            row["best_mean"] = best
            row["best_iter"] = best_it
            row["stale"] = stale
            print(f"iter {it} train {row['train_return']:+.1f} "
                  f"eval {ev['paired_mean']:+.2f} "
                  f"CI[{ev['ci95'][0]:+.1f},{ev['ci95'][1]:+.1f}] "
                  f"W{ev['wins']}/{ev['n']} best {best:+.2f}@{best_it} "
                  f"stale {stale}/{cli.patience}", flush=True)
            prog.set(phase="train", best="%+.2f@%d" % (best, best_it))
            append_jsonl(metrics_path, row)
        else:
            append_jsonl(metrics_path, row)

        # Checkpoint AFTER the eval bookkeeping, so a resume restores the
        # real best/stale. Saving before the eval silently rolled the
        # bookkeeping back one eval on every resume.
        save_ckpt(cli.out, actor, critic, opt,
                  {"torch": th.get_rng_state(),
                   "numpy": np.random.get_state(),
                   "python": random.getstate()}, it,
                  track={"best": best, "best_it": best_it, "stale": stale,
                         "evals_seen": evals_seen})

        if did_eval and stale >= cli.patience:
            stop = "PLATEAU"
            print(f"STOP: no improvement > {cli.min_gain} for "
                  f"{cli.patience} evals", flush=True)
            break
        if (time.time() - t0) / 3600.0 >= cli.max_hours:
            stop = "TIME"
            print("STOP: max-hours reached", flush=True)
            break

    # Authoritative gate decision: FULL seed list at FULL horizon. The
    # mid-run evals are a cheap subset for plateau tracking only - a small
    # subset must never be able to declare a pass (our own decision rules).
    #
    # Both candidates are scored. `final` is the last actor: no selection
    # happened, so it is the honest number. `best_full` is the checkpoint
    # that won the subset; it is what we would actually ship, but it was
    # chosen using 4 of these same 8 seeds, so it is mildly optimistic.
    # Reporting only one of them is how a noisy curve gets over-claimed.
    prog.set(phase="final-eval")
    prog.stop()
    prog.line("final full eval on %d seeds (last actor)" % len(cli.eval_seeds))
    actor.eval()
    final = evaluate(actor, cli.eval_seeds, cli.steps, cli.bonus)
    th.save(actor.state_dict(), os.path.join(cli.out, "actor_last.th"))
    print("FINAL FULL EVAL (last actor) " + json.dumps(final), flush=True)

    best_full = None
    if best_ckpt and os.path.exists(best_ckpt):
        prog.line("final full eval on %d seeds (best subset ckpt @%d)"
                  % (len(cli.eval_seeds), best_it))
        b = load_actor(cli.out, os.path.dirname(best_ckpt), cli.hidden)
        b.eval()
        best_full = evaluate(b, cli.eval_seeds, cli.steps, cli.bonus)
        th.save(b.state_dict(), os.path.join(cli.out, "actor_best.th"))
        print("FINAL FULL EVAL (best subset ckpt) " + json.dumps(best_full),
              flush=True)
    else:
        print("no best checkpoint recorded (no eval improved on the first)",
              flush=True)

    def _gate(ev):
        if not ev:
            return False
        return bool(ev["paired_mean"] >= cli.gate
                    and ev["ci95"][0] > 0
                    and ev["n"] >= cli.gate_min_seeds
                    and cli.steps >= cli.gate_min_steps)

    # Ship decision: the best checkpoint, because that is the artifact we
    # would deploy. The last-actor number is kept next to it so the
    # selection optimism stays visible instead of hidden.
    primary = best_full if best_full is not None else final
    gate_passed = _gate(primary)
    append_jsonl(metrics_path, {"iter": it, "final_full_eval": final,
                                "best_subset_full_eval": best_full})
    summary = {"final_full_eval_last_actor": final,
               "full_eval_best_subset_ckpt": best_full,
               "primary": "best_subset_ckpt" if best_full else "last_actor",
               "gate_passed": gate_passed,
               "gate_passed_last_actor_only": _gate(final),
               "stop_reason": stop or "ITERS",
               "iters_done": it, "best_subset_mean": best,
               "best_subset_iter": best_it, "evals": evals_seen,
               "selection_caveat": "subset seeds are inside eval_seeds, so "
               "the best-subset number is mildly optimistic; compare against "
               "the last-actor number",
               "gate": cli.gate, "eval_seeds": list(cli.eval_seeds),
               "subset_seeds": list(ev_seeds),
               "git_commit": git_rev()}
    with open(os.path.join(cli.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("GATE (" + summary["primary"] + ") "
          + ("PASSED" if gate_passed else "NOT PASSED"), flush=True)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()