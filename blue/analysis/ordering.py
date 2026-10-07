"""Phase 2: does host ORDERING have headroom? (No training.)

Compares sweep orderings inside the same fixed rules (HybridBluePolicy
rules 1-3: remediate CONFIRMED, verify oldest-first, sweep the rest):

  - parity   : cursor round-robin (existing hybrid_none behavior)
  - lancer   : hybrid_lancer_v2 values
  - oldest   : never-analysed first, then stalest analysis
  - obsbump  : current suspicious-view signals only, no carried value
  - oracle   : DIAGNOSTIC ONLY. Reads privileged _compromised_set() to
               order compromised hosts by onset. Upper-bounds what any
               observation-based learner could gain. Never deployed,
               never a claim.

Metrics per episode (native return PLUS): investigation coverage
(distinct analysed hosts / zone hosts), max investigation age at end,
detection delay (compromised-onset tick -> first Analyse tick on that
host; privileged ground truth, eval-only), undetected count,
remediation count. Null band: |paired Δ| inside the 95% paired CI on
the dev block (~±28 at SD 40, n=8; recomputed from data).

Usage (repo root):
  .venv-train/bin/python -m blue.analysis.ordering \
      --seeds 7629 7630 7640 7701 7702 7703 7704 7705 --steps 400 \
      --out /tmp/ordering.jsonl
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blue.core.baselines import decode_index
from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
from blue.policies.hybrid import (HybridBluePolicy, host_risk_features,
                                  make_priority)


class OldestFirst:
    """Never-analysed first, then stalest last-analysis. Stateless."""

    def __call__(self, env, agent, host):
        last = env.trackers[agent].last_analysis.get(host)
        if last is None:
            return 0.0
        return -float(last)


class ObsBump:
    """Current suspicious-view signals only (no carried lancer value)."""

    def __call__(self, env, agent, host):
        vec = host_risk_features(env, agent, host)
        unknown, density, n_ext = vec[6], vec[7], vec[9]
        return float(unknown > 0) + float(density > 0.9) + float(n_ext > 0)


class OracleOnset:
    """DIAGNOSTIC upper bound: compromised hosts first by onset tick.

    Reads privileged compromise state at select time. For measurement
    only; any policy shipping this would be cheating.

    ``_compromised_set()`` is cached per tick. The pre-repair version
    called it once per HOST per decision (435 calls/tick at 87 hosts x 5
    agents), which dominated the oracle's runtime (82 s vs 22 s per
    episode). Caching only the set -- not the per-agent onset bookkeeping --
    leaves the scores bit-identical: every call still folds the whole set
    into ``self._onset`` for the calling agent, exactly as before.
    ``privileged_calls`` is reported so the saving is visible rather than
    assumed.
    """

    def __init__(self):
        self._onset = {}
        self._tick = None
        self._comp = frozenset()
        self.privileged_calls = 0

    def reset(self):
        self._onset = {}
        self._tick = None
        self._comp = frozenset()

    def _compromised(self, env):
        tick = env._tick
        if self._tick != tick:
            self._tick = tick
            self._comp = frozenset(env._compromised_set())
            self.privileged_calls += 1
        return self._comp

    def __call__(self, env, agent, host):
        comp = self._compromised(env)
        tick = env._tick
        for h in comp:
            self._onset.setdefault((agent, h), tick)
        if host in comp:
            return -float(self._onset[(agent, host)])
        return -1e9


def _decode(env, agent, action_idx):
    name, host = decode_index(env, agent, int(action_idx))
    return name, host


# Metric columns produced by this module are KNOWN INVALID (see the module
# docstring and docs/status/blue-session.md 2026-10-04). ``return`` and
# ``steps`` are still exact, which is why this entry point remains for
# reproducing historical returns. The invalid columns are withheld unless a
# caller explicitly opts in, so nobody can quietly publish them again.
LEGACY_INVALID_METRICS = ("coverage", "max_age", "det_delay_median",
                          "n_undetected")
LEGACY_INVALID_REASON = (
    "invalid: requested-vs-completed analyses, first-analysis-only, failed "
    "requests counted, non-defendable hosts included, final-tick age reported "
    "as max age. Use blue.analysis.compare + blue.analysis.metrics. "
    "Manifest: docs/proposals/manifests/guard-maxage-20261004.json")


def run_ordering_episode(policy, seed, steps=400, emit_invalid_metrics=False,
                         **env_kwargs):
    """Mirror run_episode stepping + record privileged compromise timeline
    (eval-only) and per-host analyse ticks for delay/coverage metrics.
    Takes a CONSTRUCTED policy (fresh scorer per episode is the caller's
    job; policy.reset() is called here).

    ``emit_invalid_metrics=False`` (default) returns ``None`` for the four
    legacy metric fields and adds ``legacy_invalid_metrics`` to the result.
    ``return``/``steps``/``n_compromised``/``n_remediations`` are unaffected.
    """
    env = CC4MARLEnv(seed=seed, steps=steps, **env_kwargs)
    env.reset(seed=seed)
    policy.reset()
    cumulative = 0.0
    onset = {}
    analysed_at = {}
    analysed_hosts = set()
    remediations = 0
    zone_hosts = {a: list(env.hostnames[a]) for a in BLUE_AGENTS}
    for tick in range(1, steps + 1):
        for h in env._compromised_set():
            onset.setdefault(h, tick)
        actions = {a: int(policy.select(env, a)) for a in BLUE_AGENTS}
        _, rewards, terminated, truncated, _ = env.step(actions)
        cumulative += float(rewards[0])
        for agent in BLUE_AGENTS:
            name, host = _decode(env, agent, actions[agent])
            if name == "Analyse" and host is not None:
                analysed_hosts.add((agent, host))
                analysed_at.setdefault((agent, host), tick)
            if name in ("Remove", "Restore"):
                remediations += 1
        if terminated or truncated:
            break
    delays = []
    undetected = 0
    for h, t0 in onset.items():
        det = [t for (a, hh), t in analysed_at.items()
               if hh == h and t >= t0]
        if det:
            delays.append(min(det) - t0)
        else:
            undetected += 1
    delays.sort()
    total_hosts = sum(len(v) for v in zone_hosts.values())
    ages = []
    for agent in BLUE_AGENTS:
        for host in zone_hosts[agent]:
            last = analysed_at.get((agent, host))
            ages.append(tick - last if last is not None else tick)
    out = {"seed": seed, "steps": tick, "return": cumulative,
           "n_compromised": len(onset), "n_remediations": remediations}
    if emit_invalid_metrics:
        out.update({
            "coverage": len(analysed_hosts) / max(total_hosts, 1),
            "max_age": max(ages) if ages else tick,
            "det_delay_median": (delays[len(delays) // 2] if delays
                                 else None),
            "n_undetected": undetected,
        })
    else:
        out.update({k: None for k in LEGACY_INVALID_METRICS})
        out["legacy_invalid_metrics"] = LEGACY_INVALID_REASON
    return out


VARIANTS = {
    "parity": None,
    "lancer": ("lancer", {"fruitless_decay": 0.5}),  # == hybrid_lancer_v2
    "oldest": OldestFirst(),
    "obsbump": ObsBump(),
    "oracle": OracleOnset(),
}


def main():
    ap = argparse.ArgumentParser(description="Phase 2 ordering comparison")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[7629, 7630, 7640, 7701, 7702, 7703, 7704, 7705])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--guard", action="store_true",
                    help="use OrderedPolicy with coverage guard ON (learner "
                         "regime) instead of HybridBluePolicy; variants map "
                         "to guard-aware equivalents")
    ap.add_argument("--emit-invalid-metrics", action="store_true",
                    help="also emit the KNOWN INVALID legacy metric columns "
                         "(coverage, max_age, det_delay_median, n_undetected). "
                         "Withheld by default; use blue.analysis.compare "
                         "instead.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run a subset of variants (default: all)")
    cli = ap.parse_args()
    if not cli.emit_invalid_metrics:
        print("note: legacy metric columns withheld as KNOWN INVALID; pass "
              "--emit-invalid-metrics to reproduce them, or use "
              "blue.analysis.compare for corrected metrics")
    env_kwargs = {"temporal_features": ("ages", "belief"),
                  "include_root_session": True, "red_agent": "discovery"}
    for name, spec in VARIANTS.items():
        if cli.only is not None and name not in cli.only:
            continue
        if spec is None:
            fn = None
        elif isinstance(spec, tuple):
            fn = make_priority(spec[0], **spec[1])
        else:
            fn = spec
            try:
                fn.reset()
            except AttributeError:
                pass
        for seed in cli.seeds:
            if cli.guard:
                from blue.policies.ordered import (
                    CursorSweep, LancerValues, OrderedPolicy)
                if name == "parity":
                    policy = OrderedPolicy(order_fn=CursorSweep(),
                                           guard=True)
                elif name == "lancer":
                    policy = OrderedPolicy(
                        scorer=LancerValues(fruitless_decay=0.5),
                        guard=True)
                elif name == "oracle":
                    policy = OrderedPolicy(scorer=OracleOnset(),
                                           guard=True)
                elif name == "oldest":
                    policy = OrderedPolicy(scorer=OldestFirst(),
                                           guard=True)
                elif name == "obsbump":
                    policy = OrderedPolicy(scorer=ObsBump(), guard=True)
                r = run_ordering_episode(
                    policy, seed, cli.steps,
                    emit_invalid_metrics=cli.emit_invalid_metrics,
                    **env_kwargs)
            else:
                policy = HybridBluePolicy(priority_fn=fn)
                r = run_ordering_episode(
                    policy, seed, cli.steps,
                    emit_invalid_metrics=cli.emit_invalid_metrics,
                    **env_kwargs)
            r["variant"] = name
            r["guard"] = bool(cli.guard)
            line = (f"{name:8s} seed={seed} return={r['return']:+7.1f} "
                    f"comp={r['n_compromised']:3d} rem={r['n_remediations']:3d}")
            if cli.emit_invalid_metrics:
                line += (f" cov={r['coverage']:.2f} maxage={r['max_age']:3d} "
                         f"detmed={r['det_delay_median']} "
                         f"undet={r['n_undetected']}  [INVALID COLUMNS]")
            print(line, flush=True)
            if cli.out:
                with open(cli.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
    if cli.out:
        print("wrote", cli.out)


if __name__ == "__main__":
    main()
