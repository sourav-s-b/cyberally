"""Compare a parallel-eval manifest: table, paired diffs, trace parity.

Usage: ../.venv/bin/python blue_compare.py
       docs/proposals/manifests/<run-id>.json [--baseline round_robin]
       [--markdown]
Std is sample std (ddof=1), matching prior proposal records. Trace columns
compare sha256 vs the baseline: MATCH only makes sense for parity-type runs
(e.g. hybrid_none vs round_robin), not for genuinely different policies.
"""

import argparse
import json
import statistics
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def summarize(manifest):
    rows = {}
    for policy, per_seed in manifest["cells"].items():
        rets = [per_seed[s]["return"] for s in sorted(per_seed)]
        rows[policy] = {
            "n": len(rets), "rets": rets,
            "mean": statistics.mean(rets),
            "std": statistics.stdev(rets) if len(rets) > 1 else 0.0}
    return rows


def main():
    ap = argparse.ArgumentParser(description="Compare an eval manifest")
    ap.add_argument("manifest")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--markdown", action="store_true",
                    help="no-op (output is always markdown)")
    cli = ap.parse_args()
    m = load(cli.manifest)
    rows = summarize(m)
    base = cli.baseline
    if base is not None and base not in rows:
        raise SystemExit(f"baseline {base!r} not in manifest "
                         f"(has {sorted(rows)})")
    if base is not None:
        b = m["cells"][base]
        want = sorted(b)
        for policy in sorted(rows):
            if policy == base:
                continue
            have = sorted(m["cells"][policy])
            if have != want:
                raise SystemExit(
                    f"seed sets differ: baseline {base} has {want}, "
                    f"{policy} has {have}")
    lines = []
    lines.append(f"# {m['run_id']} ({m['git_commit'][:12]}, "
                 f"steps={m['steps']}, seeds={m['seeds']})")
    lines.append("")
    lines.append("| policy | n | mean | std | per-seed returns |")
    lines.append("|---|---|---|---|---|")
    for policy in sorted(rows):
        r = rows[policy]
        rets = " ".join(f"{v:.0f}" for v in r["rets"])
        lines.append(f"| {policy} | {r['n']} | {r['mean']:.1f} | {r['std']:.1f} "
                     f"| {rets} |")
    if base is not None:
        lines.append("")
        lines.append(f"Paired diff vs {base} (policy minus baseline, per seed):")
        b = m["cells"][base]
        for policy in sorted(rows):
            if policy == base:
                continue
            diffs = [m["cells"][policy][s]["return"] - b[s]["return"]
                     for s in sorted(b)]
            md = statistics.mean(diffs)
            lines.append(f"- {policy}: mean {md:+.1f} "
                         f"({' '.join(f'{d:+.0f}' for d in diffs)})")
            par = all(m["cells"][policy][s]["trace_sha256"]
                      == b[s]["trace_sha256"] for s in sorted(b))
            lines.append(f"  trace parity: {'MATCH' if par else 'differ'}")
    text = "\n".join(lines)
    print(text)


if __name__ == "__main__":
    main()
