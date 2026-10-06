"""Authoritative used-seed audit.

A plain number grep is not enough and has already produced two wrong
answers in this project:

- `range(8201, 8217)` consumes 8201-8216 and EXCLUDES 8217, but a grep for
  "8217" finds nothing, so 8217 looks free when it is not part of the
  range (and 8217 itself is used elsewhere).
- git SHAs contain digit runs that match seed patterns: `dab5d077317f31`
  made seed 7731 look consumed when it is not.

So this walks actual seed-bearing structures:

1. every `range(a, b)` / `range(a, b, c)` in tracked .py files, EXPANDED;
2. every list/tuple/set literal of ints in tracked .py files;
3. every integer key or value in tracked .json / .jsonl manifests;
4. argparse `nargs="*"` seed defaults;

then reports which blocks are free. Numbers are only accepted from
structured positions, so a SHA inside a string cannot create a false hit.

Usage:
    .venv-train/bin/python -m blue.training.seed_audit --check 8221 8252
    .venv-train/bin/python -m blue.training.seed_audit --free-blocks 32
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# Seed-like: 7000-9999. Anything outside is not an episode seed.
SEED_LO, SEED_HI = 7000, 9999

# Blocks the team has declared off-limits. Reported separately from
# "consumed" so a disputed claim is visible instead of silently applied.
DECLARED_RESERVED = {
    "7809-8200 final evaluation block": (7809, 8200),
    "8501+ reserved": (8501, 9999),
    "8301-8400 finite Red": (8301, 8400),
    "7801-7808 spent for decisions": (7801, 7808),
}

SKIP_DIRS = {".git", "__pycache__", ".venv", ".venv-train", "third_party",
             ".cache", "node_modules"}

# This file quotes seed numbers in its docstring and its reserved-block
# table, so scanning it would mark those seeds as consumed by itself.
SELF = os.path.basename(__file__)


def _seed_ints(node):
    """Ints that appear in a seed-bearing position (not inside strings)."""
    out = []
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return out
        if isinstance(node.value, int) and SEED_LO <= node.value <= SEED_HI:
            out.append(node.value)
    return out


def seeds_from_py(path):
    """Expanded seeds from a Python file: ranges are expanded, and bare
    ints are only accepted inside list/tuple/set/range literals."""
    found = set()
    try:
        with open(path, "r", errors="replace") as f:
            tree = ast.parse(f.read(), filename=path)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return found
    for node in ast.walk(tree):
        # range(a, b, c) -> the actual seeds it yields
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "range":
            vals = []
            for a in node.args:
                vals.append(a.value if isinstance(a, ast.Constant)
                            and isinstance(a.value, int) else None)
            if len(vals) >= 2 and vals[0] is not None and vals[1] is not None:
                step = vals[2] if len(vals) > 2 and vals[2] else 1
                if step:
                    try:
                        found.update(range(vals[0], vals[1], step))
                    except ValueError:
                        pass
        # explicit seed lists
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                found.update(_seed_ints(elt))
        # set/dict literals used as seed sets
        if isinstance(node, ast.Dict):
            for k in node.keys:
                found.update(_seed_ints(k))
    return {s for s in found if SEED_LO <= s <= SEED_HI}


def seeds_from_json(path):
    """Every int key or int value in a manifest."""
    found = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, int) and SEED_LO <= k <= SEED_HI:
                    found.add(k)
                if isinstance(k, str):
                    # JSON object keys are often stringified seeds
                    if k.isdigit() and SEED_LO <= int(k) <= SEED_HI:
                        found.add(int(k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    try:
        if path.endswith(".jsonl"):
            with open(path, errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        walk(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        else:
            with open(path, errors="replace") as f:
                walk(json.load(f))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return set()
    return {s for s in found if SEED_LO <= s <= SEED_HI}


def tracked_files(exts):
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    files = []
    for rel in out.stdout.splitlines():
        if not rel:
            continue
        if any(rel.startswith(d + "/") for d in SKIP_DIRS):
            continue
        if os.path.basename(rel) == SELF:
            continue
        if os.path.splitext(rel)[1] in exts:
            files.append(os.path.join(REPO, rel))
    return files


def untracked_seed_files(exts):
    """Files present on disk but not tracked.

    Deliberately INCLUDES gitignored files: `blue/results/` holds the real
    consumption records (eval jsons, pool cells, run stats). Excluding them
    -- the default for git -- hides seeds that were genuinely spent.
    """
    out = subprocess.run(
        ["git", "ls-files", "--others"], cwd=REPO,
        capture_output=True, text=True)
    files = []
    for rel in out.stdout.splitlines():
        if not rel or any(d in rel.split("/")[0] for d in SKIP_DIRS):
            continue
        if os.path.basename(rel) == SELF or rel == ".gitignore":
            continue
        if os.path.splitext(rel)[1] in exts:
            files.append(os.path.join(REPO, rel))
    return files


def build_used():
    used = {}
    py = tracked_files({".py"}) + untracked_seed_files({".py"})
    js = tracked_files({".json", ".jsonl"}) + \
        untracked_seed_files({".json", ".jsonl"})
    for p in py:
        for s in seeds_from_py(p):
            used.setdefault(s, set()).add(os.path.relpath(p, REPO))
    for p in js:
        for s in seeds_from_json(p):
            used.setdefault(s, set()).add(os.path.relpath(p, REPO))
    return used


def free_blocks(used, lo=7600, hi=9999, need=32):
    blocks = []
    run = []
    for s in range(lo, hi + 1):
        if s in used:
            if len(run) >= need:
                blocks.append((run[0], run[-1], len(run)))
            run = []
        else:
            run.append(s)
    if len(run) >= need:
        blocks.append((run[0], run[-1], len(run)))
    return blocks


def reserved_conflicts(lo, hi):
    out = []
    for label, (a, b) in DECLARED_RESERVED.items():
        if lo <= b and hi >= a:
            out.append((label, max(lo, a), min(hi, b)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", type=int, nargs=2, metavar=("LO", "HI"),
                    help="report whether LO..HI is free, and who used it")
    ap.add_argument("--free-blocks", type=int, default=32,
                    help="list free runs of at least N seeds")
    ap.add_argument("--dump-used", help="write the used-seed map here")
    args = ap.parse_args()

    used = build_used()
    if args.dump_used:
        with open(args.dump_used, "w") as f:
            for s in sorted(used):
                f.write("%d\t%s\n" % (s, ";".join(sorted(used[s]))))
        print("wrote %s (%d seeds)" % (args.dump_used, len(used)))

    print("used seeds discovered: %d" % len(used))
    if args.check:
        lo, hi = args.check
        clash = [s for s in range(lo, hi + 1) if s in used]
        print("\nrange %d-%d (%d seeds):" % (lo, hi, hi - lo + 1))
        if not clash:
            print("  FREE - no recorded consumption")
        else:
            print("  CONTAMINATED, %d seed(s) already used:" % len(clash))
            for s in clash:
                print("    %d  <- %s" % (s, ";".join(sorted(used[s]))))
        conf = reserved_conflicts(lo, hi)
        if conf:
            print("\n  declared-reserved overlap (needs a human call):")
            for label, a, b in conf:
                print("    %s: overlaps %d-%d" % (label, a, b))
        return 0 if not clash else 1

    print("\nfree runs of >= %d seeds:" % args.free_blocks)
    for a, b, n in free_blocks(used, need=args.free_blocks):
        conf = reserved_conflicts(a, b)
        flag = "  <-- overlaps: " + ", ".join(c[0] for c in conf) \
            if conf else ""
        print("  %d-%d  (%d seeds)%s" % (a, b, n, flag))
    return 0


if __name__ == "__main__":
    sys.exit(main())