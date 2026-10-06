"""Long-horizon unattended Blue training on Kaggle (script kernel).

Sources:
  code     -> public GitHub repo at a PINNED commit (single source of truth)
  artifacts-> attached Kaggle dataset (two gitignored model files:
              scorer_mlp.npz, risk_model_v2.pkl - deliberately not in git)

The simulator stepping is CPU-bound and single-threaded, so a plain CPU
kernel is the right target: we want wall-clock runtime, not GPU. The run
stops on evidence (plateau or dev gate), checkpoints every iteration, and
streams metrics to an append-only JSONL so a session kill loses nothing.
"""
import glob
import os
import subprocess
import sys

REPO = "https://github.com/sourav-s-b/cyberally.git"
BRANCH = "blue/metric-repair-maxage"
COMMIT = "0695518cf31590323b3bb1ada30b0fa5da3a96fd"
OUT = "remote_out"


def sh(cmd, **kw):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, **kw)


print("=== fetch code ===", flush=True)
sh(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO, "repo"])
# Pin exactly: --depth 1 clone is fine, but record the sha for provenance.
try:
    sha = subprocess.check_output(
        ["git", "-C", "repo", "rev-parse", "HEAD"]).decode().strip()
    print("cloned commit:", sha, flush=True)
    if COMMIT != "REPLACE_COMMIT" and not sha.startswith(COMMIT[:8]):
        sh(["git", "-C", "repo", "fetch", "--depth", "1", "origin", COMMIT])
        sh(["git", "-C", "repo", "checkout", COMMIT])
        sha = subprocess.check_output(
            ["git", "-C", "repo", "rev-parse", "HEAD"]).decode().strip()
        print("pinned commit:", sha, flush=True)
except Exception as e:
    print("commit pin failed:", e, flush=True)

ROOT = os.path.abspath("repo")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

print("=== artifacts ===", flush=True)
os.makedirs(os.path.join(ROOT, "blue", "results"), exist_ok=True)
for pat in ("**/scorer_mlp.npz", "**/risk_model_v2.pkl"):
    hits = glob.glob(os.path.join("/kaggle/input", pat), recursive=True)
    print(pat, "->", hits[:2], flush=True)
    for h in hits[:1]:
        dst = os.path.join(ROOT, "blue", "results", os.path.basename(h))
        if not os.path.exists(dst):
            subprocess.check_call(["cp", h, dst])
            print("installed", dst, flush=True)

print("=== deps ===", flush=True)
# numpy<2 matches the locally validated stack (gym 0.26 warns on numpy 2).
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "numpy<2", "gym==0.26.2", "pyyaml"], check=False)
import numpy
import torch
print("numpy", numpy.__version__, "torch", torch.__version__, flush=True)
import blue.training.remote_train as rt
print("trainer module:", rt.__file__, flush=True)

CMD = [
    sys.executable, "-m", "blue.training.remote_train",
    "--out", OUT,
    "--iters", "120",
    "--eps-per-iter", "4",
    "--eval-every", "5",
    "--eval-subset", "4",
    "--eval-seeds", "7629", "7630", "7640", "7701", "7702", "7703",
    "7704", "7705",
    "--steps", "400",
    "--train-min-proba", "0.5",
    "--temp", "0.5",
    "--gate", "5",
    "--patience", "4",
    # Must exceed the eval noise (observed swings of ~40 on the 4-seed
    # subset) or the plateau stop fires on noise and the run never plateaus.
    "--min-gain", "5.0",
    "--max-hours", "10",
    "--log-every", "60",
    "--resume",
]
env = dict(os.environ)
env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
env["PYTHONUNBUFFERED"] = "1"
print("=== train ===", flush=True)
print(" ".join(CMD), flush=True)
rc = subprocess.call(CMD, env=env)
print("=== trainer exit", rc, "===", flush=True)
for name in ("summary.json", "metrics.jsonl"):
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        print(f"--- {name} ---", flush=True)
        with open(p) as f:
            print(f.read()[-4000:], flush=True)
print("=== files ===", flush=True)
for root, _dirs, files in os.walk(OUT):
    for fn in files:
        fp = os.path.join(root, fn)
        print(fp, os.path.getsize(fp), flush=True)
