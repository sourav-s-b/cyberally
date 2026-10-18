import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_xargs_pool_bounds_real_child_processes(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text('''
import fcntl, json, os, pathlib, sys, time
d=pathlib.Path(sys.argv[1]); lock=open(d/"lock","w")
fcntl.flock(lock, fcntl.LOCK_EX)
a=int((d/"active").read_text()) if (d/"active").exists() else 0
a+=1; (d/"active").write_text(str(a))
m=int((d/"max").read_text()) if (d/"max").exists() else 0
(d/"max").write_text(str(max(a,m)))
with (d/"pids").open("a") as f: f.write(str(os.getpid())+"\\n")
fcntl.flock(lock, fcntl.LOCK_UN)
time.sleep(.2)
fcntl.flock(lock, fcntl.LOCK_EX)
(d/"active").write_text(str(int((d/"active").read_text())-1))
fcntl.flock(lock, fcntl.LOCK_UN)
''')
    cmd = ["bash", str(ROOT / "scripts/blue_xargs_pool.sh"), "2", "2",
           sys.executable, str(worker)]
    tasks = "".join(f"{tmp_path}\t{i}\n" for i in range(6))
    subprocess.run(cmd, input=tasks, text=True, check=True)
    assert int((tmp_path / "max").read_text()) == 2
    pids = (tmp_path / "pids").read_text().splitlines()
    assert len(pids) == 6 and len(set(pids)) == 6


def test_plan_and_pilot_are_not_detached():
    script = (ROOT / "scripts/blue_finite_red_check.sh").read_text()
    assert "plan) echo" in script
    assert "pilot) need_runtime" in script
    assert "exec setsid" not in script
    assert "nohup" not in script
