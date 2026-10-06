"""State-handling tests for the unattended trainer.

These cover the bookkeeping that silently broke long runs: a resumed run
forgetting its own best score (so it could never plateau), the plateau
counters being written before the eval that changes them, and a checkpoint
swap that could destroy the only good copy.

No simulator episodes here: these are file/state contracts, so they stay
fast enough to run routinely.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from blue.training import remote_train as rt  # noqa: E402


def _ckpt(out, it, track=None):
    actor = torch.nn.Linear(2, 2)
    critic = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(list(actor.parameters()), lr=1e-3)
    rt.save_ckpt(out, actor, critic, opt, {"torch": torch.get_rng_state(),
                                           "numpy": None,
                                           "python": None},
                 it, track=track)
    return os.path.join(out, "latest")


def test_checkpoint_persists_plateau_bookkeeping(tmp_path):
    """best/stale must survive a kill, or resume re-declares improvement."""
    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    d = _ckpt(out, 7, track={"best": -8.75, "best_it": 5, "stale": 3,
                             "evals_seen": 4})
    with open(os.path.join(d, "state.json")) as f:
        state = json.load(f)
    assert state["iter"] == 7
    assert state["best"] == pytest.approx(-8.75)
    assert state["best_it"] == 5
    assert state["stale"] == 3
    assert state["evals_seen"] == 4
    with open(os.path.join(out, "LATEST")) as f:
        assert f.read().strip() == "7"


def test_checkpoint_swap_keeps_previous_copy_until_new_one_is_in_place(tmp_path):
    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    _ckpt(out, 1)
    _ckpt(out, 2)
    latest = os.path.join(out, "latest")
    with open(os.path.join(latest, "state.json")) as f:
        assert json.load(f)["iter"] == 2
    # the atomic swap must not leave scratch dirs behind
    assert not os.path.exists(os.path.join(out, ".ckpt.tmp"))
    assert not os.path.exists(os.path.join(out, ".ckpt.trash"))


@pytest.fixture
def fake_actor(monkeypatch):
    """A stand-in actor.

    The real `build_actor` seeds itself from the Phase-3 scorer checkpoint,
    which is a gitignored artifact, so tests must not depend on it being
    present or on its hidden width.
    """
    monkeypatch.setattr(rt.ra, "build_actor",
                        lambda hidden=64: torch.nn.Linear(3, 2))
    return torch.nn.Linear(3, 2)


def test_snapshot_round_trips_through_load_actor(tmp_path, fake_actor):
    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    actor = fake_actor
    snap = rt.save_snapshot(out, actor, 12, {"paired_mean": 1.5})
    assert snap.endswith(os.path.join("snapshots", "iter_0012"))
    again = rt.load_actor(out, snap, hidden=8)
    for a, b in zip(actor.parameters(), again.parameters()):
        assert torch.equal(a, b)
    with open(os.path.join(snap, "eval.json")) as f:
        assert json.load(f)["eval"]["paired_mean"] == 1.5


def test_snapshot_dir_is_created_when_missing(tmp_path, fake_actor):
    out = str(tmp_path / "run")
    snap = rt.save_snapshot(out, fake_actor, 1, {})
    assert os.path.isdir(snap)


def test_progress_reports_state_and_heartbeat(capsys):
    import time as _time
    p = rt.Progress(every=0, t0=_time.time())
    p.set(phase="episode", iter=3, ep=2, ep_total=4, rows=17, best="+1.00@3")
    p.line("hello")
    out = capsys.readouterr().out
    assert "hello" in out
    assert "iter=3" in out
    assert "ep=2" in out and "ep_total=4" in out
    assert "rows=17" in out
    assert "+1.00@3" in out


def test_progress_heartbeat_thread_starts_and_stops():
    p = rt.Progress(every=0.01, t0=0.0).start()
    try:
        assert p._thread is not None
        assert p._thread.daemon is True
    finally:
        p.stop()
        p._thread.join(timeout=2.0)
    assert not p._thread.is_alive()


def test_progress_with_zero_interval_does_not_spawn_thread():
    """--log-every 0 must disable the thread, not spin one."""
    p = rt.Progress(every=0).start()
    assert p._thread is None
    p.stop()