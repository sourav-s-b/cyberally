"""Exercise the real pinned EPyMARL runner/learner when locally available."""
from pathlib import Path
import json

import pytest

from blue_mappo import main


def test_mappo_two_fresh_updates_and_checkpoint_reload(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("smaclite")
    epymarl = Path(__file__).resolve().parents[2] / ".tools/epymarl"
    if not (epymarl / "src/runners/episode_runner.py").is_file():
        pytest.skip("Pinned EPyMARL checkout not installed")
    manifest = main(["--epymarl-root", str(epymarl), "--output", str(tmp_path / "run"),
                     "--updates", "2", "--episodes-per-update", "1", "--steps", "12",
                     "--seed", "42", "--eval-seeds", "90001"])
    assert manifest["joint_steps"] == 22
    assert manifest["train_seeds"] == [42, 43]
    assert manifest["evaluation"][0]["joint_steps"] == 11
    assert set(manifest["checkpoint_sha256"]) == {
        "agent.th", "critic.th", "agent_opt.th", "critic_opt.th"}
    records = [json.loads(line) for line in
               (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert [r["joint_steps"] for r in records] == [11, 22]
    assert all("pg_loss" in r["metrics"] and "critic_loss" in r["metrics"]
               for r in records)
    reevaluated = main(["--epymarl-root", str(epymarl), "--output", str(tmp_path / "eval"),
                        "--evaluate-manifest", str(tmp_path / "run/manifest.json"),
                        "--eval-seeds", "90001"])
    assert reevaluated["evaluation"][0]["native_return"] == manifest["evaluation"][0]["native_return"]
    from blue_compare import main as compare
    comparison = compare(["--manifest", str(tmp_path / "run/manifest.json"),
                          "--output", str(tmp_path / "comparison"),
                          "--eval-seeds", "90001"])
    assert set(comparison["comparisons"]) == {"sleep", "masked-random", "round-robin"}
    assert all(episode["joint_steps"] == 11
               for episodes in comparison["episodes"].values() for episode in episodes)
    assert comparison["episodes"]["mappo"][0]["native_return"] == manifest["evaluation"][0]["native_return"]
    actor_path = tmp_path / "run/checkpoint/agent.th"
    actor_path.write_bytes(actor_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Checkpoint hash mismatch"):
        main(["--epymarl-root", str(epymarl), "--output", str(tmp_path / "bad-eval"),
              "--evaluate-manifest", str(tmp_path / "run/manifest.json"),
              "--eval-seeds", "90001"])


def test_training_evaluation_seed_overlap_fails_before_training(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        main(["--output", str(tmp_path / "must-not-exist"), "--seed", "42",
              "--eval-seeds", "43"])
    assert not (tmp_path / "must-not-exist").exists()
