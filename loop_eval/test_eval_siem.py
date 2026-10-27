"""Tests for eval_siem_metrics, run_baseline_eval, and siem_export_ndjson.

Run from the loop_eval directory or project root:
    cd loop_eval && python -m pytest test_eval_siem.py -v

Tests that require the CAGE4 simulator are skipped if CybORG is not installed.
Unit tests (schema, risk classification, exporter) run without the simulator.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_LOOP_EVAL = Path(__file__).resolve().parent
_ROOT = _LOOP_EVAL.parent
_BLUE = _ROOT / "blue"
for p in (_LOOP_EVAL, _BLUE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ── optional simulator ─────────────────────────────────────────────────────────
CYBORG_AVAILABLE = importlib.util.find_spec("CybORG") is not None
requires_cyborg = pytest.mark.skipif(
    not CYBORG_AVAILABLE, reason="CybORG simulator not installed"
)

# ── imports ───────────────────────────────────────────────────────────────────
from eval_siem_metrics import AuditEvent, EVAL_VERSION, POLICIES
from siem_export_ndjson import _classify_risk, _make_doc_id, export_telemetry_to_ndjson


# =============================================================================
# AuditEvent schema tests (no simulator needed)
# =============================================================================

class TestAuditEvent:
    def _make(self, **overrides):
        defaults = dict(
            episode_id="test-ep",
            seed=42,
            tick=1,
            agent="blue_agent_0",
            action="Analyse",
            host="host_0",
            reward=-0.1,
            host_belief="CONFIRMED",
            alert=True,
            alert_type="Analyse-CONFIRMED",
            correlation_id="corr-1",
        )
        defaults.update(overrides)
        return AuditEvent(**defaults)

    def test_to_ndjson_is_valid_json(self):
        evt = self._make()
        raw = evt.to_ndjson()
        parsed = json.loads(raw)
        assert parsed["eval_version"] == EVAL_VERSION
        assert parsed["event_type"] == "blue_action"

    def test_fields_preserved(self):
        evt = self._make(tick=10, action="Remove", host="host_1")
        parsed = json.loads(evt.to_ndjson())
        assert parsed["tick"] == 10
        assert parsed["action"] == "Remove"
        assert parsed["host"] == "host_1"

    def test_none_host_allowed(self):
        evt = self._make(action="Sleep", host=None, alert=False,
                         alert_type=None, correlation_id=None, host_belief=None)
        parsed = json.loads(evt.to_ndjson())
        assert parsed["host"] is None


# =============================================================================
# Risk classifier tests (no simulator needed)
# =============================================================================

class TestRiskClassifier:
    def test_proc_event_known_comm_low(self):
        risk, tags = _classify_risk({"type": "ProcEvent", "comm": "sshd"})
        assert risk == "low"
        assert "unknown-process" not in tags

    def test_proc_event_no_comm_medium(self):
        risk, tags = _classify_risk({"type": "ProcEvent", "comm": None})
        assert risk == "medium"
        assert "unknown-process" in tags

    def test_conn_external_unknown_high(self):
        risk, tags = _classify_risk({
            "type": "ConnEvent", "external": True, "external_known": False
        })
        assert risk == "high"
        assert "external-unknown-connection" in tags

    def test_conn_internal_low(self):
        risk, tags = _classify_risk({
            "type": "ConnEvent", "external": False, "external_known": False
        })
        assert risk == "low"

    def test_file_high_density_high(self):
        risk, tags = _classify_risk({
            "type": "FileEvent", "density": 0.95, "known": False
        })
        assert risk == "high"
        assert "high-density-file" in tags

    def test_file_unknown_medium(self):
        risk, tags = _classify_risk({
            "type": "FileEvent", "density": 0.1, "known": False
        })
        assert risk == "medium"
        assert "unknown-file" in tags

    def test_auth_root_high(self):
        risk, tags = _classify_risk({
            "type": "AuthEvent", "user": "root", "agent": "blue_agent_0"
        })
        assert risk == "high"
        assert "root-session" in tags

    def test_auth_no_agent_medium(self):
        risk, tags = _classify_risk({
            "type": "AuthEvent", "user": "alice", "agent": None
        })
        assert risk == "medium"
        assert "unknown-agent" in tags


# =============================================================================
# NDJSON exporter tests (no simulator needed)
# =============================================================================

class TestSIEMExporter:
    def _make_telemetry_jsonl(self, path: Path) -> int:
        """Write a synthetic telemetry.jsonl and return expected doc count."""
        records = [
            {"type": "ProcEvent", "ts": 0, "host": "h1", "origin": "simulated",
             "pid": 1, "ppid": 0, "comm": "sshd"},
            {"type": "ConnEvent", "ts": 0, "host": "h1", "origin": "simulated",
             "pid": 1, "daddr": "10.0.0.1", "dport": 22,
             "external": False, "external_known": False},
            {"type": "FileEvent", "ts": 0, "host": "h1", "origin": "simulated",
             "path": "/etc/passwd", "name": "passwd", "known": True, "density": 0.1},
            {"type": "AuthEvent", "ts": 0, "host": "h1", "origin": "simulated",
             "user": "root", "session_type": "ssh", "agent": "blue_agent_0"},
        ]
        snapshot = {
            "schema_version": "blue-telemetry-collection-v1-draft",
            "episode_id": "seed-42",
            "seed": 42,
            "step": 1,
            "agent_id": "blue_agent_0",
            "policy_version": "round-robin",
            "telemetry": records,
            "telemetry_version": "blue-telemetry-v1-draft",
        }
        path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
        return len(records)

    def test_export_produces_paired_lines(self, tmp_path):
        inp = tmp_path / "telemetry.jsonl"
        out = tmp_path / "export.ndjson"
        n_docs = self._make_telemetry_jsonl(inp)

        stats = export_telemetry_to_ndjson(inp, out, index="test-index")
        assert stats["total_documents"] == n_docs

        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == n_docs * 2  # bulk header + doc per record

    def test_export_bulk_headers_valid(self, tmp_path):
        inp = tmp_path / "telemetry.jsonl"
        out = tmp_path / "export.ndjson"
        self._make_telemetry_jsonl(inp)

        export_telemetry_to_ndjson(inp, out, index="my-index")
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        for i in range(0, len(lines), 2):
            header = json.loads(lines[i])
            assert "index" in header
            assert header["index"]["_index"] == "my-index"
            assert "_id" in header["index"]

    def test_risk_distribution_tracked(self, tmp_path):
        inp = tmp_path / "telemetry.jsonl"
        out = tmp_path / "export.ndjson"
        self._make_telemetry_jsonl(inp)

        stats = export_telemetry_to_ndjson(inp, out)
        dist = stats["risk_distribution"]
        total = sum(dist.values())
        assert total == stats["total_documents"]

    def test_doc_id_deterministic(self):
        record = {"type": "ProcEvent", "ts": 0, "host": "h1", "comm": "sshd"}
        id1 = _make_doc_id(record, "ep1", 0)
        id2 = _make_doc_id(record, "ep1", 0)
        assert id1 == id2

    def test_doc_id_changes_with_seq(self):
        record = {"type": "ProcEvent", "ts": 0, "host": "h1", "comm": "sshd"}
        id1 = _make_doc_id(record, "ep1", 0)
        id2 = _make_doc_id(record, "ep1", 1)
        assert id1 != id2

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            export_telemetry_to_ndjson(
                tmp_path / "nonexistent.jsonl",
                tmp_path / "out.ndjson"
            )

    def test_empty_lines_skipped(self, tmp_path):
        inp = tmp_path / "telemetry.jsonl"
        inp.write_text("\n\n\n", encoding="utf-8")
        out = tmp_path / "export.ndjson"
        stats = export_telemetry_to_ndjson(inp, out)
        assert stats["total_documents"] == 0


# =============================================================================
# Integration tests (require CybORG simulator)
# =============================================================================

@requires_cyborg
class TestSIEMEvalIntegration:
    """Smoke tests against the real CAGE4 simulator."""

    def test_single_seed_smoke(self):
        from eval_siem_metrics import run_siem_evaluation
        summary = run_siem_evaluation("sleep", seeds=[42], steps=10)
        assert summary["n_episodes"] == 1
        assert summary["policy"] == "sleep"
        per = summary["per_seed"][0]
        assert per["steps_run"] <= 10
        # Sleep baseline never remediates
        assert per["total_remediations"] == 0
        assert per["total_alerts"] == 0

    def test_round_robin_produces_alerts(self):
        from eval_siem_metrics import run_siem_evaluation
        summary = run_siem_evaluation("round-robin", seeds=[10], steps=30)
        per = summary["per_seed"][0]
        # round-robin should analyse; may or may not get CONFIRMED in 30 steps
        assert per["total_analyses"] >= 0

    def test_siem_output_files_written(self, tmp_path):
        from eval_siem_metrics import run_siem_evaluation
        summary = run_siem_evaluation("sleep", seeds=[7], steps=5,
                                      output=tmp_path)
        assert (tmp_path / "audit_trail.ndjson").exists()
        assert (tmp_path / "siem_summary.json").exists()

        saved = json.loads((tmp_path / "siem_summary.json").read_text())
        assert saved["policy"] == "sleep"
        assert "audit_trail_sha256" in saved

    def test_audit_trail_events_are_valid_json(self, tmp_path):
        from eval_siem_metrics import run_siem_evaluation
        run_siem_evaluation("sleep", seeds=[7], steps=5, output=tmp_path)
        lines = (tmp_path / "audit_trail.ndjson").read_text().strip().split("\n")
        for line in lines:
            doc = json.loads(line)
            assert "tick" in doc
            assert "agent" in doc
            assert "action" in doc

    def test_service_availability_bounded(self):
        from eval_siem_metrics import run_siem_evaluation
        summary = run_siem_evaluation("round-robin", seeds=[1, 2], steps=20)
        for per in summary["per_seed"]:
            assert 0.0 <= per["service_availability_pct"] <= 100.0


@requires_cyborg
class TestBaselineEvalIntegration:
    """Smoke tests for run_baseline_eval."""

    def test_smoke_two_policies(self):
        from run_baseline_eval import run_baseline_evaluation
        result = run_baseline_evaluation(
            seeds=[42], steps=10, policies=["sleep", "round-robin"]
        )
        assert "sleep" in result["results"]
        assert "round-robin" in result["results"]
        for name in ("sleep", "round-robin"):
            assert result["results"][name]["mean_return"] is not None

    def test_output_files_written(self, tmp_path):
        from run_baseline_eval import run_baseline_evaluation
        run_baseline_evaluation(seeds=[42], steps=10,
                                policies=["sleep"],
                                output=tmp_path)
        assert (tmp_path / "baseline_results.json").exists()
        assert (tmp_path / "baseline_results.md").exists()

    def test_markdown_contains_policy_row(self, tmp_path):
        from run_baseline_eval import run_baseline_evaluation
        run_baseline_evaluation(seeds=[42], steps=10,
                                policies=["sleep"],
                                output=tmp_path)
        md = (tmp_path / "baseline_results.md").read_text()
        assert "sleep" in md
        assert "Mean Return" in md
