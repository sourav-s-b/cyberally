"""SIEM NDJSON exporter: converts a collected telemetry.jsonl to Elasticsearch/Splunk NDJSON.

Reads the output of blue_collect_telemetry.py and writes a SIEM-ingestible
NDJSON file with event classification, enrichment, and an index-bulk header
suitable for Elasticsearch _bulk API.

Usage:
    python siem_export_ndjson.py --input results/telemetry_run/telemetry.jsonl \
        --output results/telemetry_run/siem_export.ndjson \
        --index cyberally-blue-telemetry

Output format (per record pair):
    {"index": {"_index": "<index>", "_id": "<sha256-of-event>"}}
    {<enriched event document>}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BLUE = _ROOT / "blue"
if str(_BLUE) not in sys.path:
    sys.path.insert(0, str(_BLUE))

from blue_telemetry import TELEMETRY_VERSION

EXPORT_VERSION = "siem-ndjson-export-v1"

# Risk classification per event type (Blue-visible heuristics only)
RISK_MAP = {
    "ProcEvent": {
        "default": "low",
        "unknown_comm": "medium",
    },
    "ConnEvent": {
        "internal": "low",
        "external_unknown": "high",
        "external_known": "medium",
    },
    "FileEvent": {
        "known": "low",
        "unknown": "medium",
        "high_density": "high",  # density > 0.8
    },
    "AuthEvent": {
        "default": "low",
        "root_session": "high",
        "unknown_agent": "medium",
    },
}


def _classify_risk(record: dict) -> tuple[str, list[str]]:
    """Return (risk_level, [tags]) based purely on Blue-visible fields."""
    rtype = record.get("type", "")
    tags = []
    risk = "info"

    if rtype == "ProcEvent":
        risk = "low"
        if not record.get("comm"):
            risk = "medium"
            tags.append("unknown-process")

    elif rtype == "ConnEvent":
        if record.get("external") and not record.get("external_known"):
            risk = "high"
            tags.append("external-unknown-connection")
        elif record.get("external"):
            risk = "medium"
            tags.append("external-known-connection")
        else:
            risk = "low"

    elif rtype == "FileEvent":
        density = float(record.get("density", 0.0))
        if density > 0.8:
            risk = "high"
            tags.append("high-density-file")
        elif not record.get("known"):
            risk = "medium"
            tags.append("unknown-file")
        else:
            risk = "low"

    elif rtype == "AuthEvent":
        if record.get("user") == "root":
            risk = "high"
            tags.append("root-session")
        elif not record.get("agent"):
            risk = "medium"
            tags.append("unknown-agent")
        else:
            risk = "low"

    return risk, tags


def _make_doc_id(record: dict, episode_id: str, seq: int) -> str:
    """Deterministic document ID from content hash."""
    raw = json.dumps({"ep": episode_id, "seq": seq, **record}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def export_telemetry_to_ndjson(
    input_path: Path,
    output_path: Path,
    index: str = "cyberally-blue-telemetry",
) -> dict:
    """Convert a telemetry.jsonl file to SIEM NDJSON bulk format."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    total_docs = 0
    risk_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "info": 0}

    with (input_path.open("r", encoding="utf-8") as fin,
          output_path.open("w", encoding="utf-8") as fout):

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                snapshot = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] line {line_no}: JSON decode error: {e}", file=sys.stderr)
                continue

            episode_id = snapshot.get("episode_id", f"line-{line_no}")
            seed = snapshot.get("seed", -1)
            step = snapshot.get("step", -1)
            agent_id = snapshot.get("agent_id", "unknown")
            policy = snapshot.get("policy_version", "unknown")

            for seq, record in enumerate(snapshot.get("telemetry", [])):
                risk, tags = _classify_risk(record)
                risk_counts[risk] = risk_counts.get(risk, 0) + 1

                doc_id = _make_doc_id(record, episode_id, seq)
                doc = {
                    "export_version": EXPORT_VERSION,
                    "telemetry_version": TELEMETRY_VERSION,
                    "episode_id": episode_id,
                    "seed": seed,
                    "step": step,
                    "agent_id": agent_id,
                    "policy": policy,
                    "risk_level": risk,
                    "tags": tags,
                    **record,
                }

                # Elasticsearch bulk header
                fout.write(json.dumps(
                    {"index": {"_index": index, "_id": doc_id}},
                    allow_nan=False
                ) + "\n")
                fout.write(json.dumps(doc, allow_nan=False) + "\n")
                total_docs += 1

    stats = {
        "export_version": EXPORT_VERSION,
        "input": str(input_path),
        "output": str(output_path),
        "index": index,
        "total_documents": total_docs,
        "risk_distribution": risk_counts,
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to telemetry.jsonl from blue_collect_telemetry.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output NDJSON file path")
    parser.add_argument("--index", default="cyberally-blue-telemetry",
                        help="Elasticsearch index name")
    args = parser.parse_args()

    stats = export_telemetry_to_ndjson(args.input, args.output, args.index)

    print(f"Exported {stats['total_documents']} documents to {args.output}")
    print(f"Risk distribution: {stats['risk_distribution']}")
