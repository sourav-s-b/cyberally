"""Draft simulator telemetry adapter; accepts ONLY caller-supplied Blue views.

Records describe observed rows, not inferred exec/connect/auth occurrences.
Repeated exports are snapshots of retained evidence, not new event counts.
No simulator/controller or privileged-label access belongs in this module.
"""
from dataclasses import asdict, dataclass, field
from ipaddress import IPv4Address
import math

from blue_obs_features import extract_subnets, is_external

TELEMETRY_VERSION = "blue-telemetry-v1-draft"


@dataclass(frozen=True)
class TelemetryRecord:
    ts: int
    host: str
    origin: str = field(default="simulated", init=False)

    def __post_init__(self):
        if type(self.ts) is not int or self.ts < 0:
            raise ValueError("ts must be a nonnegative simulator tick")
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be the nonempty simulator hostname")
        for name in ("pid", "ppid", "dport"):
            value = getattr(self, name, None)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or null")
        if getattr(self, "dport", 0) is not None and getattr(self, "dport", 0) > 65535:
            raise ValueError("dport exceeds 65535")
        for name in ("comm", "daddr", "path", "name", "user", "session_type", "agent"):
            value = getattr(self, name, None)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null")
        if hasattr(self, "density") and (type(self.density) not in (int, float)
                                         or not math.isfinite(self.density)):
            raise ValueError("density must be finite")
        for name in ("known", "external", "external_known"):
            if hasattr(self, name) and type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class ProcEvent(TelemetryRecord):
    pid: int | None
    ppid: int | None
    comm: str | None


@dataclass(frozen=True)
class ConnEvent(TelemetryRecord):
    pid: int | None
    daddr: str | None
    dport: int | None
    external: bool
    external_known: bool


@dataclass(frozen=True)
class FileEvent(TelemetryRecord):
    path: str | None
    name: str | None
    known: bool
    density: float


@dataclass(frozen=True)
class AuthEvent(TelemetryRecord):
    user: str | None
    session_type: str | None
    agent: str | None


RECORD_TYPES = {cls.__name__: cls for cls in (ProcEvent, ConnEvent, FileEvent, AuthEvent)}


def _address_known(address):
    try:
        IPv4Address(address)
        return True
    except (ValueError, TypeError):
        return False


def record_to_dict(record):
    if type(record).__name__ not in RECORD_TYPES or record.origin != "simulated":
        raise ValueError("unsupported record or origin")
    record.__post_init__()
    return {"type": type(record).__name__, **asdict(record)}


def record_from_dict(data):
    """Strict JSON boundary: reject absent/falsified origin and unknown fields."""
    values = dict(data)
    if values.pop("origin", None) != "simulated":
        raise ValueError("origin must be simulated")
    cls = RECORD_TYPES.get(values.pop("type", None))
    if cls is None:
        raise ValueError("unknown telemetry record type")
    return cls(**values)


def _text(value):
    return None if value is None else str(value)


def host_to_records(host, observation, *, tick, observed_at=None, own_subnets=None):
    """Translate one local host view without mutating it.

    Raw observations use tick. Retained wrapper views MUST supply observed_at;
    each nonempty field must then have its own timestamp (never retimestamp stale
    rows). Subnets may be carried from reset just as in host_to_vector.
    """
    TelemetryRecord(tick, host)
    subnets = extract_subnets(observation) if own_subnets is None else own_subnets
    records = []
    for source in ("Processes", "Files", "Sessions"):
        rows = observation.get(source, []) or []
        if not rows:
            continue
        ts = tick if observed_at is None else observed_at[source]
        if ts > tick:
            raise ValueError("observation timestamp is in the future")
        for row in rows:
            if source == "Processes":
                pid = row.get("PID")
                records.append(ProcEvent(ts, host, pid, row.get("PPID"),
                                         _text(row.get("process_name"))))
                for conn in row.get("Connections", []) or []:
                    address = _text(conn.get("remote_address"))
                    # Unknown subnet/address yields False for legacy parity,
                    # plus an explicit availability flag for new consumers.
                    records.append(ConnEvent(
                        ts, host, pid, address, conn.get("remote_port"),
                        is_external(address, subnets),
                        bool(subnets) and _address_known(address)))
            elif source == "Files":
                records.append(FileEvent(
                    ts, host, _text(row.get("Path")), _text(row.get("File Name")),
                    "UNKNOWN" not in str(row.get("Known File", "")),
                    float(row.get("Density", 0.0) or 0.0)))
            else:
                records.append(AuthEvent(ts, host, _text(row.get("username")),
                                         _text(row.get("Type")), _text(row.get("agent"))))
    return records


def records_to_vector(records, own_agent):
    """Same ten legacy features from one host's schema records, without raw dicts.

    This preserves legacy zero/default behavior, not a new missingness model.
    Do not concatenate snapshots across ticks before calling this function.
    """
    records = list(records)
    if len({r.host for r in records}) > 1:
        raise ValueError("features require a single host snapshot")
    sessions = [r for r in records if isinstance(r, AuthEvent)]
    files = [r for r in records if isinstance(r, FileEvent)]
    connections = [r for r in records if isinstance(r, ConnEvent)]
    return [len(sessions), sum(isinstance(r, ProcEvent) for r in records), len(files),
            sum("UNKNOWN" in (s.session_type or "") for s in sessions),
            sum(s.agent != own_agent for s in sessions),
            int(any(s.user == "root" for s in sessions)),
            sum(not f.known for f in files),
            max((f.density for f in files), default=0.0), len(connections),
            sum(c.external for c in connections)]


def telemetry_block(views, observed_at, *, tick, subnets_by_host=None):
    """Optional JSONL block for ONE agent; caller controls observation visibility.

    field_observed_at retains empty-vs-missing field freshness, which row lists
    alone cannot encode. ts on each row is the source field's observation tick.
    """
    subnets_by_host = subnets_by_host or {}
    rows = []
    for host, view in views.items():
        for key, value in observed_at[host].items():
            if key in ("Processes", "Files", "Sessions"):
                TelemetryRecord(value, host)
                if value > tick:
                    raise ValueError("observation timestamp is in the future")
        rows.extend(record_to_dict(r) for r in host_to_records(
            host, view, tick=tick, observed_at=observed_at[host],
            own_subnets=subnets_by_host.get(host)))
    return {"telemetry_version": TELEMETRY_VERSION, "telemetry_time_unit": "sim_tick",
            "telemetry_semantics": "latest_observed_rows", "telemetry": rows,
            "field_observed_at": {
                host: {key: value for key, value in observed_at[host].items()
                       if key in ("Processes", "Files", "Sessions")}
                for host in views}}
