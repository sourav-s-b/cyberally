# Proposal: blue-telemetry-schema (Path 2 translation layer)

- Status: draft, unreviewed
- Author role and branch: Blue, `blue/mappo-training`
- Producer and affected consumers: Producer Blue (draft; long-term producer
  is Environment once real sensors exist); consumers Blue (featurizer,
  anomaly scorer), Evaluation (SIEM export, MITRE audit mapping). Red: none,
  except that Red action labels may later map to TTP tags.
- Related issue/PR and exact dependency commits: follows BLUE-01 wrapper
  (`7b76fc2`, merged as `f8a9deb`); consumes `host_to_vector` feature
  definitions in `cage-challenge-4/blue_obs_features.py`. No dependency on
  container, eBPF, or LLM-Red work — this proposal explicitly does not
  require any of it.

## Problem and current behavior

Every downstream consumer (featurizer, baselines' belief tracker, the future
anomaly scorer, the future SIEM exporter) reads CybORG's raw simulation
dicts directly (`host['Processes']`, `host['Files']`, …). If real sensors
ever exist, every consumer must be rewritten. There is also a standing
credibility risk: simulator output reshaped into real-looking records must
never be describable as captured telemetry (see Honesty clause below).

## Proposed contract change

Additive only. A translation layer renders sim events into **real-shaped
schemas**; no consumer is forced to move, and nothing about the simulation
changes.

```python
# New module (proposed): cage-challenge-4/blue_telemetry.py
@dataclass(frozen=True)
class ProcEvent:   # eBPF exec-shaped
    ts: int            # sim tick (NOT wall-clock; see Honesty clause)
    host: str          # real CAGE4 hostname, never renamed
    pid: int; ppid: int | None; comm: str
    origin: str = "simulated"   # always present, always this value today

@dataclass(frozen=True)
class ConnEvent:  # eBPF connect-shaped
    ts: int; host: str; pid: int
    daddr: str; dport: int; external: bool   # external via extract_subnets
    origin: str = "simulated"

@dataclass(frozen=True)
class FileEvent:  # OSQuery file-row-shaped
    ts: int; host: str; path: str; known: bool   # known == Known File flag
    density: float
    origin: str = "simulated"

@dataclass(frozen=True)
class AuthEvent:  # syslog-shaped
    ts: int; host: str; user: str; session_type: str
    origin: str = "simulated"
```

Sim → schema mapping (initial):

| Sim source | Schema record |
|---|---|
| `Processes[]` entry (+ `Connections[]`) | one `ProcEvent` + one `ConnEvent` per connection |
| `Files[]` entry | one `FileEvent` (`known` = `Known File != UNKNOWN`) |
| `Sessions[]` entry | one `AuthEvent` (user/session type as observed) |
| `observed_at` freshness | carried as event `ts`; consumers must treat staleness explicitly |

JSONL step records gain an optional `telemetry: [...]` block of these
records. Schema version recorded as `telemetry_version` in every artifact
manifest alongside `feature_version`/`action_version`.

### Honesty clause (normative, not advisory)

- Every record carries `origin: "simulated"` until a real sensor produces
  it. Any writer that drops or falsifies this field is a contract violation.
- `ts` is a simulator tick. It must never be presented as wall-clock time,
  latency, or MTTR/MTTD evidence.
- Approved description: "simulator output structured identically to
  eBPF/syslog/OSQuery telemetry so downstream code runs unmodified once
  real sensors integrate; the underlying data is simulated."
- Forbidden description: any sentence implying captured, measured, or
  kernel-level telemetry exists in this project today.

## Required branch changes

| Role | Files/behavior to update | Dependency | Acceptance test |
|---|---|---|---|
| Blue | new `blue_telemetry.py` mapping + tests; scorer data collection reads schemas, not raw dicts | none (additive) | round-trip test: every sim field used by `host_to_vector` appears in exactly one schema record; `origin` present on all records |
| Evaluation | JSONL reader accepts optional `telemetry` block; SIEM/MITRE mapping builds on schema fields | this lands | fixture: schema records validate; sim-tick `ts` rejected where wall-clock required |
| Environment | owns the future real producer; reviews field names against real eBPF/syslog/OSQuery shapes | this lands | field-name review sign-off (no code required now) |
| Red | none | — | confirm none exist |

## Merge and migration order

1. Schema module + mapping + tests land; no consumer moves. Main stays
   runnable; current `host_to_vector` path untouched.
2. BLUE-05 data collection is the first consumer (collect through schemas).
   SIEM/auditor work follows when scheduled.
3. A future real sensor becomes a second producer of the same schemas with
   `origin` set accordingly — consumers do not change. No removal step:
   the sim producer stays for training speed permanently.

## Validation and decision

- Not yet implemented; estimated 3–5 days (schema day, mapping day,
  JSONL + round-trip day, buffer for review).
- Untested: mapping coverage of edge-case sim fields; Environment's
  field-shape review.
- Affected-role review: pending (Environment for schema realism,
  Evaluation for JSONL/SIEM fit). Do not mark accepted just because this
  file exists.
