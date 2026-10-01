# Proposal: blue-telemetry-schema (Path 2 translation layer)

- Status: draft, unreviewed; implemented locally on `blue/telemetry-schema`
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

## Local implementation and refinements (2026-10-01)

Local additive producer: `blue_telemetry.py`, optional wrapper method
`get_telemetry(agent_id)`, and `blue_collect_telemetry.py`. Tested dependency:
main `2f0cfb67ae1d3b2da6a1b1ccff28eff9e1a5c915`, with the pre-existing local
Blue training changes preserved. This is not an accepted shared contract.

The code refines the sketch below; affected-role review is still required:

- JSON records include a `type` discriminator and immutable `origin: simulated`.
  Strict decoding rejects missing/falsified origin, unknown fields/types,
  wall-clock timestamps, negative ticks, invalid IDs/ports and nonfinite density.
- Unobserved PID, PPID, process name, destination, port, path and session fields
  remain null. `comm` maps observed `process_name`. `FileEvent.name` retains
  `File Name` separately from directory `Path`. `AuthEvent.agent` preserves
  Blue-visible session ownership needed by the existing feature extractor.
  No privileged Red-session labels or attacker-name inference are added.
- `ConnEvent.external_known` marks whether subnet/address information supports
  the classification. Carried reset subnets preserve existing boundaries.
  Unknown external status gives false for legacy feature parity plus this flag.
- File `known` uses the existing UNKNOWN-marker test; missing classification
  gives true and missing/null density gives 0.0 for legacy parity. These defaults
  do not prove benignness. This adapter does not introduce a new missingness model.
- Blocks declare `telemetry_version: blue-telemetry-v1-draft`,
  `telemetry_time_unit: sim_tick`, and `telemetry_semantics: latest_observed_rows`.
  Each row retains its source-field observation tick, never export time.
  `field_observed_at` retains empty-field freshness; missing is not observed-empty.
  Repeated snapshots must not be accumulated as new event counts. Process and
  session rows do not establish that an exec or login event occurred.
- `records_to_vector` reproduces all ten legacy features from one host's schema
  records. The collection command is the first optional consumer. Existing actor
  features, masks, rewards, policy manifests and historical anomaly scripts are
  unchanged. The latter still need a separate corrected detector experiment.

Example, from the root (artifacts under ignored `runs/`):

```powershell
.venv-train/Scripts/python.exe cage-challenge-4/blue_collect_telemetry.py --output runs/telemetry-demo --seeds 8123 8124 --steps 30 --policy round-robin
```

The output directory must be new. JSONL contains five local agent snapshots per
tick, including reset/final observations, with field freshness and schema-derived
features. The manifest records actual seeds, policy, schema/feature/action versions,
source commit/hashes, runtime versions and dataset hash. Native Red is active:
this data is unlabeled, not a verified benign training dataset. No detector is fit.
Consumers group records by episode, agent and host. No truth-label file is emitted.

Validation: 18 focused telemetry tests and 60 tests in the isolated main-based
publication checkout passed (32.59 s). The earlier 67-test result includes
uncommitted local MAPPO tests and eight existing upstream warnings. Tests cover JSON round-trip and all-feature parity, sparse/stale
data, all five live agents, passive observations, Analyse completion, reset,
unchanged features/masks, manifest hashing and refusal to overwrite artifacts.

PR checklist / rollout: review additive producer first; migrate future anomaly and
SIEM consumers in separate changes. Environment reviews sensor field realism;
these are sensor-inspired shapes, not validated real-sensor interchangeable APIs.
Evaluation owns JSONL/SIEM adapters and rejection of sim ticks in wall-clock metrics.
No accepted contract or teammate notification is implied by this local work.

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

- Local implementation and tests exist as described above; review and merge
  remain pending. The original 3–5 day estimate is not measured completion time.
- Untested: integration with any real sensor; Environment's field-shape review.
- Affected-role review: pending (Environment for schema realism,
  Evaluation for JSONL/SIEM fit). Do not mark accepted just because this
  file exists.
