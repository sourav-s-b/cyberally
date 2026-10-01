"""Convert raw CAGE4 Blue observations into fixed-length numeric vectors (v2).

Bridge piece between the raw obs dict and anything ML-shaped
(IsolationForest, MAPPO policy input).

v2 changes (see experiment notes):
- v1 trained on reset snapshots, but the detector at runtime scores
  Analyse-revealed host views -- a different distribution. v2 is designed
  for Analyse-revealed inputs: file implant signals, connection details.
- New features: unknown-file count, max file density, connection counts
  incl. external-connection count (red C2 leaves the subnet, green SSH
  usually doesn't).

Layout of inputs (verified against live env):
- reset snapshot host dict: 'Interface' [{'Subnet': IPv4Network, ...}],
  'Processes', 'Sessions', 'System info', 'User Info'.
- Analyse-revealed host dict: subset with 'Files' [{'File Name', 'Density',
  'Known File', 'Known Path', 'Path'}], 'Processes' [{'PID', 'Connections':
  [{'local_address', 'remote_address', 'local_port', 'remote_port'}]}],
  'Interface' [{'ip_address'}], 'System info'.

A host is *compromised* iff some entry of host['Sessions'] has
agent == 'red_agent_N' in TRUE state. That matches the visualiser's own
ground truth (VisualiseRedExpansion.py uses `"red" in agent`).

NOTE on stealth: file-implant features are strong vs DiscoveryFSRed
(cmd.sh/escalate.sh in /tmp, Density 0.9, Known File UNKNOWN). A
stealthier Red may degrade exactly these features -- expect AUC to drop
if the Red agent changes; that means the threat changed, not that the
detector is broken.
"""

from ipaddress import IPv4Address, IPv4Network

FEATURE_NAMES = [
    "n_sessions",            # len(host['Sessions'])
    "n_processes",           # len(host['Processes'])
    "n_files",               # len(host.get('Files', []))
    "n_unknown_sessions",    # sessions whose Type is UNKNOWN
    "n_foreign_sessions",    # sessions NOT owned by this blue agent
    "has_root_session",      # any session with username == 'root' (0/1)
    "n_unknown_files",       # Files with Known File == UNKNOWN (implants)
    "max_file_density",      # max Files[].Density, 0.0 if no files
    "n_connections",         # total Processes[].Connections[] entries
    "n_external_connections",  # connections whose remote_address leaves
                               # the host's own subnets (C2-like)
]

VECTOR_LEN = len(FEATURE_NAMES)

# Index of has_root_session in the base vector. The wrapper can drop it
# (BLUE-04 ablation): a Blue-visible root session is not proof of an
# attacker root session, so its value as an actor input is tested, not
# assumed.
ROOT_SESSION_INDEX = 5

# Blue-visible belief states tracked by BlueZoneTracker (blue_action_masking).
# Ordered for a fixed one-hot layout; append-only if states are ever added.
BELIEF_STATES = ("UNKNOWN", "CLEAN", "SUSPICIOUS", "CONFIRMED", "VERIFY")

# Temporal/belief feature groups (BLUE-04). Fixed order; dims vary by subset.
# Ages and freshness are normalized by the episode horizon and capped at 1.0;
# "never observed" maps to 1.0 (maximally stale), which is distinct from
# 0.0 (just observed) — missingness is signal, never silent zero-fill.
TEMPORAL_GROUPS = ("ages", "belief", "freshness", "mission")
TEMPORAL_GROUP_LENS = {"ages": 2, "belief": len(BELIEF_STATES),
                       "freshness": 2, "mission": 1}


def temporal_len(groups=TEMPORAL_GROUPS):
    """Total temporal width for a subset of TEMPORAL_GROUPS."""
    groups = tuple(groups or ())
    unknown = [g for g in groups if g not in TEMPORAL_GROUP_LENS]
    if unknown:
        raise ValueError(f"unknown temporal groups: {unknown}")
    return sum(TEMPORAL_GROUP_LENS[g] for g in groups)


def host_to_temporal(tracker, host, observed_at, tick, horizon,
                     pending_busy=False, groups=TEMPORAL_GROUPS):
    """Blue-visible temporal/belief features for one host (BLUE-04).

    All inputs are the agent's own belief/observation bookkeeping — never
    privileged truth. ``tracker`` needs ``.state``/``.last_analysis``/
    ``.last_remediation``/``.empty_strikes`` dicts (BlueZoneTracker
    satisfies this; no import to avoid a dependency cycle).
    ``observed_at`` maps field name -> last-observed tick. ``horizon`` is
    the episode step bound used to cap and normalize ages.
    """
    groups = tuple(groups or ())
    unknown = [g for g in groups if g not in TEMPORAL_GROUP_LENS]
    if unknown:
        raise ValueError(f"unknown temporal groups: {unknown}")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    cap = horizon
    out = []
    if "ages" in groups:
        last_an = tracker.last_analysis.get(host)
        last_re = tracker.last_remediation.get(host)
        out += [min(1.0, (tick - last_an) / cap) if last_an is not None else 1.0,
                min(1.0, (tick - last_re) / cap) if last_re is not None else 1.0]
    if "belief" in groups:
        state = tracker.state.get(host, "UNKNOWN")
        out += [1.0 if state == s else 0.0 for s in BELIEF_STATES]
    if "freshness" in groups:
        stamps = [t for t in (observed_at or {}).values()
                  if isinstance(t, (int, float))]
        newest = min(1.0, (tick - max(stamps)) / cap) if stamps else 1.0
        out += [newest, 1.0 if pending_busy else 0.0]
    if "mission" in groups:
        out += [min(1.0, tick / cap)]
    return out


def _sessions(host):
    return host.get("Sessions", []) or []


def extract_subnets(host_obs):
    """Own subnets of a host from its Interface list.

    Shared helper: action-masking code should import this rather than
    reimplementing subnet parsing. No hardcoded subnet boundaries anywhere:
    subnets always come from the obs itself (reset snapshot). Analyse-
    revealed views carry only ip_address, so callers must carry subnets
    over from the reset snapshot for the same host.
    """
    subnets = []
    for iface in host_obs.get("Interface", []) or []:
        sub = iface.get("Subnet")
        if sub is None:
            continue
        if not isinstance(sub, IPv4Network):
            try:
                sub = IPv4Network(str(sub))
            except ValueError:
                continue
        subnets.append(sub)
    return subnets


def _as_address(addr):
    if addr is None:
        return None
    if isinstance(addr, IPv4Address):
        return addr
    try:
        return IPv4Address(str(addr))
    except ValueError:
        return None


def is_external(remote_address, own_subnets):
    """True if remote_address lies outside all of own_subnets.

    Returns False when subnets are unknown (can't tell) -- callers that
    need the distinction must supply subnets from the reset snapshot.
    Shared with action-masking (same "leaves the zone" notion).
    """
    remote = _as_address(remote_address)
    if remote is None or not own_subnets:
        return False
    return not any(remote in s for s in own_subnets)


def _iter_connections(host_obs):
    for proc in host_obs.get("Processes", []) or []:
        for conn in proc.get("Connections", []) or []:
            yield conn


def host_to_vector(host_obs, own_agent, own_subnets=None):
    """Map one host's obs dict -> fixed-length numeric list (v2).

    Uses ONLY unprivileged fields Blue actually sees. Red-agent-name
    matching stays quarantined in `host_ground_truth` so it can't leak
    into the unsupervised detector.
    """
    sessions = _sessions(host_obs)
    files = host_obs.get("Files", []) or []
    if own_subnets is None:
        own_subnets = extract_subnets(host_obs)
    n_conn = n_ext = 0
    for conn in _iter_connections(host_obs):
        n_conn += 1
        if is_external(conn.get("remote_address"), own_subnets):
            n_ext += 1
    densities = [f.get("Density", 0.0) or 0.0 for f in files]
    return [
        len(sessions),
        len(host_obs.get("Processes", []) or []),
        len(files),
        sum(1 for s in sessions if "UNKNOWN" in str(s.get("Type", ""))),
        sum(1 for s in sessions if s.get("agent") != own_agent),
        1 if any(s.get("username") == "root" for s in sessions) else 0,
        sum(1 for f in files if "UNKNOWN" in str(f.get("Known File", ""))),
        max(densities) if densities else 0.0,
        n_conn,
        n_ext,
    ]


def obs_to_matrix(observation, own_agent, subnets_by_host=None):
    """Full per-agent obs -> (hostnames, matrix). Skips success/action keys."""
    subnets_by_host = subnets_by_host or {}
    hostnames = [k for k in observation if k not in ("success", "action")]
    matrix = [host_to_vector(observation[h], own_agent,
                             subnets_by_host.get(h)) for h in hostnames]
    return hostnames, matrix


def host_ground_truth(host_obs):
    """PRIVILEGED label for sanity-checking the featurizer/detector only.

    Returns 0 (clean), 1 (user-level red session), 2 (root-level red session).
    NEVER feed this into the detector or policy at train or eval time.
    """
    worst = 0
    for s in _sessions(host_obs):
        if "red" in str(s.get("agent", "")):
            worst = max(worst, 2 if s.get("username") == "root" else 1)
    return worst
