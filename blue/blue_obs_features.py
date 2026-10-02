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
