"""DRAFT action-masking rules for a CAGE4 Blue policy. Sketch only -- not final.

Real action signatures (verified against live env):
    Monitor(session, agent)                          # detect, duration 1
    Analyse(session, agent, hostname)                # confirm,  duration 2
    Remove(session, agent, hostname)                 # remediate user-level
    Restore(session, agent, hostname)                # remediate root-level, disruptive
    DeployDecoy / BlockTrafficZone / AllowTrafficZone / Sleep

Per-host states: CLEAN -> SUSPICIOUS -> CONFIRMED -> REMEDIATING -> CLEAN.

LANDMINE (verified live): Analyse takes 2 steps. The first step() returns
    {'success': IN_PROGRESS}, NOT a failure. Masking logic MUST treat
    IN_PROGRESS as "wait, don't re-issue". Misreading it as failure causes the
    agent to re-trigger Analyse every step, starving Remove/Restore and
    looking like a policy bug when it is really a wrapper bug. Track pending
    actions per hostname with an expiry (pending_until_step) and mask out any
    action for that host until it resolves.
"""

# Shared subnet logic lives in blue_obs_features (extract_subnets,
# is_external) -- import from there, don't reimplement zone boundaries.
# No hardcoded subnets anywhere: always derive from the reset snapshot.
from blue_obs_features import extract_subnets, is_external


def conn_leaves_zone(conn, own_subnets):
    """Draft helper: does this observed connection leave the host's zone?

    Own subnets must be carried from the reset snapshot (Analyse-revealed
    views carry only ip_address). Returns False when subnets unknown.
    """
    return is_external(conn.get("remote_address"), own_subnets)

# Host-state -> allowed actions (DRAFT, revisit once detector exists).
MASK_RULES = {
    # No detection: only cheap sensing or nothing. Remove/Restore masked OFF.
    "CLEAN": ["Monitor", "Analyse", "DeployDecoy", "Sleep"],
    # Monitor flagged something, not yet confirmed: confirm, don't remediate.
    "SUSPICIOUS": ["Analyse", "Monitor", "Sleep"],
    # Analyse confirmed compromise: remediate. User-level -> Remove first.
    "CONFIRMED_USER": ["Remove", "Monitor", "Sleep"],
    # Root-level: Remove won't cut it, Restore (disruptive, use sparingly).
    "CONFIRMED_ROOT": ["Restore", "Remove", "Monitor", "Sleep"],
    # Action already IN_PROGRESS on this host: only Sleep (wait it out).
    "REMEDIATING": ["Sleep"],
}

# Disruptive actions excluded from the default mask until reward shaping
# justifies them: Block/AllowTrafficZone hurt Green productivity and the
# built-in random baseline (cc4BlueRandomAgent) already excludes them.
EXCLUDED_BY_DEFAULT = ["BlockTrafficZone", "AllowTrafficZone"]

# Suggested escalation order per host, cheapest-informative first:
#   Monitor -> Analyse -> (Remove | Restore) -> back to Monitor.
# Never: Analyse while IN_PROGRESS; Remove/Restore on CLEAN hosts.

# --- Review fix: single-scalar detection lives HERE (masking), not in RL obs.
# The RL policy gets raw file dims (n_files, n_unknown_files, max_density)
# and learns its own weighting; this hard yes/no is only for masks.
# High precision (0/62 clean false positives in test), imperfect recall
# (~75%: aimed Analyse reveals nothing a quarter of the time), hence the
# 2-strike clearing rule below -- "no files != clean" is encoded in the
# state machine, not just in the RL features.
def detection_hit(revealed_host):
    """Hard yes/no: did this Analyse-revealed host view show implants?"""
    files = revealed_host.get("Files", []) or []
    if not files:
        return False
    return any("UNKNOWN" in str(f.get("Known File", "")) for f in files)


# --- Review fix: 2-strike rule. One empty Analyse must NOT clear
# SUSPICIOUS -> CLEAN (25% miss rate even when aimed correctly).
# Require N consecutive empty Analyses before downgrading.
CLEAR_AFTER_EMPTY_ANALYSES = 2


class BlueZoneTracker:
    """Per-host masking state machine for one Blue agent's zone.

    States: CLEAN -> SUSPICIOUS -> CONFIRMED_USER/CONFIRMED_ROOT,
    REMEDIATING while an action is IN_PROGRESS (mask everything but Sleep).
    Call note_action_issued() when acting, note_analyse_result() on resolve,
    and mask_for(host) to get the allowed discrete-action names.
    """

    def __init__(self, hostnames):
        self.state = {h: "CLEAN" for h in hostnames}
        self.empty_strikes = {h: 0 for h in hostnames}
        self.pending_until = {}  # host -> env step when IN_PROGRESS resolves

    def note_action_issued(self, host, action_name, duration, step):
        if action_name in ("Analyse", "Remove", "Restore") and host is not None:
            self.pending_until[host] = step + duration
            self.state[host] = "REMEDIATING"

    def note_analyse_result(self, host, revealed_host, step):
        self.pending_until.pop(host, None)
        if detection_hit(revealed_host):
            self.empty_strikes[host] = 0
            self.state[host] = "CONFIRMED_USER"  # root-ness unresolved by
            # files alone; policy may still choose Restore
        else:
            self.empty_strikes[host] += 1
            if self.empty_strikes[host] >= CLEAR_AFTER_EMPTY_ANALYSES:
                # Two consecutive empties clear anything (incl. CONFIRMED:
                # treated as remediated-elsewhere until re-detected).
                self.state[host] = "CLEAN"
                self.empty_strikes[host] = 0
            elif self.state[host].startswith("CONFIRMED"):
                pass  # one empty view doesn't un-confirm; keep strikes
            else:
                # From CLEAN, SUSPICIOUS, or REMEDIATING -> SUSPICIOUS.
                # (REMEDIATING must transition here: pending was popped, so
                # note_step() will never move it -- stuck hosts otherwise.)
                self.state[host] = "SUSPICIOUS"

    def note_step(self, step):
        for h in [h for h, u in self.pending_until.items() if u <= step]:
            del self.pending_until[h]
            if self.state[h] == "REMEDIATING":
                self.state[h] = "SUSPICIOUS"  # resolved w/o recorded result

    def mask_for(self, host):
        """Allowed action-name set for one host (mirror of MASK_RULES)."""
        if host in self.pending_until:
            return {"Sleep"}
        return set(MASK_RULES.get(self.state.get(host, "CLEAN"), ["Sleep"]))
