"""Blue evidence tracking, separate from simulator action validity.

Evidence is not ground truth. Pending actions never overwrite the prior belief,
and successful remediation requires subsequent verification. The optional
``evidence`` mask is a heuristic restriction; wrappers default to validity only.
"""
from blue.core.obs_features import is_external


def conn_leaves_zone(conn, own_subnets):
    return is_external(conn.get("remote_address"), own_subnets)


def detection_hit(revealed_host):
    return any("UNKNOWN" in str(f.get("Known File", ""))
               for f in (revealed_host.get("Files", []) or []))


CLEAR_AFTER_EMPTY_ANALYSES = 2
MASK_RULES = {
    "UNKNOWN": ["Analyse", "Monitor", "Sleep"],
    "CLEAN": ["Analyse", "Monitor", "Sleep"],
    "SUSPICIOUS": ["Analyse", "Monitor", "Sleep"],
    "CONFIRMED": ["Analyse", "Remove", "Restore", "Monitor", "Sleep"],
    "VERIFY": ["Analyse", "Remove", "Restore", "Monitor", "Sleep"],
}
EXCLUDED_BY_DEFAULT = ["BlockTrafficZone", "AllowTrafficZone"]


class BlueZoneTracker:
    def __init__(self, hostnames):
        self.state = {h: "UNKNOWN" for h in hostnames}
        self.empty_strikes = {h: 0 for h in hostnames}
        self.pending_until = {}
        self.last_analysis = {h: None for h in hostnames}
        self.last_remediation = {h: None for h in hostnames}
        self.last_result = {h: None for h in hostnames}

    def note_action_issued(self, host, action_name, duration, step):
        if self.pending_until:
            raise RuntimeError("Agent already has a pending action")
        self.pending_until[host] = step + duration

    def note_analyse_result(self, host, revealed_host, step):
        self.pending_until.pop(host, None)
        self.last_analysis[host] = step
        self.last_result[host] = ("Analyse", "TRUE")
        if detection_hit(revealed_host):
            self.state[host] = "CONFIRMED"  # privilege is not observable here
            self.empty_strikes[host] = 0
        else:
            self.empty_strikes[host] += 1
            if self.empty_strikes[host] >= CLEAR_AFTER_EMPTY_ANALYSES:
                # No current detection, NOT a ground-truth clean label.
                self.state[host] = "UNKNOWN"
                self.empty_strikes[host] = 0

    def note_remediation_result(self, host, action_name, step):
        self.pending_until.pop(host, None)
        self.last_result[host] = (action_name, "TRUE")
        self.last_remediation[host] = step
        self.state[host] = "VERIFY"
        self.empty_strikes[host] = 0

    def note_failure(self, host, action_name):
        self.pending_until.pop(host, None)
        self.last_result[host] = (action_name, "FALSE")

    def note_step(self, step):
        # Duration is an expectation, not evidence of completion. Wrappers fail
        # explicitly on overdue unresolved actions instead of silently unmasking.
        pass

    def mask_for(self, host, mode="evidence"):
        if mode not in ("validity", "evidence"):
            raise ValueError("mask mode must be validity or evidence")
        if self.pending_until or host not in self.state:
            return {"Sleep"}
        if mode == "validity":
            return {"Sleep", "Monitor", "Analyse", "Remove", "Restore"}
        return set(MASK_RULES[self.state[host]])
