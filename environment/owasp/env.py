from .actions import RedAction, BlueAction
from .adapters.red_adapter import adapt_red_action
from .observations import get_red_observation, get_blue_observation
from .rewards import calculate_rewards
from .state import JuiceShopState
from .results import TernaryEnum


class OWASPEnvironment:
    """
    Simulated OWASP Juice Shop environment integrated with CAGE4.

    Maintains simulated application-layer state and provides:
    - Native OWASP step() interface.
    - CAGE4-compatible execute_red_action() interface.
    """

    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.state = JuiceShopState()

    def reset(self, seed=None):
        self.state.reset()

        return {
            "red": get_red_observation(self.state),
            "blue": get_blue_observation(self.state),
        }

    def execute_red_action(self, cage4_action):
        """
        Execute an existing CAGE4 Red action
        inside the OWASP Juice Shop simulation.

        The existing CAGE4 action is first translated
        into an OWASP RedAction by the adapter.
        """

        # Count this as one Red-agent environment step.
        self.state.step_count += 1

        # Convert the existing CAGE4 action into
        # an OWASP-specific action.
        red_action = adapt_red_action(
            cage4_action,
            self.state,
        )

        # Execute the translated action.
        result = self._execute_red_action(
            red_action
        )

        # ---------------------------------------------------------
        # Convert OWASP result into CAGE4 TernaryEnum
        # ---------------------------------------------------------

        if result in {
            "discovery_success",
            "exploit_success",
            "privilege_escalation",
        }:
            return TernaryEnum.TRUE

        if result == "attacker_blocked":
            return TernaryEnum.FALSE

        if result in {
            "application_not_discovered",
            "api_not_discovered",
            "no_user_access",
            "vulnerability_not_available",
            "already_discovered",
            "already_admin",
            "unknown_red_action",
        }:
            return TernaryEnum.FALSE

        return TernaryEnum.UNKNOWN

    def step(self, actions):
        """
        Internal OWASP simulation interface.

        Example:

            actions = {
                "red": RedAction.DISCOVER_APPLICATION,
                "blue": BlueAction.MONITOR,
            }

        Returns:

            observations,
            rewards,
            terminated,
            truncated,
            info
        """

        red_action = actions.get(
            "red",
            RedAction.NOOP,
        )

        blue_action = actions.get(
            "blue",
            BlueAction.NOOP,
        )

        # One normal environment step.
        self.state.step_count += 1

        # Execute Red action.
        red_result = self._execute_red_action(
            red_action
        )

        # Execute Blue action.
        blue_result = self._execute_blue_action(
            blue_action
        )

        # Calculate rewards.
        rewards = calculate_rewards(
            self.state,
            red_result,
            blue_result,
        )

        # Generate observations.
        observations = {
            "red": get_red_observation(
                self.state
            ),
            "blue": get_blue_observation(
                self.state
            ),
        }

        # The Red agent has achieved the main objective
        # when the application is compromised and
        # administrator access has been obtained.
        terminated = (
            self.state.compromised
            and self.state.admin_access
        )

        # Stop the episode after max_steps.
        truncated = (
            self.state.step_count
            >= self.max_steps
        )

        info = {
            "red_result": red_result,
            "blue_result": blue_result,
            "step": self.state.step_count,
            "events": list(self.state.events),
        }

        return (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )

    # =============================================================
    # RED ACTION EXECUTION
    # =============================================================

    def _execute_red_action(self, action):

        # ---------------------------------------------------------
        # Attacker blocked: blocked attacker cannot trigger new incidents
        # ---------------------------------------------------------

        if self.state.attacker_blocked:
            return "attacker_blocked"

        # ---------------------------------------------------------
        # No operation
        # ---------------------------------------------------------

        if action == RedAction.NOOP:
            return "noop"

        # ---------------------------------------------------------
        # Application discovery
        # ---------------------------------------------------------

        if action == RedAction.DISCOVER_APPLICATION:

            if not self.state.application_discovered:

                self.state.application_discovered = True

                self.state.add_event(
                    "Red discovered the Juice Shop application"
                )

                return "discovery_success"

            return "already_discovered"

        # ---------------------------------------------------------
        # API discovery
        # ---------------------------------------------------------

        if action == RedAction.DISCOVER_API:

            if not self.state.application_discovered:
                return "application_not_discovered"

            if not self.state.api_discovered:

                self.state.api_discovered = True

                self.state.add_event(
                    "Red discovered a Juice Shop API endpoint"
                )

                return "discovery_success"

            return "already_discovered"

        # ---------------------------------------------------------
        # SQL Injection (simulated OWASP incident)
        # ---------------------------------------------------------

        if action == RedAction.SQL_INJECTION:

            # Preconditions
            if not self.state.application_discovered:
                return "application_not_discovered"

            if not self.state.sql_injection_exposed:
                return "vulnerability_not_available"

            # Exploitation reaches the API as part of the attack.
            self.state.api_discovered = True

            # Application-layer simulated incident state
            self.state.incident_active = True
            self.state.incident_type = "sql_injection"
            self.state.user_access = True
            self.state.authenticated = True
            self.state.compromised = True

            self.state.add_event(
                "Simulated OWASP incident: SQL injection on port 3000 succeeded"
            )

            return "exploit_success"

        # ---------------------------------------------------------
        # Broken Authentication (simulated OWASP incident)
        # ---------------------------------------------------------

        if action == RedAction.BROKEN_AUTHENTICATION:

            if not self.state.application_discovered:
                return "application_not_discovered"

            if not self.state.broken_auth_exposed:
                return "vulnerability_not_available"

            self.state.incident_active = True
            self.state.incident_type = "broken_authentication"
            self.state.authenticated = True
            self.state.user_access = True
            self.state.compromised = True

            self.state.add_event(
                "Simulated OWASP incident: Authentication bypass succeeded"
            )

            return "exploit_success"

        # ---------------------------------------------------------
        # IDOR (simulated OWASP incident)
        # ---------------------------------------------------------

        if action == RedAction.IDOR:

            if not self.state.api_discovered:
                return "api_not_discovered"

            if not self.state.idor_exposed:
                return "vulnerability_not_available"

            self.state.incident_active = True
            self.state.incident_type = "idor"
            self.state.user_access = True
            self.state.compromised = True

            self.state.add_event(
                "Simulated OWASP incident: IDOR exploitation succeeded"
            )

            return "exploit_success"

        # ---------------------------------------------------------
        # XSS (simulated OWASP incident)
        # ---------------------------------------------------------

        if action == RedAction.XSS:

            if not self.state.application_discovered:
                return "application_not_discovered"

            if not self.state.xss_exposed:
                return "vulnerability_not_available"

            self.state.incident_active = True
            self.state.incident_type = "xss"
            self.state.compromised = True

            self.state.add_event(
                "Simulated OWASP incident: XSS attack succeeded"
            )

            return "exploit_success"

        # ---------------------------------------------------------
        # Privilege Escalation
        # ---------------------------------------------------------

        if action == RedAction.PRIVILEGE_ESCALATION:

            if not self.state.user_access:
                return "no_user_access"

            if self.state.admin_access:
                return "already_admin"

            self.state.admin_access = True

            self.state.add_event(
                "Red escalated privileges to administrator"
            )

            return "privilege_escalation"

        # ---------------------------------------------------------
        # Unknown Red action
        # ---------------------------------------------------------

        return "unknown_red_action"

    # =============================================================
    # BLUE ACTION EXECUTION
    # =============================================================

    def _execute_blue_action(self, action):

        # ---------------------------------------------------------
        # No operation
        # ---------------------------------------------------------

        if action == BlueAction.NOOP:
            return "noop"

        # ---------------------------------------------------------
        # Monitor
        # ---------------------------------------------------------

        if action == BlueAction.MONITOR:

            self.state.add_event(
                "Blue monitored application activity"
            )

            return "monitoring"

        # ---------------------------------------------------------
        # Detect
        # ---------------------------------------------------------

        if action == BlueAction.DETECT:

            # Only detect if there is an active incident or compromise
            if not (self.state.incident_active or self.state.compromised):
                return "no_attack_detected"

            # Prevent duplicate detection rewards if already detected
            if self.state.attack_detected:
                return "already_detected"

            self.state.attack_detected = True

            self.state.add_event(
                "Blue detected malicious activity"
            )

            return "attack_detected"

        # ---------------------------------------------------------
        # Block
        # ---------------------------------------------------------

        if action == BlueAction.BLOCK:

            # Blocking requires prior detection
            if not self.state.attack_detected:
                return "nothing_to_block"

            # Prevent duplicate blocking rewards if already blocked
            if self.state.attacker_blocked:
                return "already_blocked"

            self.state.attacker_blocked = True

            self.state.add_event(
                "Blue blocked the attacker"
            )

            return "attack_blocked"

        # ---------------------------------------------------------
        # Restore
        # ---------------------------------------------------------

        if action == BlueAction.RESTORE:

            # Only restore if there is an active incident, compromise, or unauthorized access
            if not (
                self.state.compromised
                or self.state.incident_active
                or self.state.user_access
                or self.state.admin_access
                or self.state.authenticated
            ):
                return "already_clean"

            self.state.compromised = False
            self.state.incident_active = False
            self.state.incident_type = "none"
            self.state.authenticated = False
            self.state.user_access = False
            self.state.admin_access = False

            # Note: self.state.attacker_blocked is intentionally NOT cleared,
            # ensuring the attacker remains blocked after application restoration.

            self.state.add_event(
                "Blue restored the application state"
            )

            return "service_restored"

        # ---------------------------------------------------------
        # Unknown Blue action
        # ---------------------------------------------------------

        return "unknown_blue_action"
