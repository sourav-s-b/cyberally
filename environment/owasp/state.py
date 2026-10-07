from dataclasses import dataclass, field


@dataclass
class JuiceShopState:
    """
    Simulated state of the OWASP Juice Shop environment.
    """

    # Application state
    web_server_up: bool = True
    api_available: bool = True
    database_available: bool = True

    # Discovery
    application_discovered: bool = False
    api_discovered: bool = False

    # Authentication / privilege
    authenticated: bool = False
    user_access: bool = False
    admin_access: bool = False

    # Vulnerability state
    sql_injection_exposed: bool = True
    broken_auth_exposed: bool = True
    idor_exposed: bool = True
    xss_exposed: bool = True

    # Attack state
    compromised: bool = False
    attack_detected: bool = False

    # Defensive state
    attacker_blocked: bool = False

    # Episode information
    step_count: int = 0
    events: list = field(default_factory=list)

    def reset(self):
        """Restore the environment to its initial state."""
        self.web_server_up = True
        self.api_available = True
        self.database_available = True

        self.application_discovered = False
        self.api_discovered = False

        self.authenticated = False
        self.user_access = False
        self.admin_access = False

        self.sql_injection_exposed = True
        self.broken_auth_exposed = True
        self.idor_exposed = True
        self.xss_exposed = True

        self.compromised = False
        self.attack_detected = False
        self.attacker_blocked = False

        self.step_count = 0
        self.events.clear()

    def add_event(self, event: str):
        self.events.append(event)