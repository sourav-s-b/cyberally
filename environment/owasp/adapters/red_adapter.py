from CybORG.Simulator.Actions import (
    DiscoverRemoteSystems,
    AggressiveServiceDiscovery,
    StealthServiceDiscovery,
    DiscoverDeception,
    ExploitRemoteService,
    PrivilegeEscalate,
    Impact,
    DegradeServices,
    Withdraw,
)

from ..actions import RedAction


def adapt_red_action(action, state=None):
    """
    Translate an existing CAGE4 Red action into an OWASP action.

    The adapter is state-aware because the CAGE4 Red agent can begin
    service discovery before our simplified OWASP model has explicitly
    marked the application as discovered.
    """

    if action is None:
        return RedAction.NOOP

    # ---------------------------------------------------------
    # Network/application discovery
    # ---------------------------------------------------------

    if isinstance(action, DiscoverRemoteSystems):
        return RedAction.DISCOVER_APPLICATION

    # ---------------------------------------------------------
    # Service discovery
    # ---------------------------------------------------------

    if isinstance(
        action,
        (AggressiveServiceDiscovery, StealthServiceDiscovery),
    ):
        if state is not None and not state.application_discovered:
            # In the simplified OWASP model, the first service
            # discovery also establishes application discovery.
            return RedAction.DISCOVER_APPLICATION

        return RedAction.DISCOVER_API

    # ---------------------------------------------------------
    # Exploitation
    # ---------------------------------------------------------

    if isinstance(action, ExploitRemoteService):
        return RedAction.SQL_INJECTION

    # ---------------------------------------------------------
    # Privilege escalation
    # ---------------------------------------------------------

    if isinstance(action, PrivilegeEscalate):
        return RedAction.PRIVILEGE_ESCALATION

    # ---------------------------------------------------------
    # Other CAGE4 actions
    # ---------------------------------------------------------

    if isinstance(action, DegradeServices):
        return RedAction.NOOP

    if isinstance(action, Impact):
        return RedAction.NOOP

    if isinstance(
        action,
        (DiscoverDeception, Withdraw),
    ):
        return RedAction.NOOP

    return RedAction.NOOP