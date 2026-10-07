def get_red_observation(state):
    """
    Generate information visible to the Red agent.
    """

    return {
        "application_discovered": state.application_discovered,
        "api_discovered": state.api_discovered,
        "authenticated": state.authenticated,
        "user_access": state.user_access,
        "admin_access": state.admin_access,
        "compromised": state.compromised,
        "attacker_blocked": state.attacker_blocked,
        "step_count": state.step_count,
    }


def get_blue_observation(state):
    """
    Generate information visible to the Blue agent.

    Blue should not automatically receive hidden
    vulnerability information.
    """

    return {
        "web_server_up": state.web_server_up,
        "api_available": state.api_available,
        "database_available": state.database_available,
        "attack_detected": state.attack_detected,
        "attacker_blocked": state.attacker_blocked,
        "compromised": state.compromised,
        "step_count": state.step_count,
        "events": list(state.events),
    }