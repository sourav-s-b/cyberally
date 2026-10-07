def calculate_rewards(state, red_result, blue_result):
    """
    Calculate rewards for the Red and Blue agents.

    Returns:
        {
            "red": float,
            "blue": float
        }
    """

    red_reward = 0.0
    blue_reward = 0.0

    # Red rewards
    if red_result == "discovery_success":
        red_reward += 1.0

    elif red_result == "exploit_success":
        red_reward += 5.0

    elif red_result == "privilege_escalation":
        red_reward += 10.0

    # Blue rewards
    if blue_result == "attack_detected":
        blue_reward += 3.0

    elif blue_result == "attack_blocked":
        blue_reward += 5.0

    elif blue_result == "service_restored":
        blue_reward += 5.0

    # Defensive penalty
    if state.compromised:
        blue_reward -= 10.0

    return {
        "red": red_reward,
        "blue": blue_reward,
    }