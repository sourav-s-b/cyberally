from ipaddress import IPv4Address, IPv4Network

from CybORG.Shared.Enums import TernaryEnum


JUICE_SHOP_IP = IPv4Address("10.0.0.10")
JUICE_SHOP_SUBNET = IPv4Network("10.0.0.0/24")
JUICE_SHOP_HOSTNAME = "juice-shop"


def build_red_observation(
    state,
    success=TernaryEnum.UNKNOWN,
    action=None,
):
    """
    Convert the internal JuiceShopState into the CAGE4-style
    observation expected by the existing PlannedRedAgent.

    The observation also exposes a CAGE4-style session when
    the OWASP simulation has granted the Red agent user access.
    This prevents the existing CAGE4 agent from interpreting
    an active OWASP session as a removed session.
    """

    observation = {
        "success": success,
        "action": action,

        str(JUICE_SHOP_IP): {
            "System info": {
                "Hostname": JUICE_SHOP_HOSTNAME,
            },

            "Interface": [
                {
                    "ip_address": JUICE_SHOP_IP,
                    "Subnet": JUICE_SHOP_SUBNET,
                }
            ],
        },
    }

    # -------------------------------------------------------------
    # Expose the simulated Red session to the existing CAGE4 agent.
    #
    # CAGE4's _session_removal_state_change() checks for a
    # "Sessions" field when a host is in U/UD/R/RD state.
    #
    # Without this field, UD would incorrectly be interpreted
    # as a removed session and reset to KD.
    # -------------------------------------------------------------

    if state.user_access:

        observation[str(JUICE_SHOP_IP)]["Sessions"] = [
            {
                "session_id": 0,
                "agent": "red_agent",
            }
        ]

    return observation