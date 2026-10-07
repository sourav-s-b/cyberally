import numpy as np

from blue.policies.hybrid import HybridBluePolicy
from blue.core.masking import BlueZoneTracker

from CybORG.Simulator.Actions.AbstractActions import (
    Analyse,
    Remove,
    Restore,
)

from CybORG.Simulator.Actions import (
    Sleep,
    Monitor,
)

from .actions import BlueAction
from .adapters.blue_adapter import adapt_blue_action


class OWASPBluePolicyAdapter:
    """
    Compatibility layer between the existing CAGE4 Blue policy
    and the OWASP Juice Shop simulation.

    Existing HybridBluePolicy and BlueZoneTracker are reused
    without modification.
    """

    def __init__(self, agent="blue_agent_0"):
        self.agent = agent

        # Existing CAGE4 Blue policy
        self.policy = HybridBluePolicy()

        # One OWASP Juice Shop host
        self.hostnames = {
            agent: [
                "10.0.0.10",
            ]
        }

        self.subnets = {
            agent: {
                "10.0.0.10": "10.0.0.0/24",
            }
        }

        # CAGE4-style host view
        self.views = {
            agent: {
                "10.0.0.10": {
                    "System info": {
                        "Hostname": "juice-shop",
                    },
                    "Interface": [
                        {
                            "ip_address": "10.0.0.10",
                            "Subnet": "10.0.0.0/24",
                        }
                    ],
                }
            }
        }

        # Existing Blue tracker
        self.trackers = {
            agent: BlueZoneTracker(
                ["10.0.0.10"]
            )
        }

        self._tick = 0

    # ---------------------------------------------------------
    # CAGE4 action mask
    # ---------------------------------------------------------
    def get_avail_agent_actions(self, agent_index=0):
        """
        Action indices:

            0 -> Sleep
            1 -> Monitor
            2 -> Analyse(juice-shop)
            3 -> Remove(juice-shop)
            4 -> Restore(juice-shop)
        """

        return np.array(
            [
                1,  # Sleep
                1,  # Monitor
                1,  # Analyse
                1,  # Remove
                1,  # Restore
            ],
            dtype=np.int8,
        )

    # ---------------------------------------------------------
    # Decode CAGE4 action index
    # ---------------------------------------------------------
    def decode_action(self, action_index):

        if action_index == 0:
            return Sleep()

        if action_index == 1:
            return Monitor()

        if action_index == 2:
            return Analyse(
                session=0,
                agent=self.agent,
                hostname="10.0.0.10",
            )

        if action_index == 3:
            return Remove(
                session=0,
                agent=self.agent,
                hostname="10.0.0.10",
            )

        if action_index == 4:
            return Restore(
                session=0,
                agent=self.agent,
                hostname="10.0.0.10",
            )

        return None

    # ---------------------------------------------------------
    # Update CAGE4-style view from OWASP state
    # ---------------------------------------------------------
    def update_from_owasp_state(self, state):

        self._tick += 1

        host = self.views[self.agent]["10.0.0.10"]

        host["System info"]["Hostname"] = "juice-shop"

        host["Interface"] = [
            {
                "ip_address": "10.0.0.10",
                "Subnet": "10.0.0.0/24",
            }
        ]

        if state.compromised:
            host["Processes"] = [
                {
                    "Properties": [
                        "suspicious"
                    ]
                }
            ]
        else:
            host.pop("Processes", None)

    # ---------------------------------------------------------
    # Build the observation returned by Analyse
    # ---------------------------------------------------------
    def build_analyse_result(self, state):
        """
        Create the minimal CAGE4-style Analyse result required
        by the existing BlueZoneTracker.

        detection_hit() checks:

            revealed_host["Files"]

        and considers a host detected when one file contains:

            "Known File": "UNKNOWN"
        """

        revealed_host = {
            "System info": {
                "Hostname": "juice-shop",
            },

            "Interface": [
                {
                    "ip_address": "10.0.0.10",
                    "Subnet": "10.0.0.0/24",
                }
            ],

            "Files": [],
        }

        if state.compromised:
            revealed_host["Files"].append(
                {
                    "Known File": "UNKNOWN",
                }
            )

        return revealed_host

    # ---------------------------------------------------------
    # Update existing BlueZoneTracker
    # ---------------------------------------------------------
    def update_tracker(
        self,
        cage4_action,
        result,
        state,
    ):
        """
        Feed the result of the OWASP action back into the
        existing CAGE4 BlueZoneTracker.

        This reproduces the important part of CAGE4's
        wrapper feedback loop.
        """

        tracker = self.trackers[self.agent]

        host = "10.0.0.10"

        # -----------------------------------------------------
        # Analyse
        # -----------------------------------------------------
        if isinstance(cage4_action, Analyse):

            if result == "attack_detected":

                revealed_host = self.build_analyse_result(
                    state
                )

                tracker.note_analyse_result(
                    host=host,
                    revealed_host=revealed_host,
                    step=self._tick,
                )

            else:

                revealed_host = {
                    "Files": []
                }

                tracker.note_analyse_result(
                    host=host,
                    revealed_host=revealed_host,
                    step=self._tick,
                )

        # -----------------------------------------------------
        # Remove / Restore
        # -----------------------------------------------------
        elif isinstance(
            cage4_action,
            (Remove, Restore),
        ):

            if result in {
                "attack_blocked",
                "service_restored",
            }:

                tracker.note_remediation_result(
                    host=host,
                    action_name=type(
                        cage4_action
                    ).__name__,
                    step=self._tick,
                )

        # -----------------------------------------------------
        # Failure
        # -----------------------------------------------------
        else:

            if result not in {
                "noop",
                "monitoring",
            }:
                tracker.note_failure(
                    host=host,
                    action_name=type(
                        cage4_action
                    ).__name__,
                )

    # ---------------------------------------------------------
    # Select Blue action
    # ---------------------------------------------------------
    def select_action(self, state):

        self.update_from_owasp_state(
            state
        )

        action_index = self.policy.select(
            self,
            self.agent,
        )

        cage4_action = self.decode_action(
            action_index
        )

        owasp_action = adapt_blue_action(
            cage4_action
        )

        return (
            action_index,
            cage4_action,
            owasp_action,
        )

    # ---------------------------------------------------------
    # Execute Blue action
    # ---------------------------------------------------------
    def execute_action(
        self,
        environment,
        cage4_action,
        owasp_action,
    ):

        if owasp_action == BlueAction.NOOP:
            result = "noop"

        else:
            result = environment._execute_blue_action(
                owasp_action
            )

        # Feed result back into existing tracker
        self.update_tracker(
            cage4_action,
            result,
            environment.state,
        )

        return result