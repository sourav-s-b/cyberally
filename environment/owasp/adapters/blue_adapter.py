from CybORG.Simulator.Actions.AbstractActions import (
    Analyse,
    Remove,
    Restore,
)

from ..actions import BlueAction


def adapt_blue_action(action):
    """
    Translate an existing CAGE4 Blue action into
    an OWASP BlueAction.

    Existing Blue policies continue to select the
    original CAGE4 action. This adapter translates
    that decision into our Juice Shop simulation.
    """

    if action is None:
        return BlueAction.NOOP

    if isinstance(action, Analyse):
        return BlueAction.DETECT

    if isinstance(action, Remove):
        return BlueAction.BLOCK

    if isinstance(action, Restore):
        return BlueAction.RESTORE

    return BlueAction.NOOP