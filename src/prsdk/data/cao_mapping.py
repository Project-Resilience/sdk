"""
Immutable NamedTuple for storing the context, actions, and outcomes for a given project.
Note: We choose to use NamedTuple over dataclasses because NamedTuple is immutable.
"""
from typing import NamedTuple


class CAOMapping(NamedTuple):
    """
    Class defining the context, actions, and outcomes for a given project.
    """
    context: list[str]
    actions: list[str]
    outcomes: list[str]
