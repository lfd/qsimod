"""Abstraction levels: the axis along which the model library is indexed.

Four levels are ordered by value: the application layer (1), the intermediate representations
(2), the hardware layer (3) and the executable layer (4).  The executable layer is named but
out of scope.  A level is not a type; structural typing is defined in
[`qsimod.structure`][qsimod.structure].
"""

from __future__ import annotations

from enum import Enum

__all__ = ["AbstractionLevel"]


class AbstractionLevel(Enum):
    """The layer of the model graph at which an artifact is located.

    The values are the level numbers, so that levels are ordered.
    """

    APPLICATION = 1
    """The application layer: a physical theory stated in its own terms, independent of any
    hardware model."""

    INTERMEDIATE = 2
    """An intermediate representation, shaped by the theory and by the hardware, usually still a
    Hamiltonian."""

    HARDWARE = 3
    """The hardware layer: the model a simulator realises natively, either an analogue
    Hamiltonian over the hardware knobs or a digital ordered product of k-local unitaries on a
    qubit register."""

    EXECUTABLE = 4
    """The executable layer: a routed and scheduled gate set, or a pulse schedule.  This layer
    is out of scope."""

    @property
    def in_scope(self) -> bool:
        """Whether the package models the level."""
        return self is not AbstractionLevel.EXECUTABLE

    @property
    def label(self) -> str:
        """A short lower-case name for reports and table columns."""
        return self.name.lower()

    def __lt__(self, other: AbstractionLevel) -> bool:
        if not isinstance(other, AbstractionLevel):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: AbstractionLevel) -> bool:
        if not isinstance(other, AbstractionLevel):
            return NotImplemented
        return self.value <= other.value

    def __str__(self) -> str:
        return f"level {self.value} ({self.label})"
