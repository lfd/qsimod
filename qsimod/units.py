"""Units, dimensions and the 2*pi convention.

The conventions are ``hbar = 1``, the energy unit rad/ms and the time unit ms; frequencies are
angular.  Ordinary frequencies quoted in Hz or kHz are converted by
[`from_hertz`][qsimod.units.from_hertz] and [`from_kilohertz`][qsimod.units.from_kilohertz]
and their inverses; for instance, ``from_hertz(33.0) == 2*pi * 33 / 1000 == 0.2073...``.
"""

from __future__ import annotations

import math
from enum import Enum

__all__ = [
    "HBAR",
    "Dimension",
    "from_hertz",
    "from_kilohertz",
    "to_hertz",
    "to_kilohertz",
]

#: The reduced Planck constant, fixed at one by convention.
HBAR = 1.0


class Dimension(Enum):
    """The physical dimension of a parameter."""

    ENERGY = "energy"
    """An energy, equivalently an angular frequency, in rad/ms."""

    TIME = "time"
    """A time, in ms."""

    DIMENSIONLESS = "dimensionless"
    """A pure number: a ratio, a count, or a lattice spacing at ``a = 1``."""

    @property
    def unit(self) -> str:
        """The unit string used when a value of this dimension is printed."""
        return {
            Dimension.ENERGY: "rad/ms",
            Dimension.TIME: "ms",
            Dimension.DIMENSIONLESS: "",
        }[self]


def from_hertz(frequency_hz: float) -> float:
    """Convert an ordinary frequency in Hz to an energy in rad/ms."""
    return 2.0 * math.pi * frequency_hz * 1e-3


def to_hertz(energy: float) -> float:
    """Convert an energy in rad/ms to an ordinary frequency in Hz."""
    return energy * 1e3 / (2.0 * math.pi)


def from_kilohertz(frequency_khz: float) -> float:
    """Convert an ordinary frequency in kHz to an energy in rad/ms."""
    return 2.0 * math.pi * frequency_khz


def to_kilohertz(energy: float) -> float:
    """Convert an energy in rad/ms to an ordinary frequency in kHz."""
    return energy / (2.0 * math.pi)
