"""Canonical local parameter names of the model library.

A builder names its parameters inside a [`Namespace`][qsimod.parameters.Namespace]; the fully
qualified name is ``<namespace>.<local>``, and the local halves are fixed in this module.
"""

from __future__ import annotations

__all__ = [
    "ANISOTROPY",
    "COUPLING",
    "ELECTRIC_GAP",
    "FIELD",
    "GAUGE_COUPLING",
    "HOPPING",
    "INTERACTION",
    "INTERACTION_DOWN",
    "INTERACTION_MIXED",
    "INTERACTION_UP",
    "LATTICE_SPACING",
    "LONGITUDINAL_BIAS",
    "LONGITUDINAL_COUPLING",
    "LONGITUDINAL_FIELD",
    "MASS",
    "SUPERLATTICE",
    "TILT",
    "TRANSVERSE_AMPLITUDE",
    "TRANSVERSE_COUPLING",
    "TRANSVERSE_FIELD",
    "TUNNELLING",
]

# -- theory-side ---------------------------------------------------------------

MASS = "m"
"""The fermion rest mass ``m``."""

COUPLING = "kappa"
"""The gauge-invariant matter-gauge coupling ``kappa``, written ``t~`` by Yang et al. (2020)."""

LATTICE_SPACING = "a"
"""The lattice spacing ``a``; dimensionless, since ``a -> 1`` below the quantum-link truncation."""

GAUGE_COUPLING = "e"
"""The gauge coupling ``e``; dimensionless, since ``e -> 1`` below the quantum-link truncation."""

ELECTRIC_GAP = "electric_gap"
"""The energy cost of one additional unit of electric flux on a link, ``a e**2 / 2``."""

# -- optical-lattice knobs -----------------------------------------------------

TUNNELLING = "J"
"""The nearest-neighbour tunnelling amplitude ``J``."""

INTERACTION = "U"
"""The on-site interaction ``U``."""

SUPERLATTICE = "delta"
"""The staggered superlattice depth ``delta``."""

TILT = "Delta"
"""The linear potential tilt ``Delta``."""

HOPPING = "t"
"""The nearest-neighbour tunnelling ``t`` of a model whose spin couplings are written ``J``."""

INTERACTION_UP = "U_uu"
"""The on-site interaction between two atoms of the first (spin-up) component."""

INTERACTION_DOWN = "U_dd"
"""The on-site interaction between two atoms of the second (spin-down) component."""

INTERACTION_MIXED = "U_ud"
"""The on-site interaction between one atom of each component."""

# -- spin-chain knobs ----------------------------------------------------------

TRANSVERSE_COUPLING = "Jxy"
"""The nearest-neighbour ``XX + YY`` (spin-exchange) coupling ``Jxy``."""

LONGITUDINAL_COUPLING = "Jz"
"""The nearest-neighbour ``ZZ`` coupling ``Jz``."""

ANISOTROPY = "Delta"
"""The XXZ anisotropy ``Delta = Jz / Jxy``, dimensionless.

The glyph coincides with that of [`TILT`][qsimod.models.names.TILT]; the two names never share
a namespace.
"""

FIELD = "h"
"""The transverse or longitudinal field strength ``h``."""

TRANSVERSE_FIELD = "hx"
"""The transverse field ``hx`` of an Ising chain in units of its coupling; dimensionless."""

LONGITUDINAL_FIELD = "hz"
"""The longitudinal field ``hz`` of an Ising chain in units of its coupling; dimensionless."""

TRANSVERSE_AMPLITUDE = "Gamma"
"""The transverse field of an Ising chain as an energy, ``Gamma = Jz * hx``."""

LONGITUDINAL_BIAS = "B"
"""The longitudinal field of an Ising chain as an energy, ``B = Jz * hz``."""
