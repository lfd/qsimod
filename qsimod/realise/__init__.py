"""Numerical realisation of symbolic models as dense ``complex128`` operators in JAX.

The package provides operator construction, basis ordering, subspace and sector projectors,
exact diagonalisation and time evolution by eigendecomposition.  It supports the numerical
validation of the static declarations made along a pipeline by a classical simulation of the
same model at small chain length.  Importing the package enables the 64-bit mode of JAX via
[`qsimod.jax_setup`][qsimod.jax_setup].
"""

from qsimod.realise.build import (
    build_operator,
    build_pauli_string,
    build_pauli_sum,
    local_subspace_projector,
    restrict,
    sandwich,
    sector_projector,
    subspace_indices,
)
from qsimod.realise.evolve import (
    commutator_spectral_norm,
    eigensystem,
    evolve_state,
    expectation,
    gauss_violation,
    hermiticity_defect,
    low_lying_spectrum,
    manifold_levels,
    max_abs_deviation,
    propagator,
    spacing_deviation,
    spectral_norm,
)
from qsimod.realise.hilbert import HilbertSpace, LocalSpace, RealisationRequest
from qsimod.realise.sector import SECTOR_TOLERANCE, SectorBasis, build_operator_in

__all__ = [
    "SECTOR_TOLERANCE",
    "HilbertSpace",
    "LocalSpace",
    "RealisationRequest",
    "SectorBasis",
    "build_operator",
    "build_operator_in",
    "build_pauli_string",
    "build_pauli_sum",
    "commutator_spectral_norm",
    "eigensystem",
    "evolve_state",
    "expectation",
    "gauss_violation",
    "hermiticity_defect",
    "local_subspace_projector",
    "low_lying_spectrum",
    "manifold_levels",
    "max_abs_deviation",
    "propagator",
    "restrict",
    "sandwich",
    "sector_projector",
    "spacing_deviation",
    "spectral_norm",
    "subspace_indices",
]
