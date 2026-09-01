"""Suzuki-Trotter product formulas, their a-priori error bounds and their resource summary.

The package implements the digital simulator model of the article, the product formula
``U_approx``.  It is generic over any Hamiltonian given as a layer decomposition of a Pauli sum:
the a-priori error bound is computed from the commutators of the layers, and the resource summary
counts gates and circuit depth in layers.  Both are derived from the term structure by Pauli
algebra.
"""

from qsimod.trotter.bounds import (
    AdaptiveNormEstimator,
    ErrorBound,
    NormEstimator,
    OneNormEstimator,
    SpectralNormEstimator,
    commutator_bound,
    error_bound,
    taylor_remainder_bound,
)
from qsimod.trotter.formulas import (
    ProductFormulaSchedule,
    Stage,
    lie_trotter,
    strang,
    supported_orders,
    suzuki,
    suzuki_parameter,
)
from qsimod.trotter.layers import Layer, LayerDecomposition, declared_layers, derive_layers
from qsimod.trotter.schedule import (
    ProductFormulaModel,
    ResourceSummary,
    UnitaryFactor,
    as_product_formula,
    factors_for,
    global_phase_for,
    layer_depth_of,
)

__all__ = [
    "AdaptiveNormEstimator",
    "ErrorBound",
    "Layer",
    "LayerDecomposition",
    "NormEstimator",
    "OneNormEstimator",
    "ProductFormulaModel",
    "ProductFormulaSchedule",
    "ResourceSummary",
    "SpectralNormEstimator",
    "Stage",
    "UnitaryFactor",
    "as_product_formula",
    "commutator_bound",
    "declared_layers",
    "derive_layers",
    "error_bound",
    "factors_for",
    "global_phase_for",
    "layer_depth_of",
    "lie_trotter",
    "strang",
    "supported_orders",
    "suzuki",
    "suzuki_parameter",
    "taylor_remainder_bound",
]
