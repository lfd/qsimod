"""Solver backends: the only modules of the package that import an optimiser.

The physics modules declare equations, admissible sets and validity conditions in
[`qsimod.relations`][qsimod.relations], [`qsimod.parameters`][qsimod.parameters] and
[`qsimod.validity`][qsimod.validity]; none of these imports a backend.
"""

from qsimod.solving.backends.base import BackendCapability, SolverBackend
from qsimod.solving.backends.closed_form import ClosedFormBackend
from qsimod.solving.backends.integer import (
    DiscreteSolution,
    MonotoneBisectionBackend,
    ResourceChoice,
    minimal_feasible_integer,
    minimal_resource_setting,
)
from qsimod.solving.backends.scipy_nlp import ScipyNlpBackend

__all__ = [
    "BackendCapability",
    "ClosedFormBackend",
    "DiscreteSolution",
    "MonotoneBisectionBackend",
    "ResourceChoice",
    "ScipyNlpBackend",
    "SolverBackend",
    "minimal_feasible_integer",
    "minimal_resource_setting",
]
