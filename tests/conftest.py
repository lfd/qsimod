"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from qsimod.artifact import HamiltonianModel
from qsimod.realise import HilbertSpace, RealisationRequest
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import (
    SchwingerGraph,
    analogue_pipeline,
    build_graph,
    local_occupation_subspace,
    particle_hole,
    superlattice,
    to_qubits,
)

#: The published reference parameters used by the digital tests.
REFERENCE_MASS = 0.41
REFERENCE_COUPLING = 0.83
REFERENCE_TIME = 2.0

#: A point well inside the perturbative window, used by the analogue tests.
WINDOW_MASS = 0.02
WINDOW_INTERACTION = 1.0
WINDOW_TUNNELLING_RATIO = 1 / 50
ELECTRIC_GAP = 0.5


@pytest.fixture
def graph3() -> SchwingerGraph:
    """The Schwinger DAG at ``N = 3``."""
    return build_graph(3)


@pytest.fixture
def graph4() -> SchwingerGraph:
    """The Schwinger DAG at ``N = 4``."""
    return build_graph(4)


def qubit_model(
    matter_sites: int,
    coupling: float = REFERENCE_COUPLING,
    mass: float = REFERENCE_MASS,
) -> HamiltonianModel:
    """The bound L2d qubit Hamiltonian at the given mass and coupling."""
    graph = build_graph(matter_sites)
    staggered = graph.graph.node("L2a").bind(**{P.MASS_L2A: mass, P.COUPLING_L2A: coupling})
    model = to_qubits().apply(particle_hole().apply(staggered))
    assert isinstance(model, HamiltonianModel)
    return model


def bosonic_space(matter_sites: int) -> HilbertSpace:
    """The Bose-Hubbard Hilbert space at the cutoff the encoding requires (``n_max = 2``)."""
    subspace = local_occupation_subspace(matter_sites)
    return HilbertSpace.of(
        superlattice(matter_sites).structure,
        RealisationRequest(boson_cutoff=subspace.required_cutoff()),
    )


def window_point(
    *,
    tunnelling_ratio: float = WINDOW_TUNNELLING_RATIO,
    tilt: float | None = None,
) -> dict[str, float]:
    """A Bose-Hubbard knob setting inside the validity window.

    ``U = 1``, ``delta = U/2 + m``, ``J = tunnelling_ratio * U``, and by default a tilt with
    ``kappa << Delta << delta, U``.
    """
    interaction = WINDOW_INTERACTION
    superlattice = WINDOW_MASS + interaction / 2
    tunnelling = tunnelling_ratio * interaction
    if tilt is None:
        # kappa ~ 8 sqrt(2) J^2 / U near resonance.
        coupling = 8.0 * 2.0**0.5 * tunnelling**2 / interaction
        tilt = min(10.5 * coupling, 0.095 * superlattice)
    return {
        P.TUNNELLING: tunnelling,
        P.INTERACTION: interaction,
        P.SUPERLATTICE: superlattice,
        P.TILT: tilt,
    }


def forward_map(knobs: Mapping[str, float]) -> dict[str, float]:
    """Evaluate the analogue pipeline's forward closed form at a knob setting."""
    relation = analogue_pipeline(3).relation
    classification = relation.classify(
        known=frozenset({P.TUNNELLING, P.INTERACTION, P.SUPERLATTICE, P.TILT})
    )
    assert classification.elimination is not None
    return relation.evaluate_closed_form(classification.elimination, dict(knobs))
