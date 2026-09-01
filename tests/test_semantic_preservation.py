"""Numerical checks that the Schwinger transformations preserve the physics they claim to."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise

import jax.numpy as jnp
import pytest
from jax import Array

from qsimod.artifact import HamiltonianModel
from qsimod.pipeline import ModelGraph
from qsimod.realise import (
    HilbertSpace,
    build_operator,
    commutator_spectral_norm,
    eigensystem,
    evolve_state,
    expectation,
    hermiticity_defect,
    local_subspace_projector,
    low_lying_spectrum,
    restrict,
    sandwich,
    spectral_norm,
    subspace_indices,
)
from qsimod.relations import RelationKind
from qsimod.scalar import Symbol
from qsimod.symbolic import number, word
from qsimod.transform import Exactness, ExactnessError, Transformation
from qsimod.transformations.basis_changes import (
    ParticleHoleTransformation,
    fermions_to_qubits_image,
)
from qsimod.transformations.encodings import hardcore_boson_encoding
from qsimod.usecases.base import UseCaseGraph
from qsimod.usecases.heisenberg import build_graph as heisenberg_graph
from qsimod.usecases.ising import build_graph as ising_graph
from qsimod.usecases.schwinger import (
    L2B,
    L2C,
    canonical_state_configuration,
    effective_bosonic,
    homogeneous_quantum_link,
    local_occupation_subspace,
    matter_occupation_observable,
    particle_hole,
    perturbation,
    staggered_quantum_link,
    superlattice,
    to_qubits,
)
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import (
    build_graph as schwinger_graph,
)
from tests.conftest import (
    WINDOW_INTERACTION,
    WINDOW_MASS,
    bosonic_space,
    forward_map,
    window_point,
)

TOLERANCE = 1e-10


def _effective_operator(matter_sites: int, effective: dict[str, float]) -> Array:
    """``P H_eff P`` in the bosonic space."""
    space = bosonic_space(matter_sites)
    projector = local_subspace_projector(local_occupation_subspace(matter_sites), space)
    return sandwich(
        build_operator(effective_bosonic(matter_sites).hamiltonian, space, effective),
        projector,
    )


def test_both_hamiltonians_are_hermitian_and_gauge_invariant() -> None:
    """H_BHM and H_eff are Hermitian and H_eff commutes with every Gauss operator."""
    matter_sites = 3
    knobs = window_point()
    effective_values = forward_map(knobs)
    effective = {
        P.MASS_L2C: effective_values[P.MASS_L2C],
        P.COUPLING_L2C: effective_values[P.COUPLING_L2C],
    }
    space = bosonic_space(matter_sites)
    l3a = superlattice(matter_sites)
    l2c = effective_bosonic(matter_sites)

    bose_hubbard = build_operator(l3a.hamiltonian, space, knobs)
    effective_operator = build_operator(l2c.hamiltonian, space, effective)
    assert hermiticity_defect(bose_hubbard) < TOLERANCE
    assert hermiticity_defect(effective_operator) < TOLERANCE

    for constraint in l2c.constraint_operators:
        gauss = build_operator(constraint.operator, space, {})
        assert commutator_spectral_norm(effective_operator, gauss) < TOLERANCE


def test_the_exact_step_l2b_to_l2c_preserves_the_spectrum_on_the_subspace() -> None:
    """L2b and ``P H_eff P`` restricted to the occupation subspace have the same spectrum."""
    matter_sites = 3
    effective_values = forward_map(window_point())
    mass = effective_values[P.MASS_L2C]
    coupling = effective_values[P.COUPLING_L2C]

    l2b = homogeneous_quantum_link(matter_sites)
    fermionic_space = HilbertSpace.of(l2b.structure)
    fermionic = build_operator(
        l2b.hamiltonian, fermionic_space, {P.MASS_L2B: mass, P.COUPLING_L2B: coupling}
    )

    space = bosonic_space(matter_sites)
    projector = local_subspace_projector(local_occupation_subspace(matter_sites), space)
    indices = subspace_indices(projector)
    assert len(indices) == 2 ** (2 * matter_sites - 1)

    bosonic = build_operator(
        effective_bosonic(matter_sites).hamiltonian,
        space,
        {P.MASS_L2C: mass, P.COUPLING_L2C: coupling},
    )
    restricted = restrict(bosonic, indices)

    fermionic_spectrum, _ = eigensystem(fermionic)
    restricted_spectrum, _ = eigensystem(restricted)
    assert float(jnp.max(jnp.abs(fermionic_spectrum - restricted_spectrum))) < TOLERANCE


@pytest.mark.parametrize("matter_sites", [2, 3])
def test_the_perturbative_deviation_decreases_monotonically(matter_sites: int) -> None:
    """The L2c/L3a deviation and the subspace leakage fall monotonically as J/U halves."""
    ratios = (1 / 10, 1 / 20, 1 / 40, 1 / 80)
    space = bosonic_space(matter_sites)
    projector = local_subspace_projector(local_occupation_subspace(matter_sites), space)
    initial = space.basis_state(canonical_state_configuration(matter_sites))
    observable = build_operator(matter_occupation_observable(matter_sites), space, {})
    l3a = superlattice(matter_sites)

    deviations: list[float] = []
    leakages: list[float] = []
    for ratio in ratios:
        depth = WINDOW_MASS + WINDOW_INTERACTION / 2
        knobs = {
            P.TUNNELLING: ratio * WINDOW_INTERACTION,
            P.INTERACTION: WINDOW_INTERACTION,
            P.SUPERLATTICE: depth,
            P.TILT: 0.05 * depth,
        }
        values = forward_map(knobs)
        effective = {
            P.MASS_L2C: values[P.MASS_L2C],
            P.COUPLING_L2C: values[P.COUPLING_L2C],
        }
        coupling = effective[P.COUPLING_L2C]
        bose_hubbard = build_operator(l3a.hamiltonian, space, knobs)
        effective_operator = _effective_operator(matter_sites, effective)
        times = [5.0 / coupling * index / 20 for index in range(21)]
        bose_states = evolve_state(bose_hubbard, initial, times)
        effective_states = evolve_state(effective_operator, initial, times)
        deviations.append(
            float(
                jnp.max(
                    jnp.abs(
                        expectation(observable, bose_states)
                        - expectation(observable, effective_states)
                    )
                )
            )
        )
        leakages.append(float(jnp.max(1.0 - expectation(projector, bose_states))))

    assert all(later < earlier for earlier, later in pairwise(deviations)), (
        f"deviations not monotone: {deviations}"
    )
    assert all(later < earlier for earlier, later in pairwise(leakages)), (
        f"leakage not monotone: {leakages}"
    )
    # Second-order scaling: more than a factor of two per halving over three halvings.
    assert deviations[0] / deviations[-1] > 8.0


@pytest.mark.parametrize("coupling", [0.83, 0.004525])
def test_the_effective_model_is_exact_only_after_projection(coupling: float) -> None:
    """``Q H_eff P`` has norm ``sqrt(2) * kappa`` while ``Q (P H_eff P)`` vanishes."""
    matter_sites = 3
    space = bosonic_space(matter_sites)
    projector = local_subspace_projector(local_occupation_subspace(matter_sites), space)
    complement = space.identity() - projector
    operator = build_operator(
        effective_bosonic(matter_sites).hamiltonian,
        space,
        {P.MASS_L2C: 0.41, P.COUPLING_L2C: coupling},
    )
    leakage = spectral_norm(complement @ operator @ projector)
    assert leakage == pytest.approx(math.sqrt(2.0) * coupling, rel=1e-9)
    assert spectral_norm(complement @ sandwich(operator, projector)) < TOLERANCE


def test_the_round_trip_recovers_the_target_parameters() -> None:
    """(m, kappa) -> (J, U, delta, Delta) -> (m, kappa) recovers the targets to 1e-10."""
    step = perturbation()
    targets = {P.MASS_L2C: WINDOW_MASS, P.COUPLING_L2C: 0.004525}
    # Pin the energy scale and the tilt, then invert.
    pinned = {
        **targets,
        P.INTERACTION: WINDOW_INTERACTION,
        P.TILT: 0.048,
    }
    inverse = step.relation.classify(known=frozenset(pinned))
    assert inverse.kind is RelationKind.CLOSED_FORM
    assert inverse.elimination is not None
    knobs = step.relation.evaluate_closed_form(inverse.elimination, pinned)

    forward = step.relation.classify(
        known=frozenset({P.TUNNELLING, P.INTERACTION, P.SUPERLATTICE, P.TILT})
    )
    assert forward.kind is RelationKind.CLOSED_FORM
    assert forward.elimination is not None
    recovered = step.relation.evaluate_closed_form(
        forward.elimination,
        {name: knobs[name] for name in (P.TUNNELLING, P.INTERACTION, P.SUPERLATTICE, P.TILT)},
    )
    for name, target in targets.items():
        assert recovered[name] == pytest.approx(target, rel=1e-10)


@pytest.mark.parametrize("matter_sites", [2, 3])
def test_exact_diagonalisation_runs_and_reports_the_low_lying_spectrum(
    matter_sites: int,
) -> None:
    """Exact diagonalisation of L2b returns a sorted low-lying spectrum."""
    l2b = homogeneous_quantum_link(matter_sites)
    space = HilbertSpace.of(l2b.structure)
    operator = build_operator(l2b.hamiltonian, space, {P.MASS_L2B: 0.3, P.COUPLING_L2B: 0.83})
    spectrum = low_lying_spectrum(operator, count=min(8, space.dimension))
    assert len(spectrum) == min(8, space.dimension)
    assert bool(jnp.all(spectrum[:-1] <= spectrum[1:] + 1e-12))


def test_large_negative_mass_fills_the_matter_sites() -> None:
    """At large negative mass the ground state has every matter site occupied."""
    matter_sites = 3
    l2b = homogeneous_quantum_link(matter_sites)
    space = HilbertSpace.of(l2b.structure)
    observable = build_operator(matter_occupation_observable(matter_sites), space, {})

    operator = build_operator(l2b.hamiltonian, space, {P.MASS_L2B: -40.0, P.COUPLING_L2B: 0.83})
    _, vectors = eigensystem(operator)
    ground = vectors[:, 0]
    assert float(expectation(observable, ground)) == pytest.approx(1.0, abs=1e-3)


def test_large_positive_mass_empties_the_matter_sites() -> None:
    """At large positive mass the ground state has no matter site occupied."""
    matter_sites = 3
    l2b = homogeneous_quantum_link(matter_sites)
    space = HilbertSpace.of(l2b.structure)
    observable = build_operator(matter_occupation_observable(matter_sites), space, {})

    operator = build_operator(l2b.hamiltonian, space, {P.MASS_L2B: 40.0, P.COUPLING_L2B: 0.83})
    _, vectors = eigensystem(operator)
    ground = vectors[:, 0]
    assert float(expectation(observable, ground)) == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Exactness, checked symbolically
# ---------------------------------------------------------------------------


def _exact_hamiltonian_edges(graph: ModelGraph) -> list[tuple[str, Transformation, str]]:
    """Every exact edge between two Hamiltonian models, as (source, step, target)."""
    return [
        (edge.source, edge.transformation, edge.target)
        for edge in graph.edges
        if edge.transformation.exactness is Exactness.EXACT
        and isinstance(graph.node(edge.source), HamiltonianModel)
        and isinstance(graph.node(edge.target), HamiltonianModel)
    ]


@pytest.mark.parametrize(
    ("build", "sizes"),
    [
        (schwinger_graph, (3, 4)),
        (heisenberg_graph, (3, 4, 5)),
        (ising_graph, (3, 4)),
    ],
    ids=["schwinger", "heisenberg", "ising"],
)
def test_every_exact_step_is_the_symbolic_image_of_its_source(
    build: Callable[[int], UseCaseGraph], sizes: tuple[int, ...]
) -> None:
    """Each exact step has a symbolic image, and the library target equals it exactly."""
    for sites in sizes:
        graph = build(sites).graph
        edges = _exact_hamiltonian_edges(graph)
        assert edges, "the use case declares no exact step"
        for source_name, step, target_name in edges:
            source = graph.node(source_name)
            target = graph.node(target_name)
            assert isinstance(source, HamiltonianModel)
            assert isinstance(target, HamiltonianModel)
            image = step.image(source)
            assert image is not None, f"{step.name} has no symbolic image"
            defect = step.exactness_defect(source, target)
            assert defect is not None
            assert defect.terms == (), f"{step.name} at N={sites}:\n{defect}"


def test_the_jordan_wigner_image_reproduces_the_realised_matrix() -> None:
    """The symbolic qubit image and the fermionic source realise to the same matrix."""
    matter_sites = 3
    source = homogeneous_quantum_link(matter_sites).bind(
        **{P.MASS_L2B: WINDOW_MASS, P.COUPLING_L2B: 0.83}
    )
    assert isinstance(source, HamiltonianModel)
    image = fermions_to_qubits_image(source)
    qubit_structure = to_qubits().target_structure(source.structure)
    qubit_space = HilbertSpace.of(qubit_structure)
    fermion_space = HilbertSpace.of(source.structure)
    values = source.binding.as_dict()
    realised_image = build_operator(image, qubit_space, values)
    realised_source = build_operator(source.hamiltonian, fermion_space, values)
    assert float(jnp.max(jnp.abs(realised_image - realised_source))) < TOLERANCE


def test_a_wrong_exact_target_is_refused_when_the_step_is_applied() -> None:
    """A library target that is not the image of its source raises on application."""
    step = particle_hole()

    @dataclass(frozen=True)
    class WrongTarget(ParticleHoleTransformation):
        def build_target(self, matter_sites: int) -> HamiltonianModel:
            target = super().build_target(matter_sites)
            extra = word(Symbol(P.MASS_L2B), number(0))
            return replace(target, hamiltonian=target.hamiltonian + extra)

    wrong = WrongTarget(**{name: getattr(step, name) for name in step.__dataclass_fields__})
    source = staggered_quantum_link(3).bind(**{P.MASS_L2A: 0.3, P.COUPLING_L2A: 0.7})
    assert isinstance(step.apply(source), HamiltonianModel)
    with pytest.raises(ExactnessError, match="declared EXACT") as caught:
        wrong.apply(source)
    assert len(caught.value.defect.terms) == 1
    assert caught.value.defect.terms[0].operators == (number(0),)


def test_an_omitted_constant_is_recorded_rather_than_refused() -> None:
    """An exact step whose target drops the source's constant records it as dropped."""
    source = particle_hole().apply(
        staggered_quantum_link(3).bind(**{P.MASS_L2A: 0.3, P.COUPLING_L2A: 0.7})
    )
    assert isinstance(source, HamiltonianModel)
    assert source.hamiltonian.constant_part().evaluate_real(source.binding.as_dict()) == (
        pytest.approx(-0.3)
    )
    dropping = hardcore_boson_encoding(L2B, L2C, retain_staggering_constant=False)
    encoded = dropping.apply(source)
    assert isinstance(encoded, HamiltonianModel)
    assert len(encoded.dropped_constants) == 1
    assert encoded.total_dropped_constant() == pytest.approx(-0.3)
    assert "omitted" in encoded.dropped_constants[0].description
