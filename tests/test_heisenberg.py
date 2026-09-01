"""Heisenberg use case: structural types, exact steps, the superexchange window and the solve.

Reference values are from Jepsen et al., Nature 588, 403-407 (2020), Methods table.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from examples.heisenberg_end_to_end import (
    PAPER_ROWS,
    Bench,
    PaperRow,
    experiment_knobs,
    reference_hopping,
)

from qsimod.artifact import ArtifactKind, as_hamiltonian
from qsimod.levels import AbstractionLevel
from qsimod.models.magnetism import coordination_numbers, mott_manifold_operators
from qsimod.pipeline import CompositionError, Pipeline
from qsimod.realise import HilbertSpace, RealisationRequest, SectorBasis, build_operator
from qsimod.relations import RelationKind
from qsimod.solving import SolveStatus, realise_parameters
from qsimod.structure import StructureTypeError
from qsimod.transform import ApproximationKind, Exactness
from qsimod.usecases.heisenberg import (
    KNOBS,
    anisotropy,
    build_graph,
    device_limits,
    device_start,
    fermion_chain,
    field_map,
    jordan_wigner,
    lattice,
    longitudinal_map,
    magnet,
    magnetisation_observable,
    spin_chain,
    superexchange,
    transverse_map,
    validity,
    weighted_magnetisation_term,
)
from qsimod.usecases.heisenberg import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import staggered_quantum_link
from qsimod.usecases.schwinger import superlattice as schwinger_device

#: The paper's row nearest the isotropic point, inside the superexchange window.
REFERENCE_ROW = PAPER_ROWS[3]

#: Tolerance for matrix equality of exact steps.
EXACT_TOLERANCE = 1e-12


@pytest.fixture
def knobs() -> dict[str, float]:
    """The experiment's four knobs at the reference row, in rad/ms."""
    return experiment_knobs(REFERENCE_ROW, reference_hopping())


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------


def test_the_graph_has_two_leaves_and_only_one_of_them_is_hardware() -> None:
    """The graph has two leaves, of which only ``M3`` is at hardware level."""
    graph = build_graph(4)
    assert graph.targets() == ("M2b", "M3")
    levels = graph.levels()
    assert levels["M1"] is AbstractionLevel.APPLICATION
    assert levels["M2a"] is levels["M2b"] is AbstractionLevel.INTERMEDIATE
    assert levels["M3"] is AbstractionLevel.HARDWARE

    branches = {branch.target: branch for branch in graph.graph.branches("M1")}
    assert branches["M2b"].exactness is Exactness.EXACT
    assert branches["M3"].exactness is Exactness.APPROXIMATE
    assert branches["M3"].approximation_kinds == frozenset({ApproximationKind.REGIME_LIMITED})


def test_a_step_refuses_a_model_from_the_other_use_case() -> None:
    """Magnetism steps refuse Schwinger models and name the mismatched aspects."""
    with pytest.raises(StructureTypeError) as error:
        jordan_wigner().apply(staggered_quantum_link(3))
    aspects = {mismatch.aspect for mismatch in error.value.mismatches}
    assert "dof[spins]" in aspects
    assert "symmetry[magnetisation]" in aspects

    with pytest.raises(StructureTypeError):
        superexchange().apply(schwinger_device(3))


def test_the_two_use_cases_pipelines_do_not_compose() -> None:
    """Pipelines mixing the two use cases fail composition from the patterns alone."""
    with pytest.raises(CompositionError):
        Pipeline.of([anisotropy(), superexchange(), jordan_wigner()])


def test_applying_the_perturbative_step_returns_an_unbound_device() -> None:
    """Applying the perturbative step yields a device whose knobs are all free."""
    bound = magnet(3).bind(**{P.TRANSVERSE_M1: 0.5, P.ANISOTROPY: 1.0})
    device = superexchange().apply(anisotropy().apply(bound))
    assert device.kind.name == "HAMILTONIAN"
    assert not device.is_fully_bound
    assert set(device.free_parameters) == set(KNOBS)


def test_both_leaves_are_hamiltonians_here_unlike_the_running_examples() -> None:
    """Both leaves are Hamiltonians and differ only in abstraction level."""
    graph = build_graph(3)
    for target in graph.targets():
        assert as_hamiltonian(graph.graph.node(target)).kind is ArtifactKind.HAMILTONIAN
    assert graph.levels()["M2b"] is not graph.levels()["M3"]


# ---------------------------------------------------------------------------
# Exactness
# ---------------------------------------------------------------------------


def test_the_anisotropy_resolution_is_invertible_and_leaves_the_hamiltonian_alone() -> None:
    """Step (a) is a closed-form reparametrisation realising the same matrix at both levels."""
    step = anisotropy()
    assert step.relation.is_invertible(
        {P.TRANSVERSE_M1, P.ANISOTROPY}, {P.TRANSVERSE_M2A, P.LONGITUDINAL_M2A}
    )
    assert step.forward_relation_kind is RelationKind.CLOSED_FORM
    assert step.inverse_relation_kind is RelationKind.CLOSED_FORM

    transverse, delta = 0.5, 0.973
    theory, chain = magnet(5), spin_chain(5)
    left = build_operator(
        theory.hamiltonian,
        HilbertSpace.of(theory.structure),
        {P.TRANSVERSE_M1: transverse, P.ANISOTROPY: delta},
    )
    right = build_operator(
        chain.hamiltonian,
        HilbertSpace.of(chain.structure),
        {P.TRANSVERSE_M2A: transverse, P.LONGITUDINAL_M2A: delta * transverse},
    )
    assert float(np.max(np.abs(np.asarray(left) - np.asarray(right)))) < EXACT_TOLERANCE


@pytest.mark.parametrize("longitudinal", [0.0, 0.35, -0.6])
def test_the_jordan_wigner_step_is_exact_as_matrices(longitudinal: float) -> None:
    """Step (b) realises the spin chain and its fermion image to the same matrix."""
    transverse = 0.5
    chain, fermions = spin_chain(6), fermion_chain(6)
    left = build_operator(
        chain.hamiltonian,
        HilbertSpace.of(chain.structure),
        {P.TRANSVERSE_M2A: transverse, P.LONGITUDINAL_M2A: longitudinal},
    )
    right = build_operator(
        fermions.hamiltonian,
        HilbertSpace.of(fermions.structure),
        {P.TRANSVERSE_M2B: transverse, P.LONGITUDINAL_M2B: longitudinal},
    )
    assert float(np.max(np.abs(np.asarray(left) - np.asarray(right)))) < EXACT_TOLERANCE


def test_the_free_fermion_band_is_the_one_the_literature_quotes() -> None:
    """At zero anisotropy the fermion chain has the free band ``-Jxy cos(qa)``."""
    sites, transverse = 8, 0.5
    model = fermion_chain(sites)
    space = HilbertSpace.of(model.structure)
    operator = np.asarray(
        build_operator(
            model.hamiltonian,
            space,
            {P.TRANSVERSE_M2B: transverse, P.LONGITUDINAL_M2B: 0.0},
        )
    )
    single = [i for i in range(space.dimension) if sum(space.configuration(i)) == 1]
    measured = np.sort(np.real(np.linalg.eigvalsh(operator[np.ix_(single, single)])))
    momenta = np.pi * np.arange(1, sites + 1) / (sites + 1)
    assert measured == pytest.approx(np.sort(-transverse * np.cos(momenta)), abs=1e-12)
    # Bandwidth of the open chain.
    bandwidth = 2 * transverse * math.cos(math.pi / (sites + 1))
    assert float(measured[-1] - measured[0]) == pytest.approx(bandwidth, rel=1e-12)


# ---------------------------------------------------------------------------
# The superexchange map, against the numbers the paper quotes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", PAPER_ROWS, ids=lambda row: f"Delta={row.anisotropy}")
def test_the_forward_map_reproduces_every_anisotropy_the_paper_quotes(row: PaperRow) -> None:
    """Step (c)'s forward map reproduces each anisotropy in the paper's Methods table."""
    setting = experiment_knobs(row, reference_hopping())
    computed = longitudinal_map().evaluate_real(setting) / transverse_map().evaluate_real(setting)
    assert computed == pytest.approx(row.anisotropy, abs=3e-3)


def test_the_transverse_coupling_is_antiferromagnetic_on_the_devices_branch(
    knobs: dict[str, float],
) -> None:
    """``Jxy = -4 t^2 / U_ud`` is positive for attractive ``U_ud``."""
    assert knobs[P.INTERACTION_MIXED] < 0.0
    assert transverse_map().evaluate_real(knobs) > 0.0
    assert 1.0 / transverse_map().evaluate_real(knobs) == pytest.approx(2.01, rel=1e-9)


def test_the_derived_longitudinal_field_vanishes_exactly_when_the_channels_balance() -> None:
    """The derived longitudinal field vanishes exactly at ``U_uu = U_dd``."""
    balanced = {
        P.HOPPING: 3.0,
        P.INTERACTION_UP: -80.0,
        P.INTERACTION_MIXED: -70.0,
        P.INTERACTION_DOWN: -80.0,
    }
    assert field_map().evaluate_real(balanced) == pytest.approx(0.0, abs=1e-15)
    unbalanced = {**balanced, P.INTERACTION_DOWN: -125.0}
    assert abs(field_map().evaluate_real(unbalanced)) > 0.1 * transverse_map().evaluate_real(
        unbalanced
    )


# ---------------------------------------------------------------------------
# Regime checking
# ---------------------------------------------------------------------------


def test_the_experiments_own_setting_is_inside_the_declared_window(
    knobs: dict[str, float],
) -> None:
    """The published operating point is valid, with ``t << U`` the weakest condition."""
    effective = {
        P.TRANSVERSE_M2A: transverse_map().evaluate_real(knobs),
        P.LONGITUDINAL_M2A: longitudinal_map().evaluate_real(knobs),
    }
    report = validity().report({**knobs, **effective})
    assert report.is_valid, [str(outcome) for outcome in report.failures]
    assert 0.0 < report.weakest_margin < 0.5
    weakest = report.weakest()
    assert weakest is not None
    assert weakest.name.startswith("t << ")


def test_a_vanishing_interaction_channel_is_a_domain_failure_not_a_regime_one(
    knobs: dict[str, float],
) -> None:
    """A vanishing interaction channel is a domain error, not a regime violation."""
    report = validity().report(
        {**knobs, P.INTERACTION_UP: 0.0}
        | {
            P.TRANSVERSE_M2A: 0.5,
            P.LONGITUDINAL_M2A: 0.5,
        }
    )
    assert report.has_domain_error
    assert {outcome.name for outcome in report.domain_errors} == {"pole: U_uu != 0"}


def test_a_large_anisotropy_leaves_the_window_through_the_longitudinal_condition() -> None:
    """A large anisotropy violates ``|Jz| << U_uu``."""
    hopping, mixed = 1.0, -100.0
    intra = -3.0  # small on the scale of U_ud: a large anisotropy
    knobs = {
        P.HOPPING: hopping,
        P.INTERACTION_UP: intra,
        P.INTERACTION_MIXED: mixed,
        P.INTERACTION_DOWN: mixed,
    }
    effective = {
        P.TRANSVERSE_M2A: transverse_map().evaluate_real(knobs),
        P.LONGITUDINAL_M2A: longitudinal_map().evaluate_real(knobs),
    }
    report = validity().report({**knobs, **effective})
    assert not report.is_valid
    assert "|Jz| << U_uu" in {outcome.name for outcome in report.regime_violations}


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def test_the_composite_relation_is_under_determined_by_two() -> None:
    """The composite relation has four equations in six unknowns."""
    classification = build_graph(3).device.classify_relation(
        frozenset({P.TRANSVERSE_M1, P.ANISOTROPY})
    )
    assert classification.kind is RelationKind.UNDER_DETERMINED
    assert classification.degrees_of_freedom == 2
    assert set(classification.unknowns) >= set(KNOBS)


@pytest.mark.parametrize("target", [-1.79, -1.02, 0.0, 0.973, 1.256])
def test_every_anisotropy_the_paper_reports_is_reachable(target: float) -> None:
    """Each anisotropy the paper reports solves exactly inside the window."""
    result = realise_parameters(
        build_graph(3).device,
        targets={P.TRANSVERSE_M1: 0.4975, P.ANISOTROPY: target},
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        initial=device_start(target),
    )
    assert result.status is SolveStatus.EXACT_SOLUTION
    assert result.max_residual < 1e-9
    assert result.validity.is_valid
    assert not result.violations

    point = result.subset(list(KNOBS))
    assert longitudinal_map().evaluate_real(point) / transverse_map().evaluate_real(
        point
    ) == pytest.approx(target, rel=1e-9)
    if target < -1.0:
        assert point[P.INTERACTION_DOWN] > 0.0


def test_an_extreme_anisotropy_runs_out_of_mott_gap() -> None:
    """An extreme anisotropy is ``UNSOLVED``: the window ``t << U`` binds, then the Mott lobe.

    With the regime conditions enforced the search stops on the perturbative window, both
    interaction channels at ``|U|/t = 10`` with the relation unsatisfied.  Enforcing only the
    apparatus, the wider Mott-lobe limit ``|U|/t >= 3.4`` is what is named.
    """
    result = realise_parameters(
        build_graph(3).device,
        targets={P.TRANSVERSE_M1: 0.4975, P.ANISOTROPY: 60.0},
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        initial=device_start(60.0),
    )
    assert result.status is SolveStatus.UNSOLVED
    assert result.max_residual > 1e-3
    assert not result.violations
    weakest = result.validity.weakest()
    assert weakest is not None
    assert weakest.name.startswith("t << U_")
    assert weakest.margin == pytest.approx(0.0, abs=1e-6)

    apparatus_only = realise_parameters(
        build_graph(3).device,
        targets={P.TRANSVERSE_M1: 0.4975, P.ANISOTROPY: 60.0},
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        initial=device_start(60.0),
        enforce_regime_conditions=False,
    )
    assert apparatus_only.status is SolveStatus.UNSOLVED
    assert any("Mott lobe" in name for name in apparatus_only.binding_constraints)


def test_the_solve_balances_the_channels_and_so_kills_the_dropped_field() -> None:
    """The default objective lands at ``|U_uu| = |U_dd|``, where the derived field vanishes."""
    result = realise_parameters(
        build_graph(3).device,
        targets={P.TRANSVERSE_M1: 0.4975, P.ANISOTROPY: 0.973},
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        initial=device_start(0.973),
    )
    point = result.subset(list(KNOBS))
    assert abs(point[P.INTERACTION_UP]) == pytest.approx(abs(point[P.INTERACTION_DOWN]), rel=1e-6)
    assert field_map().evaluate_real(point) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Semantic preservation across the approximate step
# ---------------------------------------------------------------------------


def test_the_device_conserves_particle_number_so_its_sector_realisation_is_exact() -> None:
    """The device commutes with total particle number but not with the manifold operators."""
    model = lattice(3)
    space = HilbertSpace.of(model.structure, RealisationRequest(boson_cutoff=2))
    knobs = experiment_knobs(REFERENCE_ROW, reference_hopping())
    dense = np.asarray(build_operator(model.hamiltonian, space, knobs))
    number = np.asarray(build_operator(model.constraint_operators[0].operator, space, {}))
    assert float(np.max(np.abs(dense @ number - number @ dense))) < 1e-10

    manifold = np.asarray(build_operator(mott_manifold_operators(3)[1].operator, space, {}))
    assert float(np.max(np.abs(dense @ manifold - manifold @ dense))) > 1e-3


def test_the_devices_manifold_spectrum_is_the_magnets_to_second_order(
    knobs: dict[str, float],
) -> None:
    """The device's manifold spectrum matches the XXZ chain's at the experiment's setting."""
    bench = Bench.of(3)
    outcome = bench.assess("experiment", knobs)
    assert outcome.weight > 0.9
    assert outcome.deviation < 0.05
    assert outcome.dropped > 20 * outcome.deviation


def test_the_wider_declared_margin_is_the_more_faithful_setting() -> None:
    """The setting with the wider declared margin has the smaller spectrum deviation."""
    row = REFERENCE_ROW
    experiment = experiment_knobs(row, reference_hopping())
    result = realise_parameters(
        build_graph(3).device,
        targets={
            P.TRANSVERSE_M1: transverse_map().evaluate_real(experiment),
            P.ANISOTROPY: row.anisotropy,
        },
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        initial=device_start(row.anisotropy),
    )
    assert result.status is SolveStatus.EXACT_SOLUTION

    bench = Bench.of(3)
    solved = bench.assess("framework", result.subset(list(KNOBS)))
    measured = bench.assess("experiment", experiment)
    assert solved.margin > measured.margin
    assert solved.deviation < measured.deviation
    assert solved.weight > measured.weight


# ---------------------------------------------------------------------------
# The shared layout helpers
# ---------------------------------------------------------------------------


def test_the_coordination_numbers_are_the_open_chains() -> None:
    """Coordination numbers of the open chain are ``(1, 2, ..., 2, 1)``."""
    assert coordination_numbers(2) == (1, 1)
    assert coordination_numbers(5) == (1, 2, 2, 2, 1)


def test_the_weighted_magnetisation_term_is_a_sector_constant_plus_two_end_fields() -> None:
    """``sum_j z_j S^z_j = 2 sum_j S^z_j - S^z_0 - S^z_{N-1}`` on the open chain."""
    sites = 5
    model = spin_chain(sites)
    space = HilbertSpace.of(model.structure)
    weighted = np.asarray(build_operator(weighted_magnetisation_term(sites, 1.0), space, {}))
    total = np.asarray(build_operator(weighted_magnetisation_term(sites, 0.0), space, {}))
    assert float(np.max(np.abs(total))) == 0.0

    magnetisation = np.diag(
        [sum(occupations) - sites / 2 for occupations in space.configurations()]
    )
    ends = np.diag(
        [
            configuration[0] + configuration[sites - 1] - 1.0
            for configuration in space.configurations()
        ]
    )
    assert float(np.max(np.abs(weighted - (2 * magnetisation - ends)))) < EXACT_TOLERANCE


def test_the_declared_global_symmetry_is_a_real_conservation_law() -> None:
    """The magnetisation generator commutes with the spin-chain Hamiltonian."""
    sites = 5
    model = spin_chain(sites)
    assert model.structure.symmetry("magnetisation") is not None
    space = HilbertSpace.of(model.structure)
    hamiltonian = np.asarray(
        build_operator(model.hamiltonian, space, {P.TRANSVERSE_M2A: 0.5, P.LONGITUDINAL_M2A: 0.35})
    )
    generator = np.asarray(build_operator(magnetisation_observable(sites), space, {}))
    commutator = hamiltonian @ generator - generator @ hamiltonian
    assert float(np.max(np.abs(commutator))) < EXACT_TOLERANCE


def test_the_unit_filling_sector_is_smaller_than_the_space_it_sits_in() -> None:
    """The unit-filling sector of ``N = 4`` has 266 states out of 6561."""
    model = lattice(4)
    basis = SectorBasis.of(
        model.structure,
        constraints=model.constraint_operators,
        request=RealisationRequest(boson_cutoff=2),
    )
    assert basis.space.dimension == 3**8
    assert basis.dimension == 266
    assert all(sum(configuration) == 4 for configuration in basis.configurations)
