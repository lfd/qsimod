"""Ising use case: one hardware node reached from two theories.

Reference values are from Simon et al., Nature 472, 307-312 (2011), Methods.
"""

from __future__ import annotations

import numpy as np
import pytest
from examples.simon2011 import (
    PAPER_CRITICAL_DETUNING_IN_TUNNELLINGS,
    PAPER_CRITICAL_SLOPE,
    REQUEST_TRANSVERSE_FIELD,
    Bench,
    ising_environment,
    solve_gauge,
    verdicts,
)

from qsimod.levels import AbstractionLevel
from qsimod.parameters import ConstraintOrigin
from qsimod.realise import spacing_deviation
from qsimod.relations import RelationKind
from qsimod.solving import SolveResult, SolveStatus, realise_parameters
from qsimod.structure import StructureTypeError
from qsimod.transform import ApproximationKind, Exactness
from qsimod.transformations import DipoleReduction
from qsimod.usecases import ising, schwinger
from qsimod.usecases.heisenberg import spin_chain as xxz_chain
from qsimod.usecases.ising import KNOBS
from qsimod.usecases.ising import ParameterNames as P

#: The request used throughout: unit coupling, small transverse field, on the transition line.
COUPLING = 1.0
CRITICAL_FIELD = 1.0 - PAPER_CRITICAL_SLOPE * REQUEST_TRANSVERSE_FIELD


def solve(longitudinal: float = CRITICAL_FIELD) -> SolveResult:
    """Solve the Ising device for a point of the ``(hz, hx)`` plane."""
    limits = ising.device_limits()
    return realise_parameters(
        ising.build_graph(3, admissible_set=limits).device,
        targets={
            P.COUPLING_ISING_MAGNET: COUPLING,
            P.TRANSVERSE_FIELD: REQUEST_TRANSVERSE_FIELD,
            P.LONGITUDINAL_FIELD: longitudinal,
        },
        unknowns=list(KNOBS),
        admissible_set=limits,
        initial=ising.device_start(COUPLING),
    )


# ---------------------------------------------------------------------------
# The shared node
# ---------------------------------------------------------------------------


def test_the_device_node_is_the_running_examples_own_object() -> None:
    """The Ising device is the same node, structure and parameter set as the Schwinger one."""
    assert ising.DEVICE is schwinger.BOSE_HUBBARD
    assert ising.DEVICE_TARGET == schwinger.ANALOGUE_TARGET
    assert KNOBS == (
        schwinger.ParameterNames.TUNNELLING,
        schwinger.ParameterNames.INTERACTION,
        schwinger.ParameterNames.SUPERLATTICE,
        schwinger.ParameterNames.TILT,
    )
    limits = ising.device_limits()
    mine, theirs = ising.lattice(3, limits), schwinger.superlattice(3, limits)
    assert mine.name == theirs.name
    assert mine.structure == theirs.structure
    assert mine.parameters.names == theirs.parameters.names


def test_the_shared_graph_reaches_one_device_from_two_theories() -> None:
    """The hardware node has two incoming edges, from both theories, and is a single node."""
    graph = ising.shared_graph(3)
    incoming = graph.incoming(ising.DEVICE_TARGET)
    assert len(incoming) == 2
    assert {edge.source for edge in incoming} == {"effective_bosonic", "ising_chain"}

    assert graph.terminal_targets("ising_magnet") == ("bose_hubbard",)
    assert "bose_hubbard" in graph.terminal_targets("lattice_qed")
    assert graph.node("bose_hubbard") is graph.node(ising.DEVICE_TARGET)


def test_the_spin_count_follows_the_register_rather_than_being_assumed() -> None:
    """``2N-1`` lattice sites carry ``2N-2`` spins, as ``target_sites`` declares."""
    step = ising.dipoles()
    for matter_sites in (2, 3, 4, 7):
        spins = ising.spins_for(matter_sites)
        assert spins == 2 * matter_sites - 2
        assert DipoleReduction.lattice_sites(spins) == matter_sites
        assert step.target_sites(spins) == matter_sites
        assert ising.fields().target_sites(spins) == spins
    with pytest.raises(ValueError, match="does not sit on the bonds"):
        DipoleReduction.lattice_sites(5)

    # A four-spin chain lands on a five-site register.
    device = step.apply(ising.ising_chain(3))
    assert device.structure.lattice.matter_sites == 3
    assert len(device.structure.sites) == 5


def test_a_derivation_constraint_is_droppable_where_an_apparatus_one_is_not() -> None:
    """Schwinger box constraints are derivation-origin and drop out of ``apparatus_only``."""
    box = schwinger.device_limits()
    assert box.constraints
    assert not box.constraints_from(ConstraintOrigin.APPARATUS)
    assert len(box.constraints_from(ConstraintOrigin.DERIVATION)) == len(box.constraints)

    apparatus = box.apparatus_only()
    assert not apparatus.constraints
    assert apparatus.bounds == box.bounds
    assert ising.device_limits().constraints == ()

    # The dropped constraints remain declared as regime conditions.
    names = {condition.name for condition in schwinger.validity().conditions}
    assert {"Delta << delta", "Delta << U"} <= names


def test_an_ising_step_refuses_a_chain_that_conserves_magnetisation() -> None:
    """The dipole step refuses a chain that declares a magnetisation symmetry."""
    assert not ising.magnet(3).structure.symmetries
    assert xxz_chain(4).structure.symmetry("magnetisation") is not None
    with pytest.raises(StructureTypeError):
        ising.dipoles().apply(xxz_chain(4))


def test_the_levels_are_the_ones_the_pipeline_claims() -> None:
    """Levels are application, intermediate, hardware; the dipole reduction is regime-limited."""
    graph = ising.build_graph(3)
    levels = graph.levels()
    assert levels["ising_magnet"] is AbstractionLevel.APPLICATION
    assert levels["ising_chain"] is AbstractionLevel.INTERMEDIATE
    assert levels[ising.DEVICE_TARGET] is AbstractionLevel.HARDWARE

    steps = list(graph.device)
    assert steps[0].exactness is Exactness.EXACT
    assert steps[1].exactness is Exactness.APPROXIMATE
    assert steps[1].approximation_kind is ApproximationKind.REGIME_LIMITED


# ---------------------------------------------------------------------------
# The mapping, against the source
# ---------------------------------------------------------------------------


def test_the_field_resolution_is_invertible_and_leaves_the_hamiltonian_alone() -> None:
    """The field resolution is a closed-form reparametrisation in both directions."""
    step = ising.fields()
    assert step.relation.is_invertible(
        {P.COUPLING_ISING_MAGNET, P.LONGITUDINAL_FIELD, P.TRANSVERSE_FIELD},
        {P.COUPLING_ISING_CHAIN, P.LONGITUDINAL, P.TRANSVERSE},
    )
    assert step.forward_relation_kind is RelationKind.CLOSED_FORM
    assert step.inverse_relation_kind is RelationKind.CLOSED_FORM


def test_the_forward_maps_are_the_sources_own() -> None:
    """``Jz = U``, ``Gamma = 2 sqrt(2) J`` and ``B = Jz - (Delta - U)``."""
    knobs = {P.TUNNELLING: 0.02, P.INTERACTION: 1.0, P.SUPERLATTICE: 0.001, P.TILT: 1.05}
    assert ising.coupling_map().evaluate_real(knobs) == pytest.approx(1.0)
    assert ising.transverse_map().evaluate_real(knobs) == pytest.approx(2 * np.sqrt(2) * 0.02)
    assert ising.detuning_map().evaluate_real(knobs) == pytest.approx(0.05)
    environment = ising_environment(knobs)
    assert environment[P.LONGITUDINAL] == pytest.approx(1.0 - 0.05)
    # hz = 1 - (Delta - U) / Jz, the paper's dimensionless form.
    assert environment[P.LONGITUDINAL] / environment[P.COUPLING_ISING_CHAIN] == pytest.approx(0.95)


def test_the_transition_line_comes_out_in_both_of_the_papers_forms() -> None:
    """Requesting ``hz = 1 - 0.66 hx`` reads back as ``E = U + 1.85 t``."""
    result = solve()
    assert result.status is SolveStatus.EXACT_SOLUTION
    knobs = result.subset(list(KNOBS))
    detuning = ising.detuning_map().evaluate_real(knobs) / knobs[P.TUNNELLING]
    assert detuning == pytest.approx(PAPER_CRITICAL_DETUNING_IN_TUNNELLINGS, rel=0.02)


def test_the_end_spin_field_is_half_the_coupling_and_is_not_small() -> None:
    """The end-spin field is half the coupling and exceeds the transverse field."""
    knobs = {P.TUNNELLING: 0.02, P.INTERACTION: 1.0, P.SUPERLATTICE: 0.001, P.TILT: 1.05}
    boundary = ising.boundary_field().evaluate_real(knobs)
    assert boundary == pytest.approx(0.5 * ising.coupling_map().evaluate_real(knobs))
    assert boundary > ising.transverse_map().evaluate_real(knobs)


# ---------------------------------------------------------------------------
# The solve, and the two windows
# ---------------------------------------------------------------------------


def test_the_superlattice_is_the_requests_one_residual_freedom() -> None:
    """The superlattice is absent from the Ising relation and is driven towards zero."""
    step = ising.dipoles()
    assert step.forward_relation_kind is RelationKind.CLOSED_FORM
    assert P.SUPERLATTICE not in step.relation.parameters()

    knobs = solve().subset(list(KNOBS))
    transverse = ising.transverse_map().evaluate_real(knobs)
    assert 0.0 < knobs[P.SUPERLATTICE] <= 0.1 * transverse


def test_the_lattices_box_admits_a_request_the_running_examples_box_refuses() -> None:
    """The Schwinger box refuses the Ising request; the lattice's own box admits it."""
    tight = schwinger.device_limits()
    assert len(tight.constraints) == 2
    assert not ising.device_limits().constraints

    refused = realise_parameters(
        ising.build_graph(3, admissible_set=tight).device,
        targets={
            P.COUPLING_ISING_MAGNET: COUPLING,
            P.TRANSVERSE_FIELD: REQUEST_TRANSVERSE_FIELD,
            P.LONGITUDINAL_FIELD: CRITICAL_FIELD,
        },
        unknowns=list(KNOBS),
        admissible_set=tight,
        initial=ising.device_start(COUPLING),
    )
    assert not refused.status.is_success
    assert solve().status is SolveStatus.EXACT_SOLUTION


def test_the_gauge_theorys_answer_is_the_same_in_the_lattices_box() -> None:
    """The gauge-theory solve gives the same point in either box."""
    own = realise_parameters(
        schwinger.build_graph(3, admissible_set=schwinger.device_limits()).analogue,
        targets={
            schwinger.ParameterNames.MASS_LATTICE_QED: 0.0,
            schwinger.ParameterNames.COUPLING_QUANTUM_LINK_STAGGERED: 0.0045,
            schwinger.ParameterNames.ELECTRIC_GAP: 0.5,
        },
        unknowns=list(KNOBS),
        admissible_set=schwinger.device_limits(),
    )
    wide = solve_gauge(ising.device_limits())
    assert own.status is SolveStatus.EXACT_SOLUTION
    assert wide.status is SolveStatus.EXACT_SOLUTION
    for knob in KNOBS:
        assert wide.point[knob] == pytest.approx(own.point[knob], rel=1e-6), knob


def test_no_setting_of_this_lattice_is_both_theories_at_once() -> None:
    """Each theory's solution is out of regime for the other."""
    ising_knobs = solve().subset(list(KNOBS))
    gauge_knobs = solve_gauge(ising.device_limits()).subset(list(KNOBS))

    ising_here, gauge_here = verdicts(ising_knobs)
    assert ising_here.valid
    assert ising_here.margin > 0.0
    assert not gauge_here.valid
    assert gauge_here.weakest == "Delta << delta"

    ising_there, gauge_there = verdicts(gauge_knobs)
    assert gauge_there.valid
    assert gauge_there.margin > 0.0
    assert not ising_there.valid
    assert ising_there.weakest == "delta << Gamma"


# ---------------------------------------------------------------------------
# Semantic preservation
# ---------------------------------------------------------------------------


def test_the_dipole_manifold_is_the_ising_chain_once_the_end_field_is_restored() -> None:
    """The dipole-manifold spectrum matches the Ising chain's once the end field is restored."""
    limits = ising.device_limits()
    knobs = solve().subset(list(KNOBS))
    bench = Bench.of(3, limits)
    levels, weight = bench.device_spectrum(knobs, limits)
    transverse = ising.transverse_map().evaluate_real(knobs)

    assert weight > 0.99
    assert len(bench.manifold) == len(bench.constrained) == 8  # Fibonacci over four bonds

    def gap(*, end_field: bool) -> float:
        other = bench.magnet_spectrum(knobs, end_field=end_field)
        return spacing_deviation(levels, other, transverse)

    assert gap(end_field=True) < 0.05
    assert gap(end_field=False) > 100 * gap(end_field=True)
