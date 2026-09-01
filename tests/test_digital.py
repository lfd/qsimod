"""Tests for the digital branch and its Suzuki-Trotter product formula."""

from __future__ import annotations

import ast
import importlib
import math
import pkgutil
import time
from pathlib import Path

import jax.numpy as jnp
import pytest

import qsimod
from qsimod.artifact import HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models.gauge import (
    ElectricField,
    gauss_operators,
    matter_gauge_structure,
)
from qsimod.pauli import pauli_sum_from_operator
from qsimod.realise import (
    HilbertSpace,
    build_operator,
    build_pauli_sum,
    commutator_spectral_norm,
    eigensystem,
    evolve_state,
    expectation,
    gauss_violation,
    local_subspace_projector,
    max_abs_deviation,
    propagator,
    spectral_norm,
)
from qsimod.solving import SolveStatus
from qsimod.solving.backends.integer import minimal_resource_setting
from qsimod.solving.stepcount import STEPS_SYMBOL, minimal_steps, resource_candidates
from qsimod.structure import Algebra
from qsimod.symbolic import (
    OperatorSum,
    number,
    sigma_minus,
    sigma_plus,
    word,
)
from qsimod.transform import ApproximationKind
from qsimod.transformations import derived_layers_of, interleaved_layers_of
from qsimod.trotter import (
    SpectralNormEstimator,
    as_product_formula,
    declared_layers,
    error_bound,
    suzuki,
)
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import (
    SchwingerGraph,
    build_graph,
    canonical_state_configuration,
    local_occupation_subspace,
    particle_hole,
    superlattice,
    to_qubits,
    to_qubits_from_staggered,
    trotterisation,
)
from tests.conftest import (
    REFERENCE_COUPLING,
    REFERENCE_MASS,
    REFERENCE_TIME,
    bosonic_space,
    forward_map,
    qubit_model,
    window_point,
)

TOLERANCE = 1e-10

#: The published reference table, for N = 4, kappa = 0.83, m = 0.41, t = 2.0, error
#: measured as the largest absolute entry of the difference.
REFERENCE_TABLE = {
    (1, 4): 1.685e-01,
    (1, 8): 8.336e-02,
    (1, 16): 4.145e-02,
    (1, 32): 2.066e-02,
    (1, 64): 1.032e-02,
    (2, 4): 1.888e-02,
    (2, 8): 4.674e-03,
    (2, 16): 1.166e-03,
    (2, 32): 2.912e-04,
    (2, 64): 7.280e-05,
    (4, 4): 5.241e-05,
    (4, 8): 3.273e-06,
    (4, 16): 2.045e-07,
    (4, 32): 1.278e-08,
    (4, 64): 7.989e-10,
}


# ---------------------------------------------------------------------------
# The pinned Pauli algebra
# ---------------------------------------------------------------------------


def test_the_pauli_expansion_of_the_coupling_is_the_pinned_one() -> None:
    """The coupling expands to ``(1/4)(XXX + XYY - YXY + YYX)``."""
    structure = matter_gauge_structure(2, Algebra.QUBIT, Algebra.QUBIT, name="three qubits")
    coupling = word(1.0, sigma_minus(0), sigma_plus(1), sigma_minus(2)).plus_adjoint()
    pauli = pauli_sum_from_operator(coupling, {}, structure, "coupling")
    assert len(pauli) == 4
    rendered = {str(string): pauli.terms[string].real for string in pauli.strings}
    assert rendered == pytest.approx(
        {
            "X0 X1 X2": 0.25,
            "X0 Y1 Y2": 0.25,
            "Y0 X1 Y2": -0.25,
            "Y0 Y1 X2": 0.25,
        }
    )
    assert all(abs(value.imag) < TOLERANCE for value in pauli.terms.values())


def test_the_four_strings_mutually_commute_and_factorise_exactly() -> None:
    """The four coupling strings mutually commute and ``exp(-i theta Sigma)`` factorises."""
    structure = matter_gauge_structure(2, Algebra.QUBIT, Algebra.QUBIT, name="three qubits")
    coupling = word(1.0, sigma_minus(0), sigma_plus(1), sigma_minus(2)).plus_adjoint()
    pauli = pauli_sum_from_operator(coupling, {}, structure, "coupling")
    assert pauli.internally_commuting()
    strings = pauli.strings
    for index, left in enumerate(strings):
        for right in strings[index + 1 :]:
            # Any two differ in exactly two tensor positions.
            differences = sum(
                1
                for qubit, axis in right.axes
                if left.mapping.get(qubit) is not None and left.mapping[qubit] is not axis
            )
            assert differences == 2

    qubits = 3
    theta = 0.37
    operator = build_pauli_sum(pauli, qubits)
    exact = propagator(operator, theta)
    factorised = jnp.eye(2**qubits, dtype=operator.dtype)
    identity = jnp.eye(2**qubits, dtype=operator.dtype)
    for string in strings:
        angle = theta * pauli.terms[string].real
        matrix = build_pauli_sum(type(pauli).from_terms([(string, 1.0)]), qubits)
        factorised = factorised @ (jnp.cos(angle) * identity - 1j * jnp.sin(angle) * matrix)
    assert max_abs_deviation(factorised, exact) < TOLERANCE


@pytest.mark.parametrize("matter_sites", [3, 4])
def test_every_term_commutes_with_every_gauss_operator(matter_sites: int) -> None:
    """Every Hamiltonian term and every layer commutes with each Gauss operator."""
    model = qubit_model(matter_sites)
    space = HilbertSpace.of(model.structure)
    environment = model.environment()
    gauss = [
        build_operator(constraint.operator, space, environment)
        for constraint in model.constraint_operators
    ]
    for term in model.hamiltonian.terms:
        operator = build_operator(OperatorSum((term,)), space, environment)
        for generator in gauss:
            assert commutator_spectral_norm(operator, generator) < TOLERANCE

    layers = interleaved_layers_of(model)
    for layer in layers.layers:
        operator = build_pauli_sum(layer.operator, len(model.structure.sites))
        for generator in gauss:
            assert commutator_spectral_norm(operator, generator) < TOLERANCE


def _mixed_convention_model(
    matter_sites: int,
    *,
    hopping: bool,
    alternating_link: bool,
) -> HamiltonianModel:
    """A qubit Hamiltonian with mixed hopping and link conventions."""
    coupling = REFERENCE_COUPLING
    mass = REFERENCE_MASS
    hamiltonian = OperatorSum()
    for link in range(matter_sites - 1):
        left, middle, right = 2 * link, 2 * link + 1, 2 * link + 2
        first = sigma_plus(left) if hopping else sigma_minus(left)
        centre = sigma_minus(middle) if (alternating_link and link % 2 == 1) else sigma_plus(middle)
        hamiltonian = (
            hamiltonian + word(coupling / 2, first, centre, sigma_minus(right)).plus_adjoint()
        )
    for site in range(matter_sites):
        factor = (-1) ** site if hopping else 1
        hamiltonian = hamiltonian + word(mass * factor, number(2 * site))
    return HamiltonianModel(
        name="mixed",
        level=AbstractionLevel.INTERMEDIATE,
        structure=matter_gauge_structure(
            matter_sites, Algebra.QUBIT, Algebra.QUBIT, name="mixed convention"
        ),
        hamiltonian=hamiltonian,
        constraint_operators=gauss_operators(matter_sites, field=ElectricField.QUBIT),
    )


@pytest.mark.parametrize(
    ("matter_sites", "hopping", "alternating", "expected"),
    [
        (3, False, False, 0.0),
        (4, False, False, 0.0),
        (3, True, False, 1.0),
        (4, True, False, 1.0),
        (3, True, True, 1.0),
        (4, True, True, math.sqrt(2.0)),
    ],
)
def test_mixing_the_conventions_breaks_gauge_invariance_by_a_pinned_amount(
    matter_sites: int,
    hopping: bool,
    alternating: bool,
    expected: float,
) -> None:
    """Mixing conventions gives ``||[H, G]|| / kappa`` equal to 0, 1 or ``sqrt(2)``."""
    model = _mixed_convention_model(matter_sites, hopping=hopping, alternating_link=alternating)
    space = HilbertSpace.of(model.structure)
    hamiltonian = build_operator(model.hamiltonian, space, {})
    worst = max(
        commutator_spectral_norm(hamiltonian, build_operator(constraint.operator, space, {}))
        for constraint in model.constraint_operators
    )
    assert worst / REFERENCE_COUPLING == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Semantic preservation and the bound
# ---------------------------------------------------------------------------


def _log_log_slope(xs: list[float], ys: list[float]) -> float:
    """The least-squares slope of ``log y`` against ``log x``."""
    log_x = [math.log(value) for value in xs]
    log_y = [math.log(value) for value in ys]
    mean_x = sum(log_x) / len(log_x)
    mean_y = sum(log_y) / len(log_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y, strict=True))
    return numerator / sum((x - mean_x) ** 2 for x in log_x)


@pytest.mark.parametrize("matter_sites", [3, 4])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_the_convergence_exponent_and_the_bound(matter_sites: int, order: int) -> None:
    """The fitted exponent matches the order, and the bound holds at every point."""
    model = qubit_model(matter_sites)
    space = HilbertSpace.of(model.structure)
    exact = propagator(
        build_operator(model.hamiltonian, space, model.environment()), REFERENCE_TIME
    )
    estimator = SpectralNormEstimator(len(model.structure.sites))

    counts = [4, 8, 16, 32, 64]
    largest_entries: list[float] = []
    for count in counts:
        product_formula = as_product_formula(
            trotterisation(time=REFERENCE_TIME, steps=count, order=order).apply(model)
        )
        approximate = product_formula.matrix()
        largest_entries.append(max_abs_deviation(approximate, exact))
        spectral = spectral_norm(approximate - exact)
        bound = product_formula.error_bound(estimator).value
        assert spectral <= bound, (
            f"order {order}, n={count}: measured {spectral:.6e} exceeds bound {bound:.6e}"
        )
        assert largest_entries[-1] <= bound

    slope = _log_log_slope([float(count) for count in counts], largest_entries)
    assert -slope == pytest.approx(order, abs=0.05)


def test_the_published_reference_table_reproduces() -> None:
    """Each entry of the reference table is reproduced to relative precision 1e-3."""
    model = qubit_model(4)
    space = HilbertSpace.of(model.structure)
    exact = propagator(
        build_operator(model.hamiltonian, space, model.environment()), REFERENCE_TIME
    )
    for (order, count), expected in REFERENCE_TABLE.items():
        product_formula = as_product_formula(
            trotterisation(time=REFERENCE_TIME, steps=count, order=order).apply(model)
        )
        measured = max_abs_deviation(product_formula.matrix(), exact)
        assert measured == pytest.approx(expected, rel=1e-3), (order, count)


def test_the_first_order_bound_has_the_stated_slack() -> None:
    """The first-order bound is 2.2x the spectral-norm error and 3.8x the largest entry."""
    model = qubit_model(4)
    space = HilbertSpace.of(model.structure)
    exact = propagator(
        build_operator(model.hamiltonian, space, model.environment()), REFERENCE_TIME
    )
    estimator = SpectralNormEstimator(len(model.structure.sites))
    layers = interleaved_layers_of(model)
    # Layer-pair commutator norms.
    norms = {
        f"{layers.names[i]},{layers.names[j]}": estimator.norm(layers.layer_commutator(i, j))
        for i, j in layers.non_commuting_pairs()
    }
    assert norms["H_M,H_even"] == pytest.approx(0.681, abs=1e-3)
    assert norms["H_M,H_odd"] == pytest.approx(0.340, abs=1e-3)
    assert norms["H_even,H_odd"] == pytest.approx(0.244, abs=1e-3)

    for count in (4, 64):
        product_formula = as_product_formula(
            trotterisation(time=REFERENCE_TIME, steps=count, order=1).apply(model)
        )
        approximate = product_formula.matrix()
        bound = product_formula.error_bound(estimator).value
        spectral = spectral_norm(approximate - exact)
        largest_entry = max_abs_deviation(approximate, exact)
        assert bound / spectral == pytest.approx(2.2, abs=0.05)
        assert bound / largest_entry == pytest.approx(3.8, abs=0.1)


# ---------------------------------------------------------------------------
# Gauge invariance, and the contrast with the analogue branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("order", "steps"), [(1, 3), (2, 7), (4, 5)])
def test_gauge_invariance_is_exact_digitally_and_drifts_analogue(order: int, steps: int) -> None:
    """The digital branch conserves sum <G^2> exactly; the analogue branch drifts."""
    matter_sites = 3
    model = qubit_model(matter_sites)
    space = HilbertSpace.of(model.structure)
    environment = model.environment()
    generators = [
        build_operator(constraint.operator, space, environment)
        for constraint in model.constraint_operators
    ]
    initial = space.basis_state(canonical_state_configuration(matter_sites))
    assert float(gauss_violation(generators, initial)) == pytest.approx(0.5, abs=1e-12)

    product_formula = as_product_formula(
        trotterisation(time=REFERENCE_TIME, steps=steps, order=order).apply(model)
    )
    evolved = product_formula.apply_to_state(initial)
    digital_drift = abs(float(gauss_violation(generators, evolved)) - 0.5)
    assert digital_drift < 1e-12
    assert float(jnp.linalg.norm(evolved)) == pytest.approx(1.0, abs=1e-12)

    # The same diagnostic on the analogue branch.
    knobs = window_point(tunnelling_ratio=1 / 10)
    bosonic = bosonic_space(matter_sites)
    l3a = superlattice(matter_sites)
    bose_generators = [
        build_operator(constraint.operator, bosonic, {}) for constraint in l3a.constraint_operators
    ]
    bose_initial = bosonic.basis_state(canonical_state_configuration(matter_sites))
    coupling = forward_map(knobs)[P.COUPLING_L2C]
    states = evolve_state(
        build_operator(l3a.hamiltonian, bosonic, knobs),
        bose_initial,
        [5.0 / coupling * index / 10 for index in range(11)],
    )
    analogue_drift = float(jnp.max(jnp.abs(gauss_violation(bose_generators, states) - 0.5)))
    projector = local_subspace_projector(local_occupation_subspace(matter_sites), bosonic)
    leakage = float(jnp.max(1.0 - expectation(projector, states)))
    assert analogue_drift > 1e-3
    assert leakage > 1e-3
    assert digital_drift < analogue_drift


# ---------------------------------------------------------------------------
# Resources without operators
# ---------------------------------------------------------------------------


def test_the_resource_summary_is_computed_at_99_qubits() -> None:
    """The resource summary at N = 50 (99 qubits) is computed without building an operator."""
    matter_sites = 50
    graph = build_graph(matter_sites)
    staggered = graph.graph.node("L2a").bind(
        **{P.MASS_L2A: REFERENCE_MASS, P.COUPLING_L2A: REFERENCE_COUPLING}
    )
    started = time.perf_counter()
    qubits = to_qubits().apply(particle_hole().apply(staggered))
    product_formula = as_product_formula(
        trotterisation(time=REFERENCE_TIME, steps=8, order=1).apply(qubits)
    )
    resources = product_formula.resources()
    elapsed = time.perf_counter() - started

    assert resources.qubits == 2 * matter_sites - 1 == 99
    assert elapsed < 10.0, f"the operator-free path took {elapsed:.2f}s"
    assert resources.steps == 8
    assert resources.order == 1
    assert resources.time == REFERENCE_TIME
    assert resources.hamiltonian_layers == 3
    assert resources.factors_per_step == 4 * (matter_sites - 1) + matter_sites
    assert resources.factor_count == resources.factors_per_step * 8
    assert set(resources.factors_by_locality) == {1, 3}
    assert resources.factors_by_locality[3] == 4 * (matter_sites - 1) * 8
    assert resources.depth_in_layers == resources.depth_per_step * 8
    assert math.isfinite(resources.error_bound)
    assert resources.error_bound > 0.0
    # The bound uses the operator-free norm estimate.
    assert not resources.exact_norms

    # Support structure, also operator-free.
    supports = product_formula.support_structure()
    assert len(supports) == resources.factors_per_step
    assert max(max(support) for support in supports if support) == 2 * matter_sites - 2


# ---------------------------------------------------------------------------
# The step count as a discrete solve
# ---------------------------------------------------------------------------


def test_the_step_count_is_an_integer_solve_with_witnessed_minimality() -> None:
    """The minimal step count is integral, monotone in the accuracy and witnessed minimal."""
    model = qubit_model(4)
    layers = interleaved_layers_of(model)
    estimator = SpectralNormEstimator(len(model.structure.sites))

    previous = 0
    for accuracy in (1e-1, 1e-2, 1e-3, 1e-4):
        outcome = minimal_steps(layers, 2, REFERENCE_TIME, accuracy, estimator=estimator)
        assert outcome.status.is_success
        steps = outcome.point[STEPS_SYMBOL]
        assert steps == int(steps)
        count = int(steps)
        assert count > previous
        previous = count
        assert "does not" in outcome.notes["minimality"]

        schedule = suzuki(2, len(layers))
        passing = error_bound(layers, schedule, REFERENCE_TIME, count, estimator).value
        failing = error_bound(layers, schedule, REFERENCE_TIME, count - 1, estimator).value
        assert passing <= accuracy
        assert failing > accuracy


def test_the_joint_order_and_step_choice_is_solved_and_is_not_always_the_highest_order() -> None:
    """The resource-minimal (order, steps) pair at accuracy 1e-3 is not the highest order."""
    model = qubit_model(4)
    layers = interleaved_layers_of(model)
    estimator = SpectralNormEstimator(len(model.structure.sites))

    candidates = resource_candidates(
        layers, REFERENCE_TIME, 1e-3, orders=(1, 2, 4), estimator=estimator
    )
    assert {candidate.order for candidate in candidates} == {1, 2, 4}
    best = minimal_resource_setting(candidates)
    assert best.order != max(candidate.order for candidate in candidates)
    assert best.cost == min(candidate.cost for candidate in candidates)
    assert best.steps >= 1
    assert best.depth_in_layers > 0
    # Higher order buys fewer steps but more factors per step.
    by_order = {candidate.order: candidate for candidate in candidates}
    assert by_order[4].steps < by_order[1].steps
    assert by_order[4].cost > by_order[2].cost


# ---------------------------------------------------------------------------
# Layer commutation is derived
# ---------------------------------------------------------------------------


def test_no_layer_pair_of_this_model_commutes() -> None:
    """None of the three layer pairs commute."""
    model = qubit_model(4)
    layers = interleaved_layers_of(model)
    assert layers.names == ("H_M", "H_even", "H_odd")
    assert layers.commuting_pairs() == frozenset()
    assert len(layers.non_commuting_pairs()) == 3
    for index_a, index_b in layers.non_commuting_pairs():
        assert not layers.layer_commutator(index_a, index_b).is_zero

    bound = error_bound(
        layers,
        suzuki(1, 3),
        REFERENCE_TIME,
        8,
        SpectralNormEstimator(len(model.structure.sites)),
    )
    # The bound sums all three commutators.
    assert len(bound.components) == 3
    assert bound.value == pytest.approx(
        REFERENCE_TIME**2 / (2 * 8) * sum(bound.components.values())
    )


def _charge_conserving_model(matter_sites: int) -> HamiltonianModel:
    """A variant whose coupling conserves matter charge, so the mass layer commutes with it."""
    hamiltonian = OperatorSum()
    for link in range(matter_sites - 1):
        left, middle, right = 2 * link, 2 * link + 1, 2 * link + 2
        hamiltonian = (
            hamiltonian
            + word(
                REFERENCE_COUPLING / 2, sigma_minus(left), sigma_plus(middle), sigma_plus(right)
            ).plus_adjoint()
        )
    for site in range(matter_sites):
        hamiltonian = hamiltonian + word(REFERENCE_MASS, number(2 * site))
    return HamiltonianModel(
        name="charge-conserving variant",
        level=AbstractionLevel.INTERMEDIATE,
        structure=matter_gauge_structure(
            matter_sites, Algebra.QUBIT, Algebra.QUBIT, name="charge-conserving variant"
        ),
        hamiltonian=hamiltonian,
    )


def test_a_commuting_layer_pair_is_detected_and_the_bound_and_step_count_fall() -> None:
    """A commuting layer pair is detected, and the bound and step count fall accordingly."""
    matter_sites = 4
    variant = _charge_conserving_model(matter_sites)
    environment: dict[str, float] = {}
    pauli = pauli_sum_from_operator(variant.hamiltonian, environment, variant.structure, "variant")
    mass_terms = [term for term in variant.hamiltonian.terms if len(term.support) == 1]
    even_terms = [
        term
        for term in variant.hamiltonian.terms
        if len(term.support) == 3 and (min(term.support) // 2) % 2 == 0
    ]
    odd_terms = [
        term
        for term in variant.hamiltonian.terms
        if len(term.support) == 3 and (min(term.support) // 2) % 2 == 1
    ]
    layers = declared_layers(
        [
            (
                name,
                pauli_sum_from_operator(
                    OperatorSum(tuple(terms)), environment, variant.structure, name
                ),
            )
            for name, terms in (
                ("H_M", mass_terms),
                ("H_even", even_terms),
                ("H_odd", odd_terms),
            )
        ]
    )
    assert not pauli.is_zero

    commuting = layers.commuting_pairs()
    assert (0, 1) in commuting
    assert (0, 2) in commuting
    assert (1, 2) not in commuting
    assert "commute" in layers.commutation_report()

    estimator = SpectralNormEstimator(len(variant.structure.sites))
    variant_bound = error_bound(layers, suzuki(1, 3), REFERENCE_TIME, 8, estimator)
    assert len(variant_bound.components) == 1

    original = interleaved_layers_of(qubit_model(matter_sites))
    original_bound = error_bound(original, suzuki(1, 3), REFERENCE_TIME, 8, estimator)
    assert variant_bound.value < original_bound.value

    accuracy = 1e-3
    variant_steps = minimal_steps(layers, 1, REFERENCE_TIME, accuracy, estimator=estimator)
    original_steps = minimal_steps(original, 1, REFERENCE_TIME, accuracy, estimator=estimator)
    assert variant_steps.point[STEPS_SYMBOL] < original_steps.point[STEPS_SYMBOL]


def test_the_derived_partition_is_also_available_and_internally_commuting() -> None:
    """The derived partition is internally commuting and sums to the declared one."""
    model = qubit_model(3)
    derived = derived_layers_of(model)
    declared = interleaved_layers_of(model)
    assert derived.provenance == "derived"
    assert declared.provenance == "declared"
    for layer in derived.layers:
        assert layer.is_internally_commuting()
    # Both partition the same Hamiltonian.
    difference = derived.total() - declared.total()
    assert difference.is_zero


# ---------------------------------------------------------------------------
# Both branch points give equivalent digital results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matter_sites", [3, 4])
def test_branching_at_l2a_gives_a_unitarily_equivalent_qubit_hamiltonian(
    matter_sites: int,
) -> None:
    """Both branch points yield qubit Hamiltonians with identical spectra."""
    graph = build_graph(matter_sites)
    staggered = graph.graph.node("L2a").bind(
        **{P.MASS_L2A: REFERENCE_MASS, P.COUPLING_L2A: REFERENCE_COUPLING}
    )
    via_l2b = to_qubits().apply(particle_hole().apply(staggered))
    via_l2a = to_qubits_from_staggered().apply(staggered)
    assert isinstance(via_l2b, HamiltonianModel)
    assert isinstance(via_l2a, HamiltonianModel)

    space = HilbertSpace.of(via_l2b.structure)
    left, _ = eigensystem(build_operator(via_l2b.hamiltonian, space, via_l2b.environment()))
    right, _ = eigensystem(build_operator(via_l2a.hamiltonian, space, via_l2a.environment()))
    assert float(jnp.max(jnp.abs(left - right))) < TOLERANCE


def test_the_dag_supports_more_than_one_branch_point(graph4: SchwingerGraph) -> None:
    """The graph holds both branch points and enumerates the pipelines through each."""
    assert {edge.target for edge in graph4.graph.outgoing("L2a")} == {"L2b", "L2d_st"}
    targets = set(graph4.graph.terminal_targets("L1"))
    assert {"L3a", "L3b", "L3b_st"} <= targets
    assert len(graph4.graph.pipelines("L1", "L3b_st")) == 1


# ---------------------------------------------------------------------------
# The two branches are comparable
# ---------------------------------------------------------------------------


def test_comparison_table(graph3: SchwingerGraph) -> None:
    """Both branches report exactness, approximation kinds and a cost row."""
    matter_sites = 3
    accuracy = 1e-2
    rows: dict[str, dict[str, object]] = {}

    # -- the analogue branch: cost in validity margin and required J/U ---------------
    knobs = window_point(tunnelling_ratio=1 / 40)
    effective = forward_map(knobs)
    report = graph3.analogue.error_report({**knobs, **effective, P.ELECTRIC_GAP: 0.5})
    rows["analogue"] = {
        "target": "L3a",
        "kind_of_artifact": "Hamiltonian",
        "exactness": graph3.analogue.exactness,
        "approximation_kinds": graph3.analogue.approximation_kinds,
        "cost": {
            "J_over_U": knobs[P.TUNNELLING] / knobs[P.INTERACTION],
            "kappa": effective[P.COUPLING_L2C],
            "weakest_margin": report.regime.weakest_margin,
        },
        "resource_controlled": ApproximationKind.RESOURCE_CONTROLLED
        in graph3.analogue.approximation_kinds,
    }

    # -- the digital branch: cost in steps, depth and factors ------------------------
    model = qubit_model(
        matter_sites,
        coupling=effective[P.COUPLING_L2C],
        mass=effective[P.MASS_L2C],
    )
    layers = interleaved_layers_of(model)
    estimator = SpectralNormEstimator(len(model.structure.sites))
    horizon = 5.0 / effective[P.COUPLING_L2C]
    candidates = resource_candidates(layers, horizon, accuracy, orders=(1, 2), estimator=estimator)
    assert candidates
    best = minimal_resource_setting(candidates)
    rows["digital"] = {
        "target": "L3b",
        "kind_of_artifact": "product formula",
        "exactness": graph3.digital.exactness,
        "approximation_kinds": graph3.digital.approximation_kinds,
        "cost": {
            "order": best.order,
            "steps": best.steps,
            "depth_in_layers": best.depth_in_layers,
            "k_local_factors": best.cost,
        },
        "resource_controlled": ApproximationKind.RESOURCE_CONTROLLED
        in graph3.digital.approximation_kinds,
    }

    assert set(rows) == {"analogue", "digital"}
    assert rows["analogue"]["approximation_kinds"] == frozenset({ApproximationKind.REGIME_LIMITED})
    assert rows["digital"]["approximation_kinds"] == frozenset(
        {ApproximationKind.REGIME_LIMITED, ApproximationKind.RESOURCE_CONTROLLED}
    )
    assert rows["digital"]["resource_controlled"] is True
    assert rows["analogue"]["resource_controlled"] is False
    # Both inherit the regime-limited truncation from the shared prefix.
    digital_kinds = rows["digital"]["approximation_kinds"]
    assert isinstance(digital_kinds, frozenset)
    assert ApproximationKind.REGIME_LIMITED in digital_kinds


# ---------------------------------------------------------------------------
# The level-4 line is not crossed
# ---------------------------------------------------------------------------

#: Vendor SDKs and gate-level libraries that must not appear anywhere in the package.
VENDOR_MODULES = (
    "qiskit",
    "cirq",
    "pulser",
    "braket",
    "pennylane",
    "pytket",
    "qutip",
    "projectq",
)

#: Substrings marking a level-4 concept in a public name.
LEVEL_FOUR_NAMES = (
    "cnot",
    "cx_count",
    "t_count",
    "clifford",
    "transpile",
    "routing",
    "swap_insert",
    "gate_count",
    "gate_set",
    "pulse_schedule",
    "qasm",
)


def _package_files() -> list[Path]:
    """Every module of the package."""
    root = Path(__file__).resolve().parent.parent / "qsimod"
    return sorted(root.rglob("*.py"))


def test_no_vendor_sdk_is_imported() -> None:
    """No package module imports a vendor SDK."""
    offenders: list[str] = []
    for source in _package_files():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root_name = name.split(".")[0]
                if root_name in VENDOR_MODULES:
                    offenders.append(f"{source.name} imports {name}")
    assert not offenders, offenders


def test_no_public_name_emits_a_level_four_artifact() -> None:
    """No public name contains a level-4 substring."""
    offenders: list[str] = []
    modules = [qsimod]
    for info in pkgutil.walk_packages(qsimod.__path__, prefix="qsimod."):
        modules.append(importlib.import_module(info.name))
    for module in modules:
        exported = getattr(module, "__all__", None)
        names = (
            list(exported)
            if exported
            else [name for name in dir(module) if not name.startswith("_")]
        )
        for name in names:
            lowered = name.lower()
            offenders.extend(
                f"{module.__name__}.{name}"
                for forbidden in LEVEL_FOUR_NAMES
                if forbidden in lowered
            )
    assert not offenders, offenders


def test_the_resource_model_counts_k_local_unitaries_and_layers() -> None:
    """The resource row counts k-local factors and layer depth and no gate-level quantities."""
    model = qubit_model(3)
    resources = as_product_formula(
        trotterisation(time=1.0, steps=2, order=2).apply(model)
    ).resources()
    row = resources.as_row()
    assert "factor_count" in row
    assert "depth_in_layers" in row
    assert set(resources.factors_by_locality) <= {0, 1, 2, 3}
    assert not any("cnot" in key.lower() for key in row)
    assert not any("gate" in key.lower() for key in row)


def test_the_step_budget_is_honoured() -> None:
    """``max_steps`` caps the search: below the minimum it is ``UNSOLVED``, at it exact."""
    model = qubit_model(3)
    layers = interleaved_layers_of(model)
    estimator = SpectralNormEstimator(len(model.structure.sites))
    accuracy = 1e-3

    unbounded = minimal_steps(layers, 2, REFERENCE_TIME, accuracy, estimator=estimator)
    assert unbounded.status is SolveStatus.EXACT_SOLUTION
    needed = int(unbounded.point[STEPS_SYMBOL])
    assert needed > 2

    exhausted = minimal_steps(
        layers, 2, REFERENCE_TIME, accuracy, estimator=estimator, max_steps=needed - 1
    )
    assert exhausted.status is SolveStatus.UNSOLVED
    assert exhausted.point == {}
    assert f"up to {needed - 1}" in exhausted.termination

    just_enough = minimal_steps(
        layers, 2, REFERENCE_TIME, accuracy, estimator=estimator, max_steps=needed
    )
    assert just_enough.status is SolveStatus.EXACT_SOLUTION
    assert int(just_enough.point[STEPS_SYMBOL]) == needed

    # Orders that cannot meet the target within the budget are absent from the candidates.
    candidates = resource_candidates(
        layers, REFERENCE_TIME, accuracy, orders=(1, 2), estimator=estimator, max_steps=needed
    )
    assert {candidate.order for candidate in candidates} == {2}
