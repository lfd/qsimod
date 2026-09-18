"""Unit tests for the scalar, symbolic, Pauli, realisation and structure layers.

They pin the conventions and invariants the other test modules assume.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import pytest

from qsimod.affine import bound as affine_bound
from qsimod.artifact import HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models.gauge import (
    ElectricField,
    gauge_violation_observable,
    gauss_operators,
    matter_gauge_structure,
    occupation_projector,
)
from qsimod.normal_form import constant_word, difference, equivalent, normal_form, product
from qsimod.parameters import AdmissibleSet, Bound, box
from qsimod.pauli import (
    IDENTITY_STRING,
    PauliAxis,
    PauliString,
    PauliSum,
    commutator,
    nested_commutator,
    pauli_sum_from_operator,
)
from qsimod.realise import (
    HilbertSpace,
    RealisationRequest,
    SectorBasis,
    build_operator,
    build_operator_in,
    build_pauli_sum,
    hermiticity_defect,
    local_subspace_projector,
    max_abs_deviation,
    propagator,
    restrict,
    sandwich,
    sector_projector,
    subspace_indices,
)
from qsimod.scalar import Const, DomainError, Interval, Symbol, simplify, sqrt
from qsimod.structure import (
    Algebra,
    DegreeOfFreedom,
    Lattice,
    LatticeGeometry,
    StructurePattern,
    StructureType,
    SymmetryDeclaration,
)
from qsimod.symbolic import (
    ALLOWED_OPERATORS,
    OperatorSum,
    annihilate,
    create,
    number,
    pauli_x,
    pauli_y,
    pauli_z,
    sigma_minus,
    sigma_plus,
    spin_minus,
    spin_plus,
    spin_z,
    word,
)
from qsimod.transformations import interleaved_layers_of
from qsimod.trotter import as_product_formula, suzuki
from qsimod.units import from_hertz, from_kilohertz, to_hertz, to_kilohertz
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import (
    build_graph,
    canonical_state_configuration,
    effective_bosonic,
    homogeneous_quantum_link,
    local_occupation_subspace,
    staggered_quantum_link,
    theory,
    trotterisation,
)
from tests.conftest import REFERENCE_MASS, qubit_model

TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_the_hertz_conversion_carries_the_factor_of_two_pi() -> None:
    """Frequency conversions carry the factor of 2*pi and round-trip."""
    assert from_hertz(33.0) == pytest.approx(2.0 * math.pi * 33.0e-3)
    assert to_hertz(from_hertz(33.0)) == pytest.approx(33.0)
    assert from_kilohertz(1.0) == pytest.approx(2.0 * math.pi)
    assert to_kilohertz(from_kilohertz(2.5)) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# The scalar layer
# ---------------------------------------------------------------------------


def test_a_scalar_expression_substitutes_evaluates_and_renders() -> None:
    """A scalar expression reports its symbols, evaluates, substitutes and renders."""
    tunnelling, interaction = Symbol("J"), Symbol("U")
    expression = sqrt(2) * tunnelling**2.0 / interaction
    assert expression.symbols() == {"J", "U"}
    assert expression.evaluate_real({"J": 0.02, "U": 1.0}) == pytest.approx(math.sqrt(2) * 4e-4)
    partly = expression.substitute({"U": 2.0})
    assert partly.symbols() == {"J"}
    assert "sqrt(2)" in str(expression)


def test_a_vanishing_denominator_raises_rather_than_returning_a_number() -> None:
    """Evaluating at a vanishing denominator raises ``DomainError`` naming the expression."""
    with pytest.raises(DomainError) as caught:
        (1 / (Symbol("x") - Symbol("y"))).evaluate_real({"x": 1.0, "y": 1.0})
    assert caught.value.expression.symbols() == {"x", "y"}


def test_an_unbound_parameter_is_named() -> None:
    """Evaluating with a missing parameter raises ``KeyError`` naming it."""
    with pytest.raises(KeyError, match="kappa"):
        Symbol("kappa").evaluate_real({})


def test_interval_arithmetic_is_sound_on_a_reciprocal_that_straddles_zero() -> None:
    """A reciprocal over an interval straddling zero is unbounded; a definite one is not."""
    straddling = Interval(-1.0, 2.0) ** -1.0
    assert not straddling.is_bounded
    assert straddling.contains(1e9)

    definite = Interval(0.5, 2.0) ** -1.0
    assert definite.is_bounded
    assert definite.lo == pytest.approx(0.5)
    assert definite.hi == pytest.approx(2.0)

    # 0 * inf is 0.
    assert (Interval(0.0, 0.0) * Interval(-math.inf, math.inf)).is_bounded


def test_an_interval_bound_encloses_the_true_range() -> None:
    """Sampled points must land inside the computed interval."""
    expression = sqrt(2) * Symbol("J") ** 2.0 / (Symbol("d") - Symbol("t"))
    box = {"J": Interval(0.0, 0.1), "d": Interval(0.4, 0.6), "t": Interval(0.0, 0.05)}
    bound = expression.interval(box)
    assert bound.is_bounded
    for tunnelling in (0.0, 0.05, 0.1):
        for depth in (0.4, 0.5, 0.6):
            for tilt in (0.0, 0.02, 0.05):
                value = expression.evaluate_real({"J": tunnelling, "d": depth, "t": tilt})
                assert bound.contains(value, tolerance=1e-12)


def test_an_affine_bound_is_sound_and_tighter_where_a_symbol_repeats() -> None:
    """An affine bound is sound and tighter than the interval bound when a symbol repeats."""
    interaction, depth = Symbol("U"), Symbol("d")
    box = {"U": Interval(1.0, 4.0), "d": Interval(0.5, 2.0)}

    cancelling = affine_bound(interaction - interaction, box)
    assert cancelling is not None
    assert cancelling.lo == pytest.approx(0.0)
    assert cancelling.hi == pytest.approx(0.0)
    assert (interaction - interaction).interval(box).hi == pytest.approx(3.0)

    expression = (interaction + depth) * (interaction - depth)
    bound = affine_bound(expression, box)
    assert bound is not None
    for value in (1.0, 2.5, 4.0):
        for other in (0.5, 1.25, 2.0):
            taken = expression.evaluate_real({"U": value, "d": other})
            assert bound.contains(taken, tolerance=1e-12)
    assert bound.hi < expression.interval(box).hi


def test_an_affine_bound_declines_rather_than_guessing_at_a_pole() -> None:
    """No affine bound is returned for a straddling denominator or an unbound symbol."""
    assert affine_bound(1 / (Symbol("x") - 1.0), {"x": Interval(0.0, 2.0)}) is None
    assert affine_bound(Symbol("y"), {}) is None


def test_simplification_folds_what_is_safe_and_leaves_what_is_not() -> None:
    """Simplification folds neutral constants and leaves fractional powers and guarded poles."""
    tunnelling, interaction = Symbol("J"), Symbol("U")

    nested = (tunnelling + interaction) + (interaction + 1.0) + 2.0
    assert str(nested) == "J + U + U + 3"
    assert str(simplify(1.0 * tunnelling * 1.0)) == "J"
    assert str(simplify((2 * interaction) ** 1.0)) == "2*U"
    assert "sqrt(2)" in str(sqrt(2) * tunnelling)

    guarded = 0.0 * (1 / interaction)
    assert simplify(guarded).symbols() == {"U"}
    with pytest.raises(DomainError):
        guarded.evaluate_real({"U": 0.0})


def test_a_complex_coefficient_renders_and_conjugates() -> None:
    """A complex coefficient renders its imaginary part and conjugates under the adjoint."""
    term = word(-1j / 2, create(0), annihilate(2))
    assert "-0.5i" in str(term)
    assert "0.5i" in str(term.adjoint())
    assert not str(term.adjoint()).count("-0.5i")
    assert Const(2).evaluate({}) == 2.0


# ---------------------------------------------------------------------------
# The symbolic layer
# ---------------------------------------------------------------------------


def test_an_operator_illegal_on_its_site_is_refused_by_name() -> None:
    """An operator not in the algebra of its site is refused by name."""
    model = homogeneous_quantum_link(3)
    illegal = model.hamiltonian + word(1.0, pauli_x(1))
    algebra_at = {site: model.structure.algebra_at(site) for site in model.structure.sites}
    with pytest.raises(ValueError, match="not a legal operator on a spin-1/2 site"):
        illegal.validate_against(algebra_at)
    assert pauli_x(1).symbol not in ALLOWED_OPERATORS[Algebra.SPIN_HALF]


def test_hermitian_conjugation_reverses_and_conjugates() -> None:
    """The adjoint reverses operator order and conjugates factors, giving a Hermitian sum."""
    coupling = word(Symbol("kappa") / 2, annihilate(0), spin_plus(1), annihilate(2))
    adjoint = coupling.adjoint()
    assert len(adjoint) == 1
    operators = adjoint.terms[0].operators
    assert [operator.site for operator in operators] == [2, 1, 0]
    assert str(operators[1]) == "S⁻_1"

    space = HilbertSpace.of(homogeneous_quantum_link(3).structure)
    hermitian = build_operator(coupling.plus_adjoint(), space, {"kappa": 0.7})
    assert hermiticity_defect(hermitian) < TOLERANCE


def test_terms_of_identical_operator_content_collect() -> None:
    """``collected`` merges coefficients of identical words and keeps reordered ones apart."""
    total = word(1.0, number(0)) + word(2.0, number(0)) + word(1.0, number(2))
    collected = total.collected()
    assert len(collected) == 2
    assert collected.terms[0].coefficient.evaluate_real({}) == pytest.approx(3.0)
    # Reordered words stay distinct.
    words = word(1.0, annihilate(0), annihilate(1)) + word(1.0, annihilate(1), annihilate(0))
    assert len(words.collected()) == 2


def test_support_and_locality_come_from_the_term_structure() -> None:
    """Support and maximal locality are read from the term structure."""
    coupling = word(1.0, annihilate(0), spin_plus(1), annihilate(2))
    assert coupling.support == {0, 1, 2}
    assert coupling.max_locality == 3
    doubled = word(1.0, create(1), create(1))
    assert doubled.max_locality == 1


def test_an_empty_operator_sum_has_no_support_or_parameters() -> None:
    """An empty operator sum has empty support, no parameters and locality zero."""
    empty = OperatorSum()
    assert empty.support == frozenset()
    assert empty.parameters == frozenset()
    assert empty.max_locality == 0


# ---------------------------------------------------------------------------
# Pauli algebra, and its agreement with the realiser
# ---------------------------------------------------------------------------


def test_the_pauli_triple_is_right_handed_in_both_implementations() -> None:
    """``XY = iZ`` holds, and ``qsimod.pauli`` agrees with ``qsimod.realise`` on each axis."""
    structure = matter_gauge_structure(2, Algebra.QUBIT, Algebra.QUBIT, name="one qubit's worth")
    space = HilbertSpace.of(structure)
    matrices = {
        "X": build_operator(word(1.0, pauli_x(0)), space, {}),
        "Y": build_operator(word(1.0, pauli_y(0)), space, {}),
        "Z": build_operator(word(1.0, pauli_z(0)), space, {}),
    }
    assert max_abs_deviation(matrices["X"] @ matrices["Y"], 1j * matrices["Z"]) < TOLERANCE
    assert max_abs_deviation(matrices["Y"] @ matrices["Z"], 1j * matrices["X"]) < TOLERANCE
    assert max_abs_deviation(matrices["Z"] @ matrices["X"], 1j * matrices["Y"]) < TOLERANCE

    for axis, name in ((PauliAxis.X, "X"), (PauliAxis.Y, "Y"), (PauliAxis.Z, "Z")):
        from_pauli = build_pauli_sum(
            PauliSum.from_terms([(PauliString.single(0, axis), 1.0)]), len(structure.sites)
        )
        assert max_abs_deviation(from_pauli, matrices[name]) < TOLERANCE


def test_the_occupation_conventions_agree_across_the_two_implementations() -> None:
    """``n = (Z+1)/2`` and ``sigma^+ = |1><0|``, in both the symbolic and Pauli paths."""
    structure = matter_gauge_structure(2, Algebra.QUBIT, Algebra.QUBIT, name="three qubits")
    space = HilbertSpace.of(structure)
    for symbolic in (
        word(1.0, number(0)),
        word(1.0, sigma_plus(0)),
        word(1.0, sigma_minus(2)),
        word(1.0, sigma_minus(0), sigma_plus(1), sigma_minus(2)).plus_adjoint(),
    ):
        direct = build_operator(symbolic, space, {})
        expanded = build_pauli_sum(
            pauli_sum_from_operator(symbolic, {}, structure), len(structure.sites)
        )
        assert max_abs_deviation(direct, expanded) < TOLERANCE


def test_pauli_products_commutators_and_norms() -> None:
    """Pauli string products, commutators, norms and weights follow the standard algebra."""
    x0 = PauliString.single(0, PauliAxis.X)
    y0 = PauliString.single(0, PauliAxis.Y)
    z1 = PauliString.single(1, PauliAxis.Z)

    product, phase = x0.times(y0)
    assert product == PauliString.single(0, PauliAxis.Z)
    assert phase == 1j
    assert not x0.commutes_with(y0)
    assert x0.commutes_with(z1)
    assert x0.times(x0)[0] == IDENTITY_STRING

    left = PauliSum.from_terms([(x0, 1.0)])
    right = PauliSum.from_terms([(y0, 1.0)])
    bracket = commutator(left, right)
    assert len(bracket) == 1
    assert bracket.terms[PauliString.single(0, PauliAxis.Z)] == pytest.approx(2j)
    assert commutator(left, PauliSum.from_terms([(z1, 1.0)])).is_zero
    assert left.one_norm == pytest.approx(1.0)
    assert left.max_weight == 1
    assert PauliSum.from_terms([(x0, 1.0), (x0, -1.0)]).is_zero

    with pytest.raises(ValueError, match="at least two operands"):
        nested_commutator([left])
    assert len(nested_commutator([left, right, left])) == 1


def test_a_pauli_string_is_refused_if_its_axes_are_not_one_per_qubit() -> None:
    """A Pauli string with unsorted or repeated qubits is refused."""
    with pytest.raises(ValueError, match="one per qubit"):
        PauliString(((1, PauliAxis.X), (0, PauliAxis.Y)))
    with pytest.raises(ValueError, match="one per qubit"):
        PauliString(((0, PauliAxis.X), (0, PauliAxis.Y)))


def test_a_bosonic_model_has_no_pauli_expansion_and_says_so() -> None:
    """Expanding a bosonic Hamiltonian in Pauli strings is refused."""
    model = effective_bosonic(3)
    with pytest.raises(ValueError, match="two-level"):
        pauli_sum_from_operator(
            model.hamiltonian,
            {"effective_bosonic.m": 0.1, "effective_bosonic.kappa": 0.2},
            model.structure,
        )


# ---------------------------------------------------------------------------
# The Hilbert space
# ---------------------------------------------------------------------------


def test_the_basis_index_and_the_configuration_are_inverse() -> None:
    """``index_of`` and ``configuration`` are inverse, with site 0 most significant."""
    space = HilbertSpace.of(effective_bosonic(3).structure, RealisationRequest(boson_cutoff=2))
    assert space.dimension == 3**5
    assert space.dimensions == (3, 3, 3, 3, 3)
    for index in range(space.dimension):
        assert space.index_of(space.configuration(index)) == index
    # Site 0 is the most significant factor.
    assert space.configuration(0) == (0, 0, 0, 0, 0)
    assert space.strides == (81, 27, 9, 3, 1)
    assert space.label(space.index_of((1, 0, 1, 0, 1))) == "|1 0 1 0 1>"


def test_an_out_of_range_occupation_is_refused() -> None:
    """Occupations above the cutoff, wrong-length configurations and bad indices are refused."""
    space = HilbertSpace.of(effective_bosonic(2).structure, RealisationRequest(boson_cutoff=1))
    with pytest.raises(ValueError, match="outside"):
        space.index_of((2, 0, 0))
    with pytest.raises(ValueError, match="entries"):
        space.index_of((0, 0))
    with pytest.raises(IndexError, match="out of range"):
        space.configuration(space.dimension)


def test_an_untruncated_gauge_link_cannot_be_realised() -> None:
    """A structure with an untruncated gauge link has no finite-dimensional Hilbert space."""
    with pytest.raises(ValueError, match="no finite-dimensional representation"):
        HilbertSpace.of(theory(3).structure)


def test_the_subspace_projector_needs_the_declared_cutoff() -> None:
    """The local occupation subspace requires ``boson_cutoff >= 2`` and refuses less."""
    matter_sites = 2
    subspace = local_occupation_subspace(matter_sites)
    assert subspace.required_cutoff() == 2
    too_small = HilbertSpace.of(
        effective_bosonic(matter_sites).structure, RealisationRequest(boson_cutoff=1)
    )
    with pytest.raises(ValueError, match="boson_cutoff >= 2"):
        local_subspace_projector(subspace, too_small)


def test_a_sector_realisation_agrees_exactly_with_the_dense_one() -> None:
    """The sector realisation has the spectrum of the projected dense operator."""
    environment = {P.MASS_EFFECTIVE_BOSONIC: 0.31, P.COUPLING_EFFECTIVE_BOSONIC: 0.87}
    for matter_sites in (2, 3, 4):
        model = effective_bosonic(matter_sites)
        subspace = local_occupation_subspace(matter_sites)
        constraints = gauss_operators(matter_sites, field=ElectricField.BOSON)
        request = RealisationRequest(boson_cutoff=subspace.required_cutoff())

        basis = SectorBasis.of(model.structure, subspace, constraints, request)
        in_sector = build_operator_in(model.hamiltonian, basis, environment)

        space = HilbertSpace.of(model.structure, request)
        projector = local_subspace_projector(subspace, space) @ sector_projector(
            constraints, space, {}
        )
        dense = restrict(
            sandwich(build_operator(model.hamiltonian, space, environment), projector),
            subspace_indices(projector),
        )

        assert basis.dimension == dense.shape[0]
        assert float(
            jnp.max(jnp.abs(jnp.linalg.eigvalsh(in_sector) - jnp.linalg.eigvalsh(dense)))
        ) == pytest.approx(0.0, abs=1e-9), matter_sites


def test_the_sector_of_the_effective_model_is_fibonacci_dimensional() -> None:
    """The Gauss sector of the effective model has Fibonacci dimension in the site count."""
    dimensions = []
    for matter_sites in (2, 3, 4, 5, 6):
        subspace = local_occupation_subspace(matter_sites)
        basis = SectorBasis.of(
            effective_bosonic(matter_sites).structure,
            subspace,
            gauss_operators(matter_sites, field=ElectricField.BOSON),
            RealisationRequest(boson_cutoff=subspace.required_cutoff()),
        )
        dimensions.append(basis.dimension)
    assert dimensions == [2, 3, 5, 8, 13]


def test_a_sector_realisation_refuses_fermionic_degrees_of_freedom() -> None:
    """A sector basis over fermionic degrees of freedom is refused."""
    with pytest.raises(ValueError, match="Jordan-Wigner"):
        SectorBasis.of(homogeneous_quantum_link(3).structure)


def test_an_occupation_projector_is_the_interpolating_polynomial_it_claims_to_be() -> None:
    """``P_A`` is one on occupations in ``A`` and zero elsewhere, at every cutoff."""
    for cutoff in (2, 3, 4):
        space = HilbertSpace.of(
            effective_bosonic(2).structure, RealisationRequest(boson_cutoff=cutoff)
        )
        for selected in ((0,), (1,), (0, 2), (1, cutoff)):
            matrix = build_operator(occupation_projector(1, selected, cutoff), space, {})
            for occupation in range(cutoff + 1):
                state = space.basis_state((0, occupation, 0))
                expected = 1.0 if occupation in selected else 0.0
                assert float(jnp.vdot(state, matrix @ state).real) == pytest.approx(
                    expected, abs=1e-9
                ), (cutoff, selected, occupation)


def test_the_gauge_violation_observable_is_the_link_parity_it_is_measured_as() -> None:
    """The gauge violation observable equals the mean link-occupation parity."""
    matter_sites = 3
    subspace = local_occupation_subspace(matter_sites)
    space = HilbertSpace.of(
        effective_bosonic(matter_sites).structure,
        RealisationRequest(boson_cutoff=subspace.required_cutoff()),
    )
    observable = build_operator(gauge_violation_observable(matter_sites), space, {})

    for index in range(space.dimension):
        configuration = space.configuration(index)
        links = [configuration[site] for site in (1, 3)]
        parity = sum(occupation % 2 for occupation in links) / len(links)
        state = space.basis_state(configuration)
        assert float(jnp.vdot(state, observable @ state).real) == pytest.approx(parity, abs=1e-9), (
            configuration
        )

    # The canonical state carries no violation.
    canonical = space.basis_state(canonical_state_configuration(matter_sites))
    assert float(jnp.vdot(canonical, observable @ canonical).real) == pytest.approx(0.0, abs=1e-9)


def test_the_sector_projector_keeps_the_canonical_state() -> None:
    """The sector projector with the declared boundary targets keeps the canonical state."""
    matter_sites = 3
    structure = matter_gauge_structure(
        matter_sites, Algebra.QUBIT, Algebra.QUBIT, name="qubit register"
    )
    space = HilbertSpace.of(structure)
    constraints = gauss_operators(matter_sites, field=ElectricField.QUBIT)
    projector = sector_projector(constraints, space, {})
    canonical = space.basis_state(canonical_state_configuration(matter_sites))
    assert float(jnp.vdot(canonical, projector @ canonical).real) == pytest.approx(1.0)

    # Boundary targets of zero empty the sector.
    zeroed = tuple(
        type(constraint)(
            name=constraint.name,
            index=constraint.index,
            operator=constraint.operator,
            target_value=0.0,
            is_boundary=constraint.is_boundary,
        )
        for constraint in constraints
    )
    empty = sector_projector(zeroed, space, {})
    assert float(jnp.max(jnp.abs(empty))) == pytest.approx(0.0)


def test_the_jordan_wigner_string_is_empty_between_adjacent_matter_sites() -> None:
    """The Jordan-Wigner string of a matter site lists only the preceding matter sites."""
    space = HilbertSpace.of(homogeneous_quantum_link(4).structure)
    assert space.fermionic_sites == (0, 2, 4, 6)
    assert space.jordan_wigner_string(0) == ()
    assert space.jordan_wigner_string(4) == (0, 2)
    # A gauge site carries no string at all.
    assert space.jordan_wigner_string(1) == ()


def test_the_propagator_is_unitary_and_the_evolution_exact() -> None:
    """The propagator is unitary and composes additively in time."""
    model = homogeneous_quantum_link(3)
    space = HilbertSpace.of(model.structure)
    operator = build_operator(
        model.hamiltonian,
        space,
        {P.MASS_QUANTUM_LINK_HOMOGENEOUS: 0.3, P.COUPLING_QUANTUM_LINK_HOMOGENEOUS: 0.8},
    )
    unitary = propagator(operator, 1.7)
    identity = space.identity()
    assert max_abs_deviation(unitary @ unitary.conj().T, identity) < 1e-13
    # exp(-iHt1) exp(-iHt2) == exp(-iH(t1+t2)) to machine precision.
    assert (
        max_abs_deviation(
            propagator(operator, 0.7) @ propagator(operator, 1.0),
            propagator(operator, 1.7),
        )
        < 1e-13
    )


# ---------------------------------------------------------------------------
# Structural patterns
# ---------------------------------------------------------------------------


def test_an_empty_pattern_matches_anything() -> None:
    """An empty structure pattern matches any structure."""
    assert StructurePattern().matches(homogeneous_quantum_link(3).structure)


def test_a_pattern_guarantee_meets_a_weaker_demand_and_not_a_stronger_one() -> None:
    """A pattern meets a weaker demand, fails a stronger one and reports the gap."""
    guaranteed = StructurePattern(
        geometries=frozenset({LatticeGeometry.OPEN_CHAIN_1D}),
        description="an open chain",
    )
    weaker = StructurePattern(
        geometries=frozenset({LatticeGeometry.OPEN_CHAIN_1D, LatticeGeometry.PERIODIC_CHAIN_1D}),
        description="any 1D chain",
    )
    stronger = StructurePattern(
        geometries=frozenset({LatticeGeometry.PERIODIC_CHAIN_1D}),
        description="a periodic chain",
    )
    assert guaranteed.meets(weaker)
    assert not guaranteed.meets(stronger)
    assert guaranteed.guarantee_gaps(stronger)[0].aspect == "lattice.geometry"


def test_a_structure_refuses_duplicate_roles_and_unsorted_sites() -> None:
    """Unsorted sites, duplicate roles and single-site lattices are refused."""
    with pytest.raises(ValueError, match="ascending order"):
        DegreeOfFreedom("matter", Algebra.FERMION, (2, 0))
    with pytest.raises(ValueError, match="duplicate"):
        StructureType(
            lattice=Lattice(LatticeGeometry.OPEN_CHAIN_1D, 3),
            degrees_of_freedom=(
                DegreeOfFreedom("matter", Algebra.FERMION, (0,)),
                DegreeOfFreedom("matter", Algebra.BOSON, (1,)),
            ),
        )
    with pytest.raises(ValueError, match="at least two sites"):
        Lattice(LatticeGeometry.OPEN_CHAIN_1D, 1)


def test_a_spin_z_operator_is_half_a_pauli_z() -> None:
    """``S^z`` realises as ``Z/2``."""
    spin_structure = homogeneous_quantum_link(2).structure
    spin_space = HilbertSpace.of(spin_structure)
    spin = build_operator(word(1.0, spin_z(1)), spin_space, {})

    qubit_space = HilbertSpace.of(
        matter_gauge_structure(2, Algebra.QUBIT, Algebra.QUBIT, name="qubits")
    )
    half_pauli = build_operator(word(0.5, pauli_z(1)), qubit_space, {})
    assert max_abs_deviation(spin, half_pauli) < TOLERANCE


# ---------------------------------------------------------------------------
# Inspection and extension API
# ---------------------------------------------------------------------------


def test_a_symmetry_can_be_declared_on_an_existing_structure() -> None:
    """``with_symmetry`` adds a declaration to a copy and leaves the base structure unchanged."""
    base = matter_gauge_structure(3, Algebra.QUBIT, Algebra.QUBIT, name="qubits")
    extra = SymmetryDeclaration("parity", "Z(2)", local=False)
    extended = base.with_symmetry(extra)
    assert base.symmetry("parity") is None
    assert extended.symmetry("parity") == extra
    assert extended.symmetry("gauss") is not None
    assert StructurePattern(required_symmetries=frozenset({"parity"})).matches(extended)
    assert not StructurePattern(required_symmetries=frozenset({"parity"})).matches(base)


def test_the_graph_exposes_edges_in_both_directions_and_allows_a_node_swap() -> None:
    """The graph exposes incoming and outgoing edges and replaces a node by name."""
    graph = build_graph(3).graph
    sources = {edge.source for edge in graph.incoming("qubit_register")}
    assert sources == {"quantum_link_homogeneous"}
    targets = {edge.target for edge in graph.outgoing("quantum_link_homogeneous")}
    assert targets == {"effective_bosonic", "qubit_register"}
    assert graph.incoming("lattice_qed") == ()

    bound = graph.with_binding(
        "quantum_link_staggered",
        **{P.MASS_QUANTUM_LINK_STAGGERED: 0.3, P.COUPLING_QUANTUM_LINK_STAGGERED: 0.7},
    )
    assert bound.binding[P.COUPLING_QUANTUM_LINK_STAGGERED] == pytest.approx(0.7)
    # The node itself is untouched until it is replaced.
    untouched = graph.node("quantum_link_staggered")
    assert untouched.binding.get(P.COUPLING_QUANTUM_LINK_STAGGERED) is None
    graph.replace_node(bound)
    replaced = graph.node("quantum_link_staggered")
    assert replaced.binding[P.COUPLING_QUANTUM_LINK_STAGGERED] == pytest.approx(0.7)

    with pytest.raises(KeyError, match="has no node named"):
        graph.replace_node(qubit_structure_node())


def test_a_product_formula_can_be_retuned_in_steps_and_in_order() -> None:
    """``with_steps`` and ``with_schedule`` retune a product formula."""
    model = qubit_model(3)
    base = as_product_formula(trotterisation(time=2.0, steps=4, order=2).apply(model))
    assert base.order == 2
    assert base.steps == 4
    assert base.step_size == pytest.approx(0.5)

    more_steps = base.with_steps(16)
    assert more_steps.steps == 16
    assert more_steps.order == 2
    assert more_steps.error_bound().value < base.error_bound().value

    higher_order = base.with_schedule(suzuki(4, len(base.layers)))
    assert higher_order.order == 4
    assert higher_order.formula.is_symmetric
    assert not suzuki(1, 3).is_symmetric
    assert len(higher_order.step_factors()) > len(base.step_factors())
    assert max_abs_deviation(higher_order.matrix(), higher_order.step_matrix()) > 0.0


def test_the_mass_layer_is_diagonal_and_the_coupling_layers_are_not() -> None:
    """``H_M`` is the only diagonal layer, and unknown layer names are refused."""
    layers = interleaved_layers_of(qubit_model(4))
    diagonal = {layer.name: layer.is_diagonal() for layer in layers.layers}
    assert diagonal == {"H_M": True, "H_even": False, "H_odd": False}
    assert layers.layer_named("H_M").term_count == 4
    with pytest.raises(KeyError, match="no layer named"):
        layers.layer_named("H_nowhere")


def test_a_pauli_sum_reports_its_locality_histogram() -> None:
    """Each layer's weight histogram and identity coefficient take their pinned values."""
    even = interleaved_layers_of(qubit_model(4))
    # H_M is N single-qubit Z strings plus an identity string carrying
    # m*N/2 - m*floor(N/2), which vanishes at even N.
    assert even.layer_named("H_M").operator.weight_histogram() == {1: 4}
    assert even.layer_named("H_M").operator.identity_coefficient() == pytest.approx(0.0)
    # Each even link contributes four 3-local strings; at N = 4 there are two even links.
    assert even.layer_named("H_even").operator.weight_histogram() == {3: 8}

    odd = interleaved_layers_of(qubit_model(3))
    assert odd.layer_named("H_M").operator.weight_histogram() == {0: 1, 1: 3}
    assert odd.layer_named("H_M").operator.identity_coefficient() == pytest.approx(
        REFERENCE_MASS / 2.0
    )


def qubit_structure_node() -> HamiltonianModel:
    """A node absent from the running example's graph."""
    return HamiltonianModel(
        name="nowhere",
        level=AbstractionLevel.INTERMEDIATE,
        structure=matter_gauge_structure(2, Algebra.QUBIT, Algebra.QUBIT, name="elsewhere"),
    )


# ---------------------------------------------------------------------------
# The staggered Gauss sector on an open chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matter_sites", [3, 4])
def test_the_staggered_boundary_generators_take_their_particle_hole_image(
    matter_sites: int,
) -> None:
    """The staggered boundary generators target ``-1/2`` and ``(-1)**N / 2``, never zero."""
    staggered = staggered_quantum_link(matter_sites)
    homogeneous = homogeneous_quantum_link(matter_sites)
    space = HilbertSpace.of(staggered.structure)

    for constraint in staggered.constraint_operators:
        generator = build_operator(constraint.operator, space, {})
        eigenvalues = set(jnp.round(jnp.real(jnp.diagonal(generator)), 6).tolist())
        assert constraint.target_value in eigenvalues
        if constraint.is_boundary:
            assert 0.0 not in eigenvalues
    targets = [constraint.target_value for constraint in staggered.constraint_operators]
    assert targets[0] == -0.5
    assert targets[-1] == (-1) ** matter_sites * 0.5
    assert all(target == 0.0 for target in targets[1:-1])

    # Particle-hole image of the canonical state.
    configuration = list(canonical_state_configuration(matter_sites))
    for site in range(1, matter_sites, 2):
        configuration[2 * site] = 1 - configuration[2 * site]
    for link in range(0, matter_sites - 1, 2):
        configuration[2 * link + 1] = 1 - configuration[2 * link + 1]
    image = space.basis_state(configuration)
    projector = sector_projector(staggered.constraint_operators, space, {})
    assert float(jnp.vdot(image, projector @ image).real) == pytest.approx(1.0)

    # Both sectors have the same dimension.
    other = sector_projector(homogeneous.constraint_operators, space, {})
    assert int(jnp.round(jnp.real(jnp.trace(projector)))) == int(
        jnp.round(jnp.real(jnp.trace(other)))
    )


def test_the_truncation_records_the_electric_constant_it_drops() -> None:
    """``(a/2) sum_l E^2`` becomes ``(N-1) a e^2 / 8`` and is recorded on the target."""
    matter_sites = 3
    bound = theory(matter_sites).bind(
        **{
            P.MASS_LATTICE_QED: 0.1,
            P.LATTICE_SPACING: 1.0,
            P.GAUGE_COUPLING: 1.0,
            P.ELECTRIC_GAP: 0.5,
        }
    )
    step = build_graph(matter_sites).analogue.steps[0]
    truncated = step.apply(bound)
    assert isinstance(truncated, HamiltonianModel)
    assert len(truncated.dropped_constants) == 1
    assert truncated.total_dropped_constant() == pytest.approx((matter_sites - 1) / 8)
    assert step.dropped_constant in str(truncated.dropped_constants[0])

    # The constant travels down the pipeline unchanged.
    downstream = build_graph(matter_sites).analogue.sub_pipeline(1, 3).apply(truncated)
    assert isinstance(downstream, HamiltonianModel)
    assert downstream.dropped_constants == truncated.dropped_constants


def test_a_strict_bound_excludes_its_endpoint() -> None:
    """``(0, 1]`` refuses ``0`` and admits ``1``; the admissible set reports the refusal."""
    bound = Bound("J", 0.0, 1.0, strict_lower=True)
    assert not bound.contains(0.0)
    assert bound.contains(1.0)
    assert bound.contains(1e-12)
    assert bound.violation(0.0) == 0.0
    assert not Bound("J", 0.0, 1.0, strict_upper=True).contains(1.0)
    assert Bound("J", 0.0, 1.0).contains(0.0)
    device = AdmissibleSet(bounds=(bound,), description="a device")
    assert not device.admits({"J": 0.0})
    assert device.violations({"J": 0.0})[0].parameter == "J"
    assert device.admits({"J": 0.5})


def test_an_admissible_set_bounds_each_knob_once() -> None:
    """A union intersects bounds on a shared knob, and duplicate bounds are refused."""
    joined = box(J=(0.0, 1.0)).union(box(J=(0.5, 2.0), U=(1.0, 3.0)))
    assert joined.bound_for("J") == Bound("J", 0.5, 1.0)
    assert joined.interval_environment()["J"] == Interval(0.5, 1.0)
    assert joined.bound_for("U") == Bound("U", 1.0, 3.0)
    strict = Bound("J", 0.0, 1.0, strict_lower=True).intersect(Bound("J", 0.0, 0.5))
    assert strict == Bound("J", 0.0, 0.5, strict_lower=True)
    with pytest.raises(ValueError, match="more than once"):
        AdmissibleSet(bounds=(Bound("J", 0.0, 1.0), Bound("J", 0.0, 2.0)))
    with pytest.raises(ValueError, match="empty bound"):
        Bound("J", 0.0, 1.0).intersect(Bound("J", 2.0, 3.0))


def test_a_non_finite_constant_renders() -> None:
    """Infinite and undefined constants render rather than overflow."""
    assert str(Const(math.inf)) == "inf"
    assert str(Const(-math.inf)) == "-inf"
    assert str(Const(math.nan)) == "nan"
    assert str(Symbol("x") ** math.inf) == "x**inf"
    assert Interval(0.5, 2.0) ** math.inf == Interval(0.0, math.inf)


def test_a_model_with_an_illegal_operator_is_refused_at_construction() -> None:
    """A Hamiltonian model validates its operators when it is built, not on request."""
    model = homogeneous_quantum_link(3)
    with pytest.raises(ValueError, match="not a legal operator on a spin-1/2 site"):
        dataclasses.replace(model, hamiltonian=model.hamiltonian + word(1.0, pauli_x(1)))


def test_the_normal_form_carries_the_algebra_of_each_site() -> None:
    """Same-site products reduce, different-site fermions anticommute, constants fold."""
    algebra_at = {0: Algebra.FERMION, 1: Algebra.FERMION, 2: Algebra.QUBIT, 3: Algebra.SPIN_HALF}
    m = Symbol("m")
    assert equivalent(word(1, create(0), annihilate(0)), word(1, number(0)), algebra_at)
    assert equivalent(
        word(1, annihilate(0), create(0)), constant_word(1) + word(-1, number(0)), algebra_at
    )
    parity = constant_word(1) + word(-2, number(0))
    assert equivalent(product(parity, parity), constant_word(1), algebra_at)
    assert equivalent(
        word(1, annihilate(1), annihilate(0)), word(-1, annihilate(0), annihilate(1)), algebra_at
    )
    assert equivalent(word(1, sigma_plus(2), sigma_minus(2)), word(1, number(2)), algebra_at)
    assert equivalent(
        word(m, spin_plus(3), spin_minus(3)),
        word(m / 2) + word(m, spin_z(3)),
        algebra_at,
    )
    assert equivalent(word(2 * m, number(0)), word(m + m, number(0)), algebra_at)
    assert not equivalent(word(2 * m, number(0)), word(m, number(0)), algebra_at)
    assert str(difference(word(2 * m, number(0)), word(m, number(0)), algebra_at)).count("n_0") == 1


def test_the_normal_form_normal_orders_bosons() -> None:
    """``a a^dag = a^dag a + 1`` and ``n^2 = a^dag a + a^dag a^dag a a``."""
    algebra_at = {0: Algebra.BOSON}
    assert equivalent(
        word(1, annihilate(0), create(0)),
        word(1, create(0), annihilate(0)) + constant_word(1),
        algebra_at,
    )
    squared = normal_form(word(1, number(0), number(0)), algebra_at)
    assert [term.operators for term in squared.terms] == [
        (create(0), annihilate(0)),
        (create(0), create(0), annihilate(0), annihilate(0)),
    ]
