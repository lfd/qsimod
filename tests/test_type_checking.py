"""Structural type checking of transformations and pipelines, without numerics."""

from __future__ import annotations

import time

import pytest

from qsimod.artifact import ArtifactKind
from qsimod.pipeline import CompositionError, Pipeline
from qsimod.structure import (
    Algebra,
    DegreeOfFreedom,
    Lattice,
    LatticeGeometry,
    StructureType,
    StructureTypeError,
    SymmetryDeclaration,
)
from qsimod.transform import ApproximationKind, ArtifactKindError, Exactness
from qsimod.usecases.schwinger import (
    ANALOGUE_TARGET,
    DIGITAL_TARGET,
    SOURCE,
    SchwingerGraph,
    effective_bosonic,
    encoding,
    homogeneous_quantum_link,
    particle_hole,
    perturbation,
    to_qubits,
    trotterisation,
    truncation,
)
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)

#: A chain length whose Hilbert space (dimension 2**399) cannot be allocated.
INFEASIBLE_SITES = 200
#: Time budget, in seconds, for a purely structural operation.
STRUCTURAL_TIME_BUDGET = 2.0


def test_analogue_pipeline_type_checks_and_is_regime_limited(graph3: SchwingerGraph) -> None:
    """The analogue pipeline composes, is APPROXIMATE, and has two regime-limited steps."""
    pipeline = graph3.analogue
    assert [step.name for step in pipeline.steps] == [
        "spin-1/2 quantum-link truncation",
        "particle-hole transformation",
        "Jordan-Wigner + hardcore-boson encoding",
        "second-order degenerate perturbation theory, inverted",
    ]
    assert pipeline.exactness is Exactness.APPROXIMATE
    assert pipeline.approximation_kinds == frozenset({ApproximationKind.REGIME_LIMITED})
    kinds = [step.approximation_kind for step in pipeline.approximate_steps]
    assert kinds == [ApproximationKind.REGIME_LIMITED, ApproximationKind.REGIME_LIMITED]


def test_digital_pipeline_type_checks_and_has_a_resource_controlled_step(
    graph4: SchwingerGraph,
) -> None:
    """The digital pipeline composes and contains a resource-controlled step."""
    pipeline = graph4.digital
    assert pipeline.exactness is Exactness.APPROXIMATE
    assert ApproximationKind.RESOURCE_CONTROLLED in pipeline.approximation_kinds
    assert ApproximationKind.REGIME_LIMITED in pipeline.approximation_kinds


def test_both_pipelines_are_enumerable_from_the_shared_node(graph4: SchwingerGraph) -> None:
    """Both branches leave the branch point, and each target is reached by exactly one pipeline."""
    outgoing = {edge.target for edge in graph4.graph.outgoing("quantum_link_homogeneous")}
    assert outgoing == {"effective_bosonic", "qubit_register"}

    reachable = set(graph4.graph.reachable(SOURCE))
    assert {ANALOGUE_TARGET, DIGITAL_TARGET} <= reachable

    targets = set(graph4.graph.terminal_targets(SOURCE))
    assert {ANALOGUE_TARGET, DIGITAL_TARGET} <= targets

    analogue_paths = graph4.graph.pipelines(SOURCE, ANALOGUE_TARGET)
    digital_paths = graph4.graph.pipelines(SOURCE, DIGITAL_TARGET)
    assert len(analogue_paths) == 1
    assert len(digital_paths) == 1

    branches = {summary.target: summary for summary in graph4.graph.branches(SOURCE)}
    assert branches[ANALOGUE_TARGET].artifact_kind is ArtifactKind.HAMILTONIAN
    assert branches[DIGITAL_TARGET].artifact_kind is ArtifactKind.PRODUCT_FORMULA


def test_a_hamiltonian_step_rejects_a_product_formula(graph4: SchwingerGraph) -> None:
    """A Hamiltonian-consuming step raises ``ArtifactKindError`` on a product formula."""
    staggered = graph4.graph.node("quantum_link_staggered").bind(
        **{P.MASS_QUANTUM_LINK_STAGGERED: 0.41, P.COUPLING_QUANTUM_LINK_STAGGERED: 0.83}
    )
    qubits = to_qubits().apply(particle_hole().apply(staggered))
    product_formula = trotterisation(time=1.0, steps=2, order=2).apply(qubits)
    assert product_formula.kind is ArtifactKind.PRODUCT_FORMULA

    for step in (trotterisation(time=1.0, steps=2, order=2), to_qubits()):
        with pytest.raises(ArtifactKindError) as caught:
            step.apply(product_formula)
        assert caught.value.expected is ArtifactKind.HAMILTONIAN
        assert caught.value.found is ArtifactKind.PRODUCT_FORMULA
        assert "not interchangeable" in str(caught.value)


def test_exact_prefixes_report_exact() -> None:
    """The shared prefixes of both branches after the truncation are EXACT."""
    analogue_prefix = Pipeline.of([particle_hole(), encoding()], name="analogue prefix")
    digital_prefix = Pipeline.of([particle_hole(), to_qubits()], name="digital prefix")
    for prefix in (analogue_prefix, digital_prefix):
        assert prefix.exactness is Exactness.EXACT
        assert prefix.approximation is None
        assert prefix.approximation_kinds == frozenset()


def test_wrong_algebra_raises_a_typed_error_naming_the_mismatch() -> None:
    """Bosonic matter is rejected by the Jordan-Wigner step, naming the algebra mismatch."""
    bosonic = effective_bosonic(INFEASIBLE_SITES)
    started = time.perf_counter()
    with pytest.raises(StructureTypeError) as caught:
        to_qubits().apply(bosonic)
    elapsed = time.perf_counter() - started

    assert elapsed < STRUCTURAL_TIME_BUDGET
    aspects = {mismatch.aspect for mismatch in caught.value.mismatches}
    assert "dof[matter].algebra" in aspects
    mismatch = next(m for m in caught.value.mismatches if m.aspect == "dof[matter].algebra")
    assert mismatch.required == "fermion"
    assert mismatch.found == "boson"
    assert "incompatible" not in str(caught.value).lower()


def test_wrong_lattice_raises_a_typed_error_naming_the_geometry() -> None:
    """A 2D lattice is rejected by a 1D-chain transformation, naming the geometry."""
    two_dimensional = StructureType(
        lattice=Lattice(LatticeGeometry.OPEN_SQUARE_2D, INFEASIBLE_SITES),
        degrees_of_freedom=(
            DegreeOfFreedom("matter", Algebra.FERMION, tuple(range(0, 2 * INFEASIBLE_SITES, 2))),
            DegreeOfFreedom("gauge", Algebra.SPIN_HALF, tuple(range(1, 2 * INFEASIBLE_SITES, 2))),
        ),
        symmetries=frozenset({SymmetryDeclaration("gauss", "U(1)")}),
        name="a 2D lattice",
    )
    model = homogeneous_quantum_link(3)
    wrong = type(model)(
        name="2D model",
        level=model.level,
        structure=two_dimensional,
        parameters=model.parameters,
        hamiltonian=model.hamiltonian,
    )
    started = time.perf_counter()
    with pytest.raises(StructureTypeError) as caught:
        to_qubits().apply(wrong)
    assert time.perf_counter() - started < STRUCTURAL_TIME_BUDGET
    aspects = {mismatch.aspect for mismatch in caught.value.mismatches}
    assert "lattice.geometry" in aspects


def test_missing_declared_symmetry_is_named() -> None:
    """A model without the declared gauge symmetry is rejected, naming the symmetry."""
    model = homogeneous_quantum_link(INFEASIBLE_SITES)
    without = type(model)(
        name="no declared symmetry",
        level=model.level,
        structure=StructureType(
            lattice=model.structure.lattice,
            degrees_of_freedom=model.structure.degrees_of_freedom,
            symmetries=frozenset(),
            name=model.structure.name,
        ),
        parameters=model.parameters,
        hamiltonian=model.hamiltonian,
    )
    started = time.perf_counter()
    with pytest.raises(StructureTypeError) as caught:
        to_qubits().apply(without)
    assert time.perf_counter() - started < STRUCTURAL_TIME_BUDGET
    assert any(m.aspect == "symmetry[gauss]" for m in caught.value.mismatches)


def test_incompatible_composition_raises_at_composition_time() -> None:
    """Composing steps whose structural types do not meet raises ``CompositionError``."""
    with pytest.raises(CompositionError) as caught:
        Pipeline.of([encoding(), truncation()])
    assert caught.value.first.startswith("Jordan-Wigner + hardcore-boson encoding")
    assert caught.value.second.startswith("spin-1/2 quantum-link truncation")
    aspects = {gap.aspect for gap in caught.value.gaps}
    assert "dof[matter].algebra" in aspects or "dof[gauge].algebra" in aspects


def test_incompatible_artifact_kinds_raise_at_composition_time() -> None:
    """A Hamiltonian-consuming step after a product-formula step fails to compose."""
    with pytest.raises(CompositionError) as caught:
        Pipeline.of([trotterisation(time=1.0, steps=1, order=1), perturbation()])
    assert "product formula" in caught.value.kind_problem
    assert "Hamiltonian" in caught.value.kind_problem


def test_a_valid_composition_still_composes(graph3: SchwingerGraph) -> None:
    """The Schwinger pipelines compose, and a pipeline of sub-pipelines composes too."""
    assert len(graph3.analogue.steps) == 4
    assert len(graph3.digital.steps) == 4
    assert len(graph3.alternative_digital.steps) == 3
    prefix = graph3.analogue.sub_pipeline(0, 2)
    suffix = graph3.analogue.sub_pipeline(2, None)
    combined = Pipeline.of([prefix, suffix], name="reassembled")
    assert combined.exactness is graph3.analogue.exactness
    assert combined.approximation_kinds == graph3.analogue.approximation_kinds
