"""Adding a hardware target and a theory from outside the package, via the public API only."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from qsimod.artifact import Artifact, HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.parameters import (
    AdmissibleSet,
    Bound,
    Parameter,
    ParameterSet,
)
from qsimod.pipeline import Pipeline
from qsimod.relations import Definition, ParameterRelation, equation
from qsimod.scalar import Symbol
from qsimod.solving import SolveStatus, realise_parameters
from qsimod.structure import (
    Algebra,
    DegreeOfFreedom,
    DofRequirement,
    Lattice,
    LatticeGeometry,
    StructurePattern,
    StructureType,
    StructureTypeError,
    SymmetryDeclaration,
)
from qsimod.symbolic import OperatorSum, pauli_x, pauli_z, word
from qsimod.transform import Exactness, Transformation
from qsimod.units import Dimension
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import (
    build_graph,
    particle_hole,
    to_qubits,
    truncation,
)

# ---------------------------------------------------------------------------
# A toy third hardware target: a transverse-field Ising chain
# ---------------------------------------------------------------------------

ISING_COUPLING = "ising.Jz"
ISING_FIELD = "ising.h"

ISING_PARAMETERS = ParameterSet(
    (
        Parameter(ISING_COUPLING, Dimension.ENERGY, "nearest-neighbour ZZ coupling"),
        Parameter(ISING_FIELD, Dimension.ENERGY, "transverse field"),
    )
)

#: The source type the toy transformation demands, written with the core API.
QUBIT_SOURCE_PATTERN = StructurePattern(
    geometries=frozenset({LatticeGeometry.OPEN_CHAIN_1D}),
    dofs=(
        DofRequirement("matter", frozenset({Algebra.QUBIT}), min_sites=2),
        DofRequirement("gauge", frozenset({Algebra.QUBIT}), min_sites=1),
    ),
    required_symmetries=frozenset({"gauss"}),
    description="an interleaved qubit register carrying a declared local U(1)",
)

ISING_PATTERN = StructurePattern(
    geometries=frozenset({LatticeGeometry.OPEN_CHAIN_1D}),
    dofs=(DofRequirement("spins", frozenset({Algebra.QUBIT}), min_sites=2),),
    description="a transverse-field Ising chain of qubits",
)


def ising_structure(sites: int) -> StructureType:
    """The toy target's structural type: one qubit family, no gauge sector."""
    return StructureType(
        lattice=Lattice(LatticeGeometry.OPEN_CHAIN_1D, sites),
        degrees_of_freedom=(DegreeOfFreedom("spins", Algebra.QUBIT, tuple(range(2 * sites - 1))),),
        symmetries=frozenset({SymmetryDeclaration("ising_z2", "Z(2)", local=False)}),
        name="H_TFIM",
    )


def ising_admissible_set() -> AdmissibleSet:
    """The toy device's declared knob limits."""
    return AdmissibleSet(
        bounds=(
            Bound(ISING_COUPLING, 0.0, 5.0, strict_lower=True),
            Bound(ISING_FIELD, 0.0, 2.0),
        ),
        description="toy transverse-field Ising simulator",
    )


def ising_model(sites: int) -> HamiltonianModel:
    """The toy hardware model, built from the public symbolic API."""
    coupling = Symbol(ISING_COUPLING)
    field = Symbol(ISING_FIELD)
    hamiltonian = OperatorSum()
    register = 2 * sites - 1
    for site in range(register - 1):
        hamiltonian = hamiltonian + word(coupling, pauli_z(site), pauli_z(site + 1))
    for site in range(register):
        hamiltonian = hamiltonian + word(field, pauli_x(site))
    return HamiltonianModel(
        name="transverse_ising",
        level=AbstractionLevel.HARDWARE,
        structure=ising_structure(sites),
        parameters=ISING_PARAMETERS,
        admissible_set=ising_admissible_set(),
        hamiltonian=hamiltonian.renamed("H_TFIM"),
        origin="a toy third hardware target, added from outside the package",
    )


@dataclass(frozen=True)
class IsingRealisation(Transformation):
    """A toy transformation from the qubit level into the Ising target."""

    def target_structure(self, source: StructureType) -> StructureType:
        """The Ising structure over the source's register."""
        return ising_structure(source.lattice.matter_sites)

    def _transform(self, artifact: Artifact) -> Artifact:
        """Build the toy target and carry the parameters over."""
        target = ising_model(artifact.structure.lattice.matter_sites)
        known = artifact.binding.as_dict()
        values = {
            definition.parameter: definition.expression.evaluate_real(known)
            for definition in self.relation.definitions
            if definition.parameter in target.parameters.names and definition.inputs() <= set(known)
        }
        return target.bind_all(values)


def ising_realisation() -> IsingRealisation:
    """The toy transformation with its parameter relation and structural types."""
    coupling_in = Symbol(P.COUPLING_QUBIT_REGISTER)
    mass_in = Symbol(P.MASS_QUBIT_REGISTER)
    coupling_out = Symbol(ISING_COUPLING)
    field_out = Symbol(ISING_FIELD)
    relation = ParameterRelation(
        name="toy Ising realisation",
        equations=(
            equation("coupling", coupling_out, coupling_in / 2, "a made-up rescaling"),
            equation("field", field_out, mass_in, "the mass becomes the field"),
        ),
        definitions=(
            Definition(ISING_COUPLING, coupling_in / 2, "coupling"),
            Definition(P.COUPLING_QUBIT_REGISTER, 2 * coupling_out, "coupling"),
            Definition(ISING_FIELD, mass_in, "field"),
            Definition(P.MASS_QUBIT_REGISTER, field_out, "field"),
        ),
    )
    return IsingRealisation(
        name="(toy) qubit Hamiltonian -> transverse-field Ising chain",
        source_pattern=QUBIT_SOURCE_PATTERN,
        target_pattern=ISING_PATTERN,
        exactness=Exactness.EXACT,
        relation=relation,
        source_parameters=(P.MASS_QUBIT_REGISTER, P.COUPLING_QUBIT_REGISTER),
        target_parameters=(ISING_COUPLING, ISING_FIELD),
        description="a trivial third target, added from outside the package",
    )


def test_a_toy_hardware_target_is_added_from_outside_and_type_checks() -> None:
    """A hardware model and transformation added to the DAG become an enumerable pipeline."""
    graph = build_graph(3)
    graph.graph.add_node(ising_model(3))
    edge = graph.graph.add_edge("qubit_register", "transverse_ising", ising_realisation())
    assert edge.target == "transverse_ising"

    targets = set(graph.graph.terminal_targets("lattice_qed"))
    assert "transverse_ising" in targets
    pipelines = graph.graph.pipelines("lattice_qed", "transverse_ising")
    assert len(pipelines) == 1
    pipeline = pipelines[0]
    assert pipeline.exactness is Exactness.APPROXIMATE  # from the QLM truncation
    assert [step.name for step in pipeline.steps][:3] == [
        "spin-1/2 quantum-link truncation",
        "particle-hole transformation",
        "Jordan-Wigner to qubits",
    ]


def test_the_toy_target_can_be_applied_and_solved_for() -> None:
    """The added target can be applied to and solved for its knobs."""
    pipeline = Pipeline.of(
        [
            truncation(),
            particle_hole(),
            to_qubits(),
            ising_realisation(),
        ],
        name="lattice_qed -> ... -> toy Ising",
    )
    staggered = (
        build_graph(3)
        .graph.node("quantum_link_staggered")
        .bind(**{P.MASS_QUANTUM_LINK_STAGGERED: 0.4, P.COUPLING_QUANTUM_LINK_STAGGERED: 0.8})
    )
    applied = pipeline.sub_pipeline(1, None).apply(staggered)
    assert isinstance(applied, HamiltonianModel)
    assert applied.name == "transverse_ising"
    assert applied.binding[ISING_COUPLING] == pytest.approx(0.4)
    assert applied.binding[ISING_FIELD] == pytest.approx(0.4)

    result = realise_parameters(
        pipeline,
        targets={
            P.MASS_LATTICE_QED: 0.4,
            P.COUPLING_QUANTUM_LINK_STAGGERED: 0.8,
            P.ELECTRIC_GAP: 1.0,
        },
        unknowns=[ISING_COUPLING, ISING_FIELD],
        admissible_set=ising_admissible_set(),
    )
    assert result.status is SolveStatus.EXACT_SOLUTION
    assert result.point[ISING_COUPLING] == pytest.approx(0.4, abs=1e-8)
    assert result.point[ISING_FIELD] == pytest.approx(0.4, abs=1e-8)


def test_a_second_application_level_theory_is_rejected_where_it_lacks_structure() -> None:
    """A theory without a gauge sector is rejected by the truncation's pattern check."""
    sites = 200
    scalar_theory = HamiltonianModel(
        name="lattice_qed'",
        level=AbstractionLevel.APPLICATION,
        structure=StructureType(
            lattice=Lattice(LatticeGeometry.OPEN_CHAIN_1D, sites),
            degrees_of_freedom=(
                DegreeOfFreedom("matter", Algebra.FERMION, tuple(range(0, 2 * sites, 2))),
            ),
            name="a free fermion chain with no gauge field",
        ),
        parameters=ParameterSet((Parameter("lattice_qed'.m", Dimension.ENERGY, "mass"),)),
        hamiltonian=OperatorSum(),
        origin="a second application-level theory, added from outside the package",
    )

    with pytest.raises(StructureTypeError) as caught:
        truncation().apply(scalar_theory)
    aspects = {mismatch.aspect for mismatch in caught.value.mismatches}
    assert "dof[gauge]" in aspects
    assert "symmetry[gauss]" in aspects
    gauge = next(m for m in caught.value.mismatches if m.aspect == "dof[gauge]")
    assert gauge.required == "present"
    assert "absent" in gauge.found


def test_the_new_theory_type_checks_against_a_transformation_that_fits_it() -> None:
    """A fermion chain matches a pattern that does not demand a gauge sector."""
    sites = 5
    pattern = StructurePattern(
        geometries=frozenset({LatticeGeometry.OPEN_CHAIN_1D}),
        dofs=(DofRequirement("matter", frozenset({Algebra.FERMION}), min_sites=2),),
        description="any fermionic 1D chain",
    )
    structure = StructureType(
        lattice=Lattice(LatticeGeometry.OPEN_CHAIN_1D, sites),
        degrees_of_freedom=(
            DegreeOfFreedom("matter", Algebra.FERMION, tuple(range(0, 2 * sites, 2))),
        ),
        name="a free fermion chain",
    )
    assert pattern.matches(structure)
    assert pattern.mismatches(structure) == []
