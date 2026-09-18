"""The case study of the article: the lattice Schwinger model, assembled from the libraries.

The graph runs from ``lattice_qed`` (``H_sys``) through ``quantum_link_staggered`` (``H_IR1``)
to the branch point ``quantum_link_homogeneous`` (``H_IR2``).  The analogue branch continues
through ``effective_bosonic`` (``H_IR3``) to the Bose-Hubbard chain ``bose_hubbard``
(``H_sim``); the digital branch through ``qubit_register`` (``H_IR4``) to the Trotter product
formula ``trotter`` (``U_approx``).  A second digital branch leaves ``quantum_link_staggered``
directly, via ``qubit_register_staggered`` to ``trotter_staggered``.  The guide page of the
documentation gives the Hamiltonians, the transformations and the conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

from qsimod.artifact import HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models import names
from qsimod.models.application import kogut_susskind_gauge_theory
from qsimod.models.gauge import (
    canonical_state_configuration,
    gauge_violation_observable,
    local_occupation_subspace,
    matter_occupation_observable,
)
from qsimod.models.hardware import (
    DEFAULT_INTERACTION_RANGE,
    DEFAULT_SUPERLATTICE_MAX,
    DEFAULT_TILT_MAX,
    DEFAULT_TUNNELLING_MAX,
    bose_hubbard_admissible_set,
    tilted_bose_hubbard_chain,
)
from qsimod.models.intermediate import (
    HOMOGENEOUS_CONVENTION,
    STAGGERED_CONVENTION,
    bosonic_pair_coupling_model,
    k_local_qubit_model,
    quantum_link_model,
)
from qsimod.parameters import AdmissibleSet, Namespace
from qsimod.pauli import PauliString, PauliSum
from qsimod.pipeline import ModelGraph, Pipeline
from qsimod.scalar import Scalar
from qsimod.transformations import (
    HardcoreBosonEncoding,
    JordanWignerToQubits,
    ParticleHoleTransformation,
    QuantumLinkTruncation,
    SecondOrderPerturbationTheory,
    SuzukiTrotter,
    hardcore_boson_encoding,
    jordan_wigner_to_qubits,
    particle_hole_transformation,
    quantum_link_truncation,
    second_order_coupling,
    second_order_mass,
    second_order_perturbation_theory,
    staggering_constant,
    superlattice_validity,
    suzuki_trotter,
)
from qsimod.trotter import ProductFormulaModel, declared_layers, lie_trotter
from qsimod.usecases.base import UseCaseGraph
from qsimod.validity import Conjunction

__all__ = [
    "ALTERNATIVE_DIGITAL_TARGET",
    "ANALOGUE_TARGET",
    "DIGITAL_TARGET",
    "NAMESPACES",
    "SOURCE",
    "ParameterNames",
    "SchwingerGraph",
    "analogue_pipeline",
    "build_graph",
    "canonical_state_configuration",
    "coupling_map",
    "device_limits",
    "digital_pipeline",
    "effective_bosonic",
    "encoding",
    "gauge_violation_observable",
    "homogeneous_quantum_link",
    "local_occupation_subspace",
    "mass_map",
    "matter_occupation_observable",
    "particle_hole",
    "perturbation",
    "qubit_register",
    "qubit_register_from_staggered",
    "staggered_quantum_link",
    "superlattice",
    "theory",
    "to_qubits",
    "to_qubits_from_staggered",
    "trotterisation",
    "truncation",
    "validity",
]

# ---------------------------------------------------------------------------
# The namespaces this use case gives its models
# ---------------------------------------------------------------------------

LATTICE_QED = Namespace("lattice_qed")
QUANTUM_LINK_STAGGERED = Namespace("quantum_link_staggered")
QUANTUM_LINK_HOMOGENEOUS = Namespace("quantum_link_homogeneous")
EFFECTIVE_BOSONIC = Namespace("effective_bosonic")
QUBIT_REGISTER = Namespace("qubit_register")
QUBIT_REGISTER_STAGGERED = Namespace("qubit_register_staggered")
BOSE_HUBBARD = Namespace("bose_hubbard")
TROTTER = Namespace("trotter")
TROTTER_STAGGERED = Namespace("trotter_staggered")

#: Every namespace this use case owns, in pipeline order.
NAMESPACES = (
    LATTICE_QED,
    QUANTUM_LINK_STAGGERED,
    QUANTUM_LINK_HOMOGENEOUS,
    EFFECTIVE_BOSONIC,
    QUBIT_REGISTER,
    QUBIT_REGISTER_STAGGERED,
    BOSE_HUBBARD,
    TROTTER,
    TROTTER_STAGGERED,
)

SOURCE = "lattice_qed"
ANALOGUE_TARGET = "bose_hubbard"
DIGITAL_TARGET = "trotter"
ALTERNATIVE_DIGITAL_TARGET = "trotter_staggered"


class ParameterNames:
    """The fully qualified parameter names of this use case."""

    MASS_LATTICE_QED = LATTICE_QED(names.MASS)
    LATTICE_SPACING = LATTICE_QED(names.LATTICE_SPACING)
    GAUGE_COUPLING = LATTICE_QED(names.GAUGE_COUPLING)
    ELECTRIC_GAP = LATTICE_QED(names.ELECTRIC_GAP)

    MASS_QUANTUM_LINK_STAGGERED = QUANTUM_LINK_STAGGERED(names.MASS)
    COUPLING_QUANTUM_LINK_STAGGERED = QUANTUM_LINK_STAGGERED(names.COUPLING)
    MASS_QUANTUM_LINK_HOMOGENEOUS = QUANTUM_LINK_HOMOGENEOUS(names.MASS)
    COUPLING_QUANTUM_LINK_HOMOGENEOUS = QUANTUM_LINK_HOMOGENEOUS(names.COUPLING)
    MASS_EFFECTIVE_BOSONIC = EFFECTIVE_BOSONIC(names.MASS)
    COUPLING_EFFECTIVE_BOSONIC = EFFECTIVE_BOSONIC(names.COUPLING)
    MASS_QUBIT_REGISTER = QUBIT_REGISTER(names.MASS)
    COUPLING_QUBIT_REGISTER = QUBIT_REGISTER(names.COUPLING)
    MASS_QUBIT_REGISTER_STAGGERED = QUBIT_REGISTER_STAGGERED(names.MASS)
    COUPLING_QUBIT_REGISTER_STAGGERED = QUBIT_REGISTER_STAGGERED(names.COUPLING)

    TUNNELLING = BOSE_HUBBARD(names.TUNNELLING)
    INTERACTION = BOSE_HUBBARD(names.INTERACTION)
    SUPERLATTICE = BOSE_HUBBARD(names.SUPERLATTICE)
    TILT = BOSE_HUBBARD(names.TILT)

    MASS_TROTTER = TROTTER(names.MASS)
    COUPLING_TROTTER = TROTTER(names.COUPLING)


# ---------------------------------------------------------------------------
# The nodes
# ---------------------------------------------------------------------------


def theory(matter_sites: int) -> HamiltonianModel:
    """``lattice_qed``: the Kogut-Susskind lattice Schwinger model, ``H_sys``."""
    return kogut_susskind_gauge_theory(
        matter_sites, LATTICE_QED, name="lattice_qed", hamiltonian_name="H_QED"
    )


def staggered_quantum_link(matter_sites: int) -> HamiltonianModel:
    """``quantum_link_staggered``: the spin-1/2 quantum-link model, staggered, ``H_IR1``."""
    return quantum_link_model(
        matter_sites,
        QUANTUM_LINK_STAGGERED,
        STAGGERED_CONVENTION,
        name="quantum_link_staggered",
        hamiltonian_name="H_QLM^st",
    )


def homogeneous_quantum_link(matter_sites: int) -> HamiltonianModel:
    """``quantum_link_homogeneous``: the quantum-link model after particle-hole, ``H_IR2``.

    The model carries the constant ``-m * floor(N/2)`` the mass staggering left behind and is
    the branch point of the case study.
    """
    return quantum_link_model(
        matter_sites,
        QUANTUM_LINK_HOMOGENEOUS,
        HOMOGENEOUS_CONVENTION,
        name="quantum_link_homogeneous",
        hamiltonian_name="H_QLM",
        additive_constant=staggering_constant(
            matter_sites, QUANTUM_LINK_HOMOGENEOUS.symbol(names.MASS)
        ),
    )


def effective_bosonic(matter_sites: int) -> HamiltonianModel:
    """``effective_bosonic``: the effective bosonic pair-coupling model, ``H_IR3``.

    The model is the one realised by the superlattice.
    """
    return bosonic_pair_coupling_model(
        matter_sites,
        EFFECTIVE_BOSONIC,
        name="effective_bosonic",
        hamiltonian_name="H_eff",
        additive_constant=staggering_constant(matter_sites, EFFECTIVE_BOSONIC.symbol(names.MASS)),
    )


def superlattice(
    matter_sites: int,
    admissible_set: AdmissibleSet | None = None,
) -> HamiltonianModel:
    """``bose_hubbard``: the tilted, staggered Bose-Hubbard chain, the analogue ``H_sim``."""
    return tilted_bose_hubbard_chain(
        matter_sites,
        BOSE_HUBBARD,
        admissible_set=admissible_set,
        name="bose_hubbard",
        hamiltonian_name="H_BHM",
    )


def qubit_register(matter_sites: int) -> HamiltonianModel:
    """``qubit_register``: the 3-local Pauli Hamiltonian on the qubit register, ``H_IR4``."""
    return k_local_qubit_model(
        matter_sites,
        QUBIT_REGISTER,
        HOMOGENEOUS_CONVENTION,
        name="qubit_register",
        hamiltonian_name="H_qubit",
        additive_constant=staggering_constant(matter_sites, QUBIT_REGISTER.symbol(names.MASS)),
    )


def qubit_register_from_staggered(matter_sites: int) -> HamiltonianModel:
    """``qubit_register_staggered``: the qubit register reached from the staggered form.

    The branch point is ``quantum_link_staggered`` instead of ``quantum_link_homogeneous``.
    """
    return k_local_qubit_model(
        matter_sites,
        QUBIT_REGISTER_STAGGERED,
        STAGGERED_CONVENTION,
        name="qubit_register_staggered",
        hamiltonian_name="H_qubit^st",
    )


# ---------------------------------------------------------------------------
# The edges
# ---------------------------------------------------------------------------


def truncation() -> QuantumLinkTruncation:
    """The quantum-link truncation, ``lattice_qed -> quantum_link_staggered``."""
    return quantum_link_truncation(
        LATTICE_QED,
        QUANTUM_LINK_STAGGERED,
        target_name="quantum_link_staggered",
        name="spin-1/2 quantum-link truncation",
    )


def particle_hole() -> ParticleHoleTransformation:
    """The particle-hole transformation, ``quantum_link_staggered -> quantum_link_homogeneous``."""
    return particle_hole_transformation(
        QUANTUM_LINK_STAGGERED,
        QUANTUM_LINK_HOMOGENEOUS,
        target_name="quantum_link_homogeneous",
        name="particle-hole transformation",
    )


def encoding() -> HardcoreBosonEncoding:
    """The boson encoding, ``quantum_link_homogeneous -> effective_bosonic``."""
    return hardcore_boson_encoding(
        QUANTUM_LINK_HOMOGENEOUS,
        EFFECTIVE_BOSONIC,
        target_name="effective_bosonic",
        name="Jordan-Wigner + hardcore-boson encoding",
    )


def perturbation(
    admissible_set: AdmissibleSet | None = None,
) -> SecondOrderPerturbationTheory:
    """Second-order degenerate perturbation theory, ``effective_bosonic -> bose_hubbard``.

    The relation is solved for the knob settings ``Theta_sim = {J, U, delta, Delta}``.
    """
    return second_order_perturbation_theory(
        EFFECTIVE_BOSONIC,
        BOSE_HUBBARD,
        admissible_set=admissible_set or bose_hubbard_admissible_set(BOSE_HUBBARD),
        target_name="bose_hubbard",
        name="second-order degenerate perturbation theory, inverted",
    )


def to_qubits() -> JordanWignerToQubits:
    """The Jordan-Wigner transformation, ``quantum_link_homogeneous -> qubit_register``."""
    return jordan_wigner_to_qubits(
        QUANTUM_LINK_HOMOGENEOUS,
        QUBIT_REGISTER,
        target_name="qubit_register",
        name="Jordan-Wigner to qubits",
    )


def to_qubits_from_staggered() -> JordanWignerToQubits:
    """The Jordan-Wigner transformation from the staggered quantum-link model.

    The transformation is the edge ``quantum_link_staggered -> qubit_register_staggered``.
    """
    return jordan_wigner_to_qubits(
        QUANTUM_LINK_STAGGERED,
        QUBIT_REGISTER_STAGGERED,
        convention=STAGGERED_CONVENTION,
        retain_staggering_constant=False,
        target_name="qubit_register_staggered",
        name="Jordan-Wigner to qubits, from the staggered quantum-link model",
    )


def trotterisation(
    *,
    time: float = 1.0,
    steps: int = 1,
    order: int = 2,
    from_staggered: bool = False,
) -> SuzukiTrotter:
    """The Trotterisation, ``qubit_register -> trotter``.

    Args:
        time: the simulated time ``t``.
        steps: the step count ``n``.
        order: the order of the product formula.
        from_staggered: whether the transformation of the second digital branch,
            ``qubit_register_staggered -> trotter_staggered``, is built instead.

    Returns:
        The transformation.

    """
    source, target, node, suffix = (
        (QUBIT_REGISTER_STAGGERED, TROTTER_STAGGERED, ALTERNATIVE_DIGITAL_TARGET, ", staggered")
        if from_staggered
        else (QUBIT_REGISTER, TROTTER, DIGITAL_TARGET, "")
    )
    return suzuki_trotter(
        source,
        target,
        time=time,
        steps=steps,
        order=order,
        target_name=node,
        name=f"Suzuki-Trotter product formula, order {order}{suffix}",
    )


def validity(**thresholds: float) -> Conjunction:
    """The validity conditions of the perturbative step, over the namespaces of this use case."""
    return superlattice_validity(EFFECTIVE_BOSONIC, BOSE_HUBBARD, **thresholds)


def coupling_map() -> Scalar:
    """The coupling of the perturbative step in the forward direction of the derivation.

    The expression is stated over the hardware namespace of this use case.
    """
    return second_order_coupling(BOSE_HUBBARD)


def mass_map() -> Scalar:
    """The mass of the perturbative step in the forward direction of the derivation.

    The expression is stated over the hardware namespace of this use case.
    """
    return second_order_mass(BOSE_HUBBARD)


def device_limits(
    *,
    tunnelling_max: float = DEFAULT_TUNNELLING_MAX,
    interaction_range: tuple[float, float] = DEFAULT_INTERACTION_RANGE,
    superlattice_max: float = DEFAULT_SUPERLATTICE_MAX,
    tilt_max: float = DEFAULT_TILT_MAX,
    tilt_fraction: float = 0.5,
) -> AdmissibleSet:
    """The admissible knob set of the superlattice hardware, over the namespace of this use case.

    The keywords are those of
    [`bose_hubbard_admissible_set`][qsimod.models.hardware.bose_hubbard_admissible_set].
    """
    return bose_hubbard_admissible_set(
        BOSE_HUBBARD,
        tunnelling_max=tunnelling_max,
        interaction_range=interaction_range,
        superlattice_max=superlattice_max,
        tilt_max=tilt_max,
        tilt_fraction=tilt_fraction,
    )


def _placeholder_product_formula(
    name: str,
    structure_source: HamiltonianModel,
) -> ProductFormulaModel:
    """A placeholder product-formula artifact that lets the model graph be enumerated unbound.

    A product formula requires bound parameters for its angles; the placeholder carries the
    correct structural type and artifact kind, and applying the pipeline produces the actual
    artifact.
    """
    trivial = PauliSum.from_terms([(PauliString(), 0.0)], "placeholder")
    return ProductFormulaModel(
        name=name,
        structure=structure_source.structure,
        level=AbstractionLevel.HARDWARE,
        origin=(
            f"placeholder node for the product formula derived from {structure_source.name}; "
            "applying the pipeline produces the real artifact, whose angles need bound "
            "parameters"
        ),
        layers=declared_layers([("placeholder", trivial)]),
        formula=lie_trotter(1),
        time=1.0,
        steps=1,
    )


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchwingerGraph(UseCaseGraph):
    """The model graph of the use case, together with the pipelines through it.

    Attributes:
        matter_sites: the chain length ``N``.
        analogue: the pipeline to ``bose_hubbard`` through ``effective_bosonic``.
        digital: the pipeline to ``trotter`` through ``quantum_link_homogeneous`` and
            ``qubit_register``.
        alternative_digital: the pipeline to ``trotter_staggered`` that leaves
            ``quantum_link_staggered`` directly.

    """

    source = SOURCE

    matter_sites: int
    analogue: Pipeline
    digital: Pipeline
    alternative_digital: Pipeline


def build_graph(
    matter_sites: int,
    *,
    admissible_set: AdmissibleSet | None = None,
    time: float = 1.0,
    steps: int = 1,
    order: int = 2,
) -> SchwingerGraph:
    """Assemble the model graph of the use case at a given chain length.

    Args:
        matter_sites: the chain length ``N``.
        admissible_set: the admissible knob set of the superlattice hardware; defaults to
            [`bose_hubbard_admissible_set`][qsimod.models.hardware.bose_hubbard_admissible_set]
            on the namespace of this use case.
        time: the simulated time with which the Trotterisation transformations are configured.
        steps: the step count with which they are configured.
        order: the order of the product formula.

    Returns:
        The model graph and the three pipelines through it.

    """
    limits = admissible_set or bose_hubbard_admissible_set(BOSE_HUBBARD)
    graph = ModelGraph(name=f"Schwinger model graph at N = {matter_sites}")
    qubits = qubit_register(matter_sites)
    qubits_staggered = qubit_register_from_staggered(matter_sites)

    graph.add_node(theory(matter_sites))
    graph.add_node(staggered_quantum_link(matter_sites))
    graph.add_node(homogeneous_quantum_link(matter_sites))
    graph.add_node(effective_bosonic(matter_sites))
    graph.add_node(superlattice(matter_sites, limits))
    graph.add_node(qubits)
    graph.add_node(qubits_staggered)
    graph.add_node(_placeholder_product_formula(DIGITAL_TARGET, qubits))
    graph.add_node(_placeholder_product_formula(ALTERNATIVE_DIGITAL_TARGET, qubits_staggered))

    truncate = truncation()
    relabel = particle_hole()
    encode = encoding()
    perturb = perturbation(limits)
    jordan_wigner = to_qubits()
    jordan_wigner_staggered = to_qubits_from_staggered()
    trotterise = trotterisation(time=time, steps=steps, order=order)
    trotterise_staggered = trotterisation(time=time, steps=steps, order=order, from_staggered=True)

    graph.add_edge("lattice_qed", "quantum_link_staggered", truncate)
    graph.add_edge("quantum_link_staggered", "quantum_link_homogeneous", relabel)
    graph.add_edge("quantum_link_homogeneous", "effective_bosonic", encode)
    graph.add_edge("effective_bosonic", ANALOGUE_TARGET, perturb)
    graph.add_edge("quantum_link_homogeneous", "qubit_register", jordan_wigner)
    graph.add_edge("qubit_register", DIGITAL_TARGET, trotterise)
    graph.add_edge("quantum_link_staggered", "qubit_register_staggered", jordan_wigner_staggered)
    graph.add_edge("qubit_register_staggered", ALTERNATIVE_DIGITAL_TARGET, trotterise_staggered)

    return SchwingerGraph(
        matter_sites=matter_sites,
        graph=graph,
        analogue=Pipeline.of(
            [truncate, relabel, encode, perturb],
            name="analogue branch, to the Bose-Hubbard chain",
        ),
        digital=Pipeline.of(
            [truncate, relabel, jordan_wigner, trotterise],
            name="digital branch, to the Trotter product formula",
        ),
        alternative_digital=Pipeline.of(
            [truncate, jordan_wigner_staggered, trotterise_staggered],
            name="digital branch from the staggered quantum-link model",
        ),
    )


def analogue_pipeline(
    matter_sites: int = 3,
    admissible_set: AdmissibleSet | None = None,
) -> Pipeline:
    """The analogue pipeline alone."""
    return build_graph(matter_sites, admissible_set=admissible_set).analogue


def digital_pipeline(
    matter_sites: int = 4,
    *,
    time: float = 2.0,
    steps: int = 16,
    order: int = 2,
) -> Pipeline:
    """The digital pipeline alone, at the given resource settings."""
    return build_graph(matter_sites, time=time, steps=steps, order=order).digital
