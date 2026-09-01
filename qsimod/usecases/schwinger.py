"""The case study of the article: the lattice Schwinger model, assembled from the libraries.

```
L1   H_sys    Kogut-Susskind lattice QED             Theta_sys = {m, a, e}    (application)
 |    (a) quantum-link truncation                    APPROXIMATE (regime conditions)
L2a  H_IR1    quantum-link model, staggered mass     Theta_IR1 = {m, kappa}   (intermediate)
 |    (b) particle-hole transformation               EXACT
L2b  H_IR2    quantum-link model, pair coupling      Theta_IR2 = {m, kappa}   (intermediate)
 |            the branch point
 +---------------- ANALOGUE -----------------+-------------- DIGITAL ----------------
 |  (c) boson encoding                       |  (e) Jordan-Wigner transformation
 |      EXACT on the encoded subspace        |      EXACT
L2c  H_IR3    effective bosonic model        L2d  H_IR4    qubit Hamiltonian, 3-local
 |            Theta_IR3 = {m, kappa}         |             Theta_IR4 = {m, kappa}
 |  (d) second-order degenerate perturbation |  (f) Trotterisation
 |      theory, solved for the knob settings |      APPROXIMATE (resource-controlled)
 |      APPROXIMATE (regime conditions)      |
L3a  H_sim    tilted, staggered Bose-Hubbard L3b  U_approx  Trotter product formula
              chain (hardware, analogue)                   U_approx(t; n, order)
              Theta_sim = {J, U, delta, Delta}             with m, kappa (hardware, digital)
```

The names ``H_sys``, ``H_IR1`` to ``H_IR4``, ``H_sim`` and ``U_approx`` are those of the
article.  In the code, the Hamiltonians of the artifacts are named ``H_QED`` (``L1``),
``H_QLM^st`` (``L2a``), ``H_QLM`` (``L2b``), ``H_eff`` (``L2c``), ``H_qubit`` (``L2d``),
``H_BHM`` (``L3a``) and ``U_Trotter`` (``L3b``).  The transformations are (a) the quantum-link
truncation, (b) the particle-hole transformation, (c) the boson encoding, (d) second-order
degenerate perturbation theory solved for the knob settings, (e) the Jordan-Wigner
transformation and (f) the Trotterisation.

A second digital branch leaves ``L2a`` directly: ``(e')`` leads to ``L2d_st`` and ``(f')`` to
``L3b_st``.  Its qubit Hamiltonian is unitarily equivalent to that of ``L2d``.

Conventions: the coupling phase is ``(kappa/2)(... + h.c.)``; the link normalisation is
``U -> -i (2/sqrt(3)) S^+``, with ``kappa`` released as a free parameter at ``L2a``; the
electric-field substitution is uniform, with ``U ~ S^+`` raising the field; the particle-hole
transformation acts on the odd matter sites and the even links and passes ``+m`` to ``L2b``;
``L2b`` carries the pair coupling, a uniform mass and a uniform link operator; the boundary
Gauss background is ``+1/2``; the Bose-Hubbard tunnelling is ``-J(...)``.
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

L1 = Namespace("L1")
L2A = Namespace("L2a")
L2B = Namespace("L2b")
L2C = Namespace("L2c")
L2D = Namespace("L2d")
L2D_ST = Namespace("L2d_st")
L3A = Namespace("L3a")
L3B = Namespace("L3b")
L3B_ST = Namespace("L3b_st")

#: Every namespace this use case owns, in pipeline order.
NAMESPACES = (L1, L2A, L2B, L2C, L2D, L2D_ST, L3A, L3B, L3B_ST)

SOURCE = "L1"
ANALOGUE_TARGET = "L3a"
DIGITAL_TARGET = "L3b"
ALTERNATIVE_DIGITAL_TARGET = "L3b_st"


class ParameterNames:
    """The fully qualified parameter names of this use case."""

    MASS_L1 = L1(names.MASS)
    LATTICE_SPACING = L1(names.LATTICE_SPACING)
    GAUGE_COUPLING = L1(names.GAUGE_COUPLING)
    ELECTRIC_GAP = L1(names.ELECTRIC_GAP)

    MASS_L2A = L2A(names.MASS)
    COUPLING_L2A = L2A(names.COUPLING)
    MASS_L2B = L2B(names.MASS)
    COUPLING_L2B = L2B(names.COUPLING)
    MASS_L2C = L2C(names.MASS)
    COUPLING_L2C = L2C(names.COUPLING)
    MASS_L2D = L2D(names.MASS)
    COUPLING_L2D = L2D(names.COUPLING)
    MASS_L2D_FROM_L2A = L2D_ST(names.MASS)
    COUPLING_L2D_FROM_L2A = L2D_ST(names.COUPLING)

    TUNNELLING = L3A(names.TUNNELLING)
    INTERACTION = L3A(names.INTERACTION)
    SUPERLATTICE = L3A(names.SUPERLATTICE)
    TILT = L3A(names.TILT)

    MASS_L3B = L3B(names.MASS)
    COUPLING_L3B = L3B(names.COUPLING)


# ---------------------------------------------------------------------------
# The nodes
# ---------------------------------------------------------------------------


def theory(matter_sites: int) -> HamiltonianModel:
    """``L1``: the Kogut-Susskind lattice Schwinger model, the application model ``H_sys``."""
    return kogut_susskind_gauge_theory(matter_sites, L1, name="L1", hamiltonian_name="H_QED")


def staggered_quantum_link(matter_sites: int) -> HamiltonianModel:
    """``L2a``: the spin-1/2 quantum-link model with staggered fermions, ``H_IR1``."""
    return quantum_link_model(
        matter_sites, L2A, STAGGERED_CONVENTION, name="L2a", hamiltonian_name="H_QLM^st"
    )


def homogeneous_quantum_link(matter_sites: int) -> HamiltonianModel:
    """``L2b``: the quantum-link model after the particle-hole transformation, ``H_IR2``.

    The model carries the constant ``-m * floor(N/2)`` the mass staggering left behind and is
    the branch point of the case study.
    """
    return quantum_link_model(
        matter_sites,
        L2B,
        HOMOGENEOUS_CONVENTION,
        name="L2b",
        hamiltonian_name="H_QLM",
        additive_constant=staggering_constant(matter_sites, L2B.symbol(names.MASS)),
    )


def effective_bosonic(matter_sites: int) -> HamiltonianModel:
    """``L2c``: the effective bosonic pair-coupling model, ``H_IR3``.

    The model is the one realised by the superlattice.
    """
    return bosonic_pair_coupling_model(
        matter_sites,
        L2C,
        name="L2c",
        hamiltonian_name="H_eff",
        additive_constant=staggering_constant(matter_sites, L2C.symbol(names.MASS)),
    )


def superlattice(
    matter_sites: int,
    admissible_set: AdmissibleSet | None = None,
) -> HamiltonianModel:
    """``L3a``: the tilted, staggered Bose-Hubbard chain, the analogue simulator model ``H_sim``."""
    return tilted_bose_hubbard_chain(
        matter_sites,
        L3A,
        admissible_set=admissible_set,
        name="L3a",
        hamiltonian_name="H_BHM",
    )


def qubit_register(matter_sites: int) -> HamiltonianModel:
    """``L2d``: the 3-local Pauli Hamiltonian on the interleaved qubit register, ``H_IR4``."""
    return k_local_qubit_model(
        matter_sites,
        L2D,
        HOMOGENEOUS_CONVENTION,
        name="L2d",
        hamiltonian_name="H_qubit",
        additive_constant=staggering_constant(matter_sites, L2D.symbol(names.MASS)),
    )


def qubit_register_from_staggered(matter_sites: int) -> HamiltonianModel:
    """``L2d_st``: the qubit register reached by branching at ``L2a`` instead of ``L2b``."""
    return k_local_qubit_model(
        matter_sites,
        L2D_ST,
        STAGGERED_CONVENTION,
        name="L2d_st",
        hamiltonian_name="H_qubit^st",
    )


# ---------------------------------------------------------------------------
# The edges
# ---------------------------------------------------------------------------


def truncation() -> QuantumLinkTruncation:
    """Transformation (a): the quantum-link truncation, ``L1 -> L2a``."""
    return quantum_link_truncation(
        L1, L2A, target_name="L2a", name="(a) spin-1/2 quantum-link truncation"
    )


def particle_hole() -> ParticleHoleTransformation:
    """Transformation (b): the particle-hole transformation, ``L2a -> L2b``."""
    return particle_hole_transformation(
        L2A, L2B, target_name="L2b", name="(b) particle-hole transformation"
    )


def encoding() -> HardcoreBosonEncoding:
    """Transformation (c): the boson encoding, ``L2b -> L2c``."""
    return hardcore_boson_encoding(
        L2B, L2C, target_name="L2c", name="(c) Jordan-Wigner + hardcore-boson encoding"
    )


def perturbation(
    admissible_set: AdmissibleSet | None = None,
) -> SecondOrderPerturbationTheory:
    """Transformation (d): second-order degenerate perturbation theory, ``L2c -> L3a``.

    The relation is solved for the knob settings ``Theta_sim = {J, U, delta, Delta}``.
    """
    return second_order_perturbation_theory(
        L2C,
        L3A,
        admissible_set=admissible_set or bose_hubbard_admissible_set(L3A),
        target_name="L3a",
        name="(d) second-order degenerate perturbation theory, inverted",
    )


def to_qubits() -> JordanWignerToQubits:
    """Transformation (e): the Jordan-Wigner transformation to qubits, ``L2b -> L2d``."""
    return jordan_wigner_to_qubits(L2B, L2D, target_name="L2d", name="(e) Jordan-Wigner to qubits")


def to_qubits_from_staggered() -> JordanWignerToQubits:
    """Transformation (e'): the Jordan-Wigner transformation from ``L2a``, ``L2a -> L2d_st``."""
    return jordan_wigner_to_qubits(
        L2A,
        L2D_ST,
        convention=STAGGERED_CONVENTION,
        retain_staggering_constant=False,
        target_name="L2d_st",
        name="(e') Jordan-Wigner to qubits, branching at L2a",
    )


def trotterisation(
    *,
    time: float = 1.0,
    steps: int = 1,
    order: int = 2,
    from_staggered: bool = False,
) -> SuzukiTrotter:
    """Transformation (f): the Trotterisation, ``L2d -> L3b``, or (f'), ``L2d_st -> L3b_st``.

    Args:
        time: the simulated time ``t``.
        steps: the step count ``n``.
        order: the order of the product formula.
        from_staggered: whether the transformation ``(f')`` out of ``L2d_st`` is built instead.

    Returns:
        The transformation.

    """
    source, target, tag, node = (
        (L2D_ST, L3B_ST, "(f')", ALTERNATIVE_DIGITAL_TARGET)
        if from_staggered
        else (L2D, L3B, "(f)", DIGITAL_TARGET)
    )
    return suzuki_trotter(
        source,
        target,
        time=time,
        steps=steps,
        order=order,
        target_name=node,
        name=f"{tag} Suzuki-Trotter product formula, order {order}",
    )


def validity(**thresholds: float) -> Conjunction:
    """The validity conditions of transformation (d), over the namespaces of this use case."""
    return superlattice_validity(L2C, L3A, **thresholds)


def coupling_map() -> Scalar:
    """The coupling of transformation (d) in the forward direction of the derivation.

    The expression is stated over the hardware namespace of this use case.
    """
    return second_order_coupling(L3A)


def mass_map() -> Scalar:
    """The mass of transformation (d) in the forward direction of the derivation.

    The expression is stated over the hardware namespace of this use case.
    """
    return second_order_mass(L3A)


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
        L3A,
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
        analogue: ``L1 -> L2a -> L2b -> L2c -> L3a``.
        digital: ``L1 -> L2a -> L2b -> L2d -> L3b``.
        alternative_digital: ``L1 -> L2a -> L2d_st -> L3b_st``.

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
    limits = admissible_set or bose_hubbard_admissible_set(L3A)
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

    step_a = truncation()
    step_b = particle_hole()
    step_c = encoding()
    step_d = perturbation(limits)
    step_e = to_qubits()
    step_e_alt = to_qubits_from_staggered()
    step_f = trotterisation(time=time, steps=steps, order=order)
    step_f_alt = trotterisation(time=time, steps=steps, order=order, from_staggered=True)

    graph.add_edge("L1", "L2a", step_a)
    graph.add_edge("L2a", "L2b", step_b)
    graph.add_edge("L2b", "L2c", step_c)
    graph.add_edge("L2c", ANALOGUE_TARGET, step_d)
    graph.add_edge("L2b", "L2d", step_e)
    graph.add_edge("L2d", DIGITAL_TARGET, step_f)
    graph.add_edge("L2a", "L2d_st", step_e_alt)
    graph.add_edge("L2d_st", ALTERNATIVE_DIGITAL_TARGET, step_f_alt)

    return SchwingerGraph(
        matter_sites=matter_sites,
        graph=graph,
        analogue=Pipeline.of(
            [step_a, step_b, step_c, step_d],
            name="analogue: L1 -> L2a -> L2b -> L2c -> L3a",
        ),
        digital=Pipeline.of(
            [step_a, step_b, step_e, step_f],
            name="digital: L1 -> L2a -> L2b -> L2d -> L3b",
        ),
        alternative_digital=Pipeline.of(
            [step_a, step_e_alt, step_f_alt],
            name="digital (other branch point): L1 -> L2a -> L2d_st -> L3b_st",
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
