"""The antiferromagnetic Ising chain, on the hardware model of the case study.

```
ising_magnet   H_Ising   antiferromagnetic Ising chain, by its dimensionless fields  (application)
 |    field resolution                                 EXACT
ising_chain    H_Ising   Ising chain, by three energies                             (intermediate)
 |    resonant dipole reduction, solved for the knobs  APPROXIMATE (regime conditions)
bose_hubbard   H_sim     tilted Bose-Hubbard chain     <-- the artifact the gauge theory reaches
```

The hardware model is the artifact ``bose_hubbard`` of [`schwinger`][qsimod.usecases.schwinger],
the analogue simulator model ``H_sim`` of the case study, with its namespace and parameters;
[`shared_graph`][qsimod.usecases.ising.shared_graph] holds both theories in one model graph.
A dipole resides on a bond, so a register of ``2N-1`` sites carries ``2N-2`` spins.  The
physics is that of Simon et al., *Quantum simulation of antiferromagnetic spin chains in an
optical lattice*, Nature **472**, 307-312 (2011).

Conventions: ``Jz > 0``; the fields are written ``-hz S^z`` and ``-hx S^x``, so that the
multicritical point is ``(hz, hx) = (1, 0)``; the tilt is positive and the dipole is formed
down the gradient, so that the resonance is ``Delta = U``; the constraint penalty equals
``U``; the end-spin field omitted by the Hamiltonian of the target is returned by
[`boundary_field`][qsimod.usecases.ising.boundary_field].
"""

from __future__ import annotations

from dataclasses import dataclass

from qsimod.artifact import HamiltonianModel
from qsimod.models import names
from qsimod.models.application import ising_magnet
from qsimod.models.hardware import (
    DEFAULT_INTERACTION_RANGE,
    DEFAULT_SUPERLATTICE_MAX,
    DEFAULT_TUNNELLING_MAX,
)
from qsimod.models.intermediate import ising_spin_chain
from qsimod.models.magnetism import end_magnetisation_term
from qsimod.parameters import AdmissibleSet, Namespace
from qsimod.pipeline import ModelGraph, Pipeline
from qsimod.scalar import Scalar
from qsimod.transformations import (
    DipoleReduction,
    FieldResolution,
    dipole_boundary_field,
    dipole_coupling,
    dipole_detuning,
    dipole_longitudinal,
    dipole_reduction,
    dipole_transverse,
    dipole_validity,
    field_resolution,
)
from qsimod.usecases import schwinger
from qsimod.usecases.base import UseCaseGraph
from qsimod.validity import Conjunction

__all__ = [
    "DEFAULT_RESONANT_TILT_MAX",
    "DEVICE",
    "DEVICE_TARGET",
    "KNOBS",
    "SOURCE",
    "IsingGraph",
    "ParameterNames",
    "boundary_field",
    "build_graph",
    "coupling_map",
    "detuning_map",
    "device_limits",
    "device_pipeline",
    "device_start",
    "dipoles",
    "end_magnetisation_term",
    "fields",
    "ising_chain",
    "lattice",
    "longitudinal_map",
    "magnet",
    "shared_graph",
    "spins_for",
    "transverse_map",
    "validity",
]

# ---------------------------------------------------------------------------
# The namespaces this use case gives its models
# ---------------------------------------------------------------------------

ISING_MAGNET = Namespace("ising_magnet")
ISING_CHAIN = Namespace("ising_chain")

#: The namespace of the hardware model, which [`schwinger`][qsimod.usecases.schwinger] assigned
#: to ``bose_hubbard``.
DEVICE = schwinger.BOSE_HUBBARD

#: Every namespace this use case reads, in pipeline order.
NAMESPACES = (ISING_MAGNET, ISING_CHAIN, DEVICE)

SOURCE = "ising_magnet"
DEVICE_TARGET = schwinger.ANALOGUE_TARGET


class ParameterNames:
    """The fully qualified parameter names; the hardware half is that of ``bose_hubbard``."""

    COUPLING_ISING_MAGNET = ISING_MAGNET(names.LONGITUDINAL_COUPLING)
    LONGITUDINAL_FIELD = ISING_MAGNET(names.LONGITUDINAL_FIELD)
    TRANSVERSE_FIELD = ISING_MAGNET(names.TRANSVERSE_FIELD)

    COUPLING_ISING_CHAIN = ISING_CHAIN(names.LONGITUDINAL_COUPLING)
    TRANSVERSE = ISING_CHAIN(names.TRANSVERSE_AMPLITUDE)
    LONGITUDINAL = ISING_CHAIN(names.LONGITUDINAL_BIAS)

    TUNNELLING = schwinger.ParameterNames.TUNNELLING
    INTERACTION = schwinger.ParameterNames.INTERACTION
    SUPERLATTICE = schwinger.ParameterNames.SUPERLATTICE
    TILT = schwinger.ParameterNames.TILT


#: The hardware knobs, in the order in which the tables of this use case print them.
KNOBS = (
    ParameterNames.TUNNELLING,
    ParameterNames.INTERACTION,
    ParameterNames.SUPERLATTICE,
    ParameterNames.TILT,
)


def spins_for(matter_sites: int) -> int:
    """The number of Ising spins on a register of ``2N-1`` sites: ``2N-2``, one per bond."""
    return 2 * matter_sites - 2


# ---------------------------------------------------------------------------
# The nodes
# ---------------------------------------------------------------------------


def magnet(matter_sites: int) -> HamiltonianModel:
    """``ising_magnet``: the Ising chain stated by its two dimensionless fields."""
    return ising_magnet(
        spins_for(matter_sites), ISING_MAGNET, name="ising_magnet", hamiltonian_name="H_Ising"
    )


def ising_chain(matter_sites: int) -> HamiltonianModel:
    """``ising_chain``: the same chain, stated by three energies."""
    return ising_spin_chain(
        spins_for(matter_sites), ISING_CHAIN, name="ising_chain", hamiltonian_name="H_Ising"
    )


def lattice(
    matter_sites: int,
    admissible_set: AdmissibleSet | None = None,
) -> HamiltonianModel:
    """``bose_hubbard``: the tilted Bose-Hubbard chain, the analogue simulator of the case study."""
    return schwinger.superlattice(matter_sites, admissible_set)


# ---------------------------------------------------------------------------
# The edges
# ---------------------------------------------------------------------------


def fields() -> FieldResolution:
    """The field resolution, ``ising_magnet -> ising_chain``."""
    return field_resolution(
        ISING_MAGNET, ISING_CHAIN, target_name="ising_chain", name="field resolution"
    )


def dipoles(admissible_set: AdmissibleSet | None = None) -> DipoleReduction:
    """The resonant dipole reduction solved for the knobs, ``ising_chain -> bose_hubbard``."""
    return dipole_reduction(
        ISING_CHAIN,
        DEVICE,
        admissible_set=admissible_set or device_limits(),
        target_name=DEVICE_TARGET,
        name="resonant dipole reduction, inverted",
    )


def validity(**thresholds: float) -> Conjunction:
    """The validity conditions of the dipole reduction, over the namespaces of this use case."""
    return dipole_validity(ISING_CHAIN, DEVICE, **thresholds)


def coupling_map() -> Scalar:
    """The coupling ``Jz`` in the forward direction of the derivation.

    The expression is stated over the namespace of the shared hardware model.
    """
    return dipole_coupling(DEVICE)


def transverse_map() -> Scalar:
    """The transverse field ``Gamma`` in the forward direction of the derivation.

    The expression is stated over the namespace of the shared hardware model.
    """
    return dipole_transverse(DEVICE)


def longitudinal_map() -> Scalar:
    """The longitudinal field ``B`` in the forward direction of the derivation.

    The expression is stated over the namespaces of this use case.
    """
    return dipole_longitudinal(ISING_CHAIN, DEVICE)


def detuning_map() -> Scalar:
    """The detuning ``Delta - U`` from the dipole resonance."""
    return dipole_detuning(DEVICE)


def boundary_field() -> Scalar:
    """The end-spin field ``Jz / 2`` that the dipole reduction also produces.

    The field is not part of the target model; see
    [`dipole_boundary_field`][qsimod.transformations.perturbative.dipole_boundary_field].
    """
    return dipole_boundary_field(DEVICE)


#: The largest tilt the lattice reaches.  A dipole resonance requires ``Delta ~= U``, and the
#: reference experiment ramps the tilt from ``0.7 U`` to ``1.2 U``.
DEFAULT_RESONANT_TILT_MAX = 1.5 * DEFAULT_INTERACTION_RANGE[1]


def device_limits(
    *,
    tunnelling_max: float = DEFAULT_TUNNELLING_MAX,
    interaction_range: tuple[float, float] = DEFAULT_INTERACTION_RANGE,
    superlattice_max: float = DEFAULT_SUPERLATTICE_MAX,
    tilt_max: float = DEFAULT_RESONANT_TILT_MAX,
) -> AdmissibleSet:
    """The admissible knob set of the lattice, of which that of the case study is a sub-region.

    The set is built from [`schwinger.device_limits`][qsimod.usecases.schwinger.device_limits]
    with the constraints at the level of the derivation removed by
    [`apparatus_only`][qsimod.parameters.AdmissibleSet.apparatus_only], and with the upper
    limit of the tilt raised above that of the interaction.  The validity conditions of the
    superlattice transformation of the case study still declare ``Delta << delta`` and
    ``Delta << U``.

    Returns:
        The admissible set.

    """
    return schwinger.device_limits(
        tunnelling_max=tunnelling_max,
        interaction_range=interaction_range,
        superlattice_max=superlattice_max,
        tilt_max=tilt_max,
    ).apparatus_only()


def device_start(coupling: float) -> dict[str, float]:
    """A deterministic starting point for the dipole solve, in the resonant corner.

    Args:
        coupling: the requested Ising coupling ``Jz``, which is the interaction and, near the
            multicritical point, the tilt as well.

    Returns:
        A starting value per hardware knob.

    """
    return {
        ParameterNames.TUNNELLING: 0.02 * coupling,
        ParameterNames.INTERACTION: coupling,
        ParameterNames.SUPERLATTICE: 0.05 * coupling,
        ParameterNames.TILT: coupling,
    }


# ---------------------------------------------------------------------------
# The graphs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IsingGraph(UseCaseGraph):
    """The model graph of the use case, together with the pipeline through it.

    Attributes:
        matter_sites: the chain length ``N`` of the hardware model; the magnet has ``2N-2``
            spins.
        device: the pipeline to the shared ``bose_hubbard``.

    """

    source = SOURCE

    matter_sites: int
    device: Pipeline

    @property
    def spins(self) -> int:
        """The chain length of the magnet."""
        return spins_for(self.matter_sites)


def build_graph(
    matter_sites: int,
    *,
    admissible_set: AdmissibleSet | None = None,
) -> IsingGraph:
    """Assemble the model graph of the use case at a given chain length of the hardware model.

    Args:
        matter_sites: the chain length ``N`` of the hardware model; the magnet is built at
            ``2N-2`` spins.
        admissible_set: the admissible knob set of the hardware; defaults to
            [`device_limits`][qsimod.usecases.ising.device_limits].

    Returns:
        The model graph and the pipeline through it.

    """
    limits = admissible_set or device_limits()
    graph = ModelGraph(
        name=f"Ising chain graph at N = {matter_sites} ({spins_for(matter_sites)} spins)"
    )

    graph.add_node(magnet(matter_sites))
    graph.add_node(ising_chain(matter_sites))
    graph.add_node(lattice(matter_sites, limits))

    resolve = fields()
    reduce = dipoles(limits)
    graph.add_edge(SOURCE, "ising_chain", resolve)
    graph.add_edge("ising_chain", DEVICE_TARGET, reduce)

    return IsingGraph(
        matter_sites=matter_sites,
        graph=graph,
        device=Pipeline.of([resolve, reduce], name="device branch, to the shared lattice"),
    )


def shared_graph(
    matter_sites: int,
    *,
    admissible_set: AdmissibleSet | None = None,
) -> ModelGraph:
    """One model graph carrying both theories, which meet at the artifact ``bose_hubbard``.

    Both requests are posed against the admissible set of the lattice from
    [`device_limits`][qsimod.usecases.ising.device_limits].

    Args:
        matter_sites: the chain length ``N`` of the hardware model.
        admissible_set: the admissible knob set of the hardware, shared by both theories.

    Returns:
        The combined model graph.

    """
    limits = admissible_set or device_limits()
    gauge = schwinger.build_graph(matter_sites, admissible_set=limits)
    graph = ModelGraph(name=f"two theories, one device, N = {matter_sites}")

    for node in gauge.graph.nodes:
        graph.add_node(gauge.graph.node(node))
    for edge in gauge.graph.edges:
        graph.add_edge(edge.source, edge.target, edge.transformation)

    # The hardware model is already registered; only the magnet's two nodes and the
    # arriving edge are new.
    graph.add_node(magnet(matter_sites))
    graph.add_node(ising_chain(matter_sites))
    graph.add_edge(SOURCE, "ising_chain", fields())
    graph.add_edge("ising_chain", DEVICE_TARGET, dipoles(limits))
    return graph


def device_pipeline(
    matter_sites: int = 3,
    admissible_set: AdmissibleSet | None = None,
) -> Pipeline:
    """The Ising pipeline alone."""
    return build_graph(matter_sites, admissible_set=admissible_set).device
