"""The anisotropic Heisenberg magnet, assembled from the model and transformation libraries.

```
M1   H_XXZ     Heisenberg XXZ magnet, by its anisotropy            (application)
 |    (a) anisotropy resolution                        EXACT
M2a  H_XXZ     XXZ chain, by its two couplings                     (intermediate)  branch point
 +------------- ANALYTIC -------------+-------------- HARDWARE --------------
 |  (b) Jordan-Wigner transformation  |  (c) second-order superexchange,
 |               EXACT                |      solved for the knob settings
 |                                    |      APPROXIMATE (regime conditions)
M2b  H_tV   spinless fermions with V   M3   H_2BHM  two-component Bose-Hubbard chain
            (intermediate)                         (hardware, analogue)
```

``M2b`` is a terminal artifact among the intermediate representations: at zero anisotropy it
is a free-fermion chain.  The physics is that of Jepsen et al., *Spin transport in a tunable
Heisenberg model realized with ultracold atoms*, Nature **588**, 403-407 (2020).

Conventions: the transverse term is ``(Jxy/2)(S^+S^- + h.c.)``; the anisotropy is
``Delta = Jz / Jxy`` with ``Jxy > 0`` (antiferromagnetic), which confines ``U_ud`` to the
attractive branch of the Feshbach resonance; the two-component register is interleaved, with
the first component at ``2j`` and the second at ``2j+1``; the Jordan-Wigner transformation
uses alternating signs, as in [`qsimod.realise.build`][qsimod.realise.build]; the longitudinal
field the superexchange derivation also produces is returned by
[`field_map`][qsimod.usecases.heisenberg.field_map] and is not part of ``M2a``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qsimod.artifact import HamiltonianModel
from qsimod.models import names
from qsimod.models.application import heisenberg_magnet
from qsimod.models.hardware import (
    DEFAULT_HOPPING_MAX,
    DEFAULT_INTERACTION_MAGNITUDE_RANGE,
    DEFAULT_MOTT_RATIO,
    two_component_admissible_set,
    two_component_bose_hubbard_chain,
)
from qsimod.models.intermediate import interacting_fermion_chain, xxz_spin_chain
from qsimod.models.magnetism import (
    coordination_numbers,
    magnetisation_observable,
    mott_manifold_operators,
    weighted_magnetisation_term,
)
from qsimod.parameters import AdmissibleSet, Namespace
from qsimod.pipeline import ModelGraph, Pipeline
from qsimod.scalar import Scalar
from qsimod.transformations import (
    AnisotropyResolution,
    JordanWignerToFermions,
    SuperexchangeReduction,
    anisotropy_resolution,
    jordan_wigner_to_fermions,
    superexchange_field,
    superexchange_longitudinal,
    superexchange_reduction,
    superexchange_transverse,
    superexchange_validity,
)
from qsimod.usecases.base import UseCaseGraph
from qsimod.validity import Conjunction

__all__ = [
    "DEVICE_TARGET",
    "FERMION_TARGET",
    "NAMESPACES",
    "SOURCE",
    "START_HOPPING",
    "START_INTERACTION",
    "HeisenbergGraph",
    "ParameterNames",
    "anisotropy",
    "build_graph",
    "coordination_numbers",
    "device_limits",
    "device_pipeline",
    "device_start",
    "fermion_chain",
    "fermion_pipeline",
    "field_map",
    "jordan_wigner",
    "lattice",
    "longitudinal_map",
    "magnet",
    "magnetisation_observable",
    "mott_manifold_operators",
    "spin_chain",
    "superexchange",
    "transverse_map",
    "validity",
    "weighted_magnetisation_term",
]

# ---------------------------------------------------------------------------
# The namespaces this use case gives its models
# ---------------------------------------------------------------------------

M1 = Namespace("M1")
M2A = Namespace("M2a")
M2B = Namespace("M2b")
M3 = Namespace("M3")

#: Every namespace this use case owns, in pipeline order.
NAMESPACES = (M1, M2A, M2B, M3)

SOURCE = "M1"
DEVICE_TARGET = "M3"
FERMION_TARGET = "M2b"


class ParameterNames:
    """The fully qualified parameter names of this use case."""

    TRANSVERSE_M1 = M1(names.TRANSVERSE_COUPLING)
    ANISOTROPY = M1(names.ANISOTROPY)

    TRANSVERSE_M2A = M2A(names.TRANSVERSE_COUPLING)
    LONGITUDINAL_M2A = M2A(names.LONGITUDINAL_COUPLING)
    TRANSVERSE_M2B = M2B(names.TRANSVERSE_COUPLING)
    LONGITUDINAL_M2B = M2B(names.LONGITUDINAL_COUPLING)

    HOPPING = M3(names.HOPPING)
    INTERACTION_UP = M3(names.INTERACTION_UP)
    INTERACTION_MIXED = M3(names.INTERACTION_MIXED)
    INTERACTION_DOWN = M3(names.INTERACTION_DOWN)


#: The hardware knobs, in the order in which the tables of this use case print them.
KNOBS = (
    ParameterNames.HOPPING,
    ParameterNames.INTERACTION_UP,
    ParameterNames.INTERACTION_MIXED,
    ParameterNames.INTERACTION_DOWN,
)


# ---------------------------------------------------------------------------
# The nodes
# ---------------------------------------------------------------------------


def magnet(sites: int) -> HamiltonianModel:
    """``M1``: the XXZ magnet stated by its anisotropy."""
    return heisenberg_magnet(sites, M1, name="M1", hamiltonian_name="H_XXZ")


def spin_chain(sites: int) -> HamiltonianModel:
    """``M2a``: the same chain, stated by two coupling energies."""
    return xxz_spin_chain(sites, M2A, name="M2a", hamiltonian_name="H_XXZ")


def fermion_chain(sites: int) -> HamiltonianModel:
    """``M2b``: the Jordan-Wigner image, spinless fermions with a nearest-neighbour ``V``."""
    return interacting_fermion_chain(sites, M2B, name="M2b", hamiltonian_name="H_tV")


def lattice(
    sites: int,
    admissible_set: AdmissibleSet | None = None,
) -> HamiltonianModel:
    """``M3``: the two-component Bose-Hubbard chain, the hardware model."""
    return two_component_bose_hubbard_chain(
        sites, M3, admissible_set=admissible_set, name="M3", hamiltonian_name="H_2BHM"
    )


# ---------------------------------------------------------------------------
# The edges
# ---------------------------------------------------------------------------


def anisotropy() -> AnisotropyResolution:
    """Transformation (a): the anisotropy resolution, ``M1 -> M2a``."""
    return anisotropy_resolution(M1, M2A, target_name="M2a", name="(a) anisotropy resolution")


def jordan_wigner() -> JordanWignerToFermions:
    """Transformation (b): the Jordan-Wigner transformation to fermions, ``M2a -> M2b``."""
    return jordan_wigner_to_fermions(
        M2A, M2B, target_name="M2b", name="(b) Jordan-Wigner to spinless fermions"
    )


def superexchange(
    admissible_set: AdmissibleSet | None = None,
) -> SuperexchangeReduction:
    """Transformation (c): second-order superexchange solved for the knobs, ``M2a -> M3``."""
    return superexchange_reduction(
        M2A,
        M3,
        admissible_set=admissible_set or two_component_admissible_set(M3),
        target_name="M3",
        name="(c) second-order superexchange, inverted",
    )


def validity(**thresholds: float) -> Conjunction:
    """The validity conditions of transformation (c), over the namespaces of this use case."""
    return superexchange_validity(M2A, M3, **thresholds)


def transverse_map() -> Scalar:
    """The coupling ``Jxy`` in the forward direction of the derivation.

    The expression is stated over the hardware namespace of this use case.
    """
    return superexchange_transverse(M3)


def longitudinal_map() -> Scalar:
    """The coupling ``Jz`` in the forward direction of the derivation.

    The expression is stated over the hardware namespace of this use case.
    """
    return superexchange_longitudinal(M3)


def field_map() -> Scalar:
    """The strength of the longitudinal field that transformation (c) also produces.

    The field is not part of the target model; see
    [`superexchange_field`][qsimod.transformations.perturbative.superexchange_field].
    """
    return superexchange_field(M3)


def device_limits(
    *,
    hopping_max: float = DEFAULT_HOPPING_MAX,
    interaction_magnitude_range: tuple[float, float] = DEFAULT_INTERACTION_MAGNITUDE_RANGE,
    mott_ratio: float = DEFAULT_MOTT_RATIO,
) -> AdmissibleSet:
    """The admissible knob set of the lattice, over the namespace of this use case.

    The keywords are those of
    [`two_component_admissible_set`][qsimod.models.hardware.two_component_admissible_set].
    """
    return two_component_admissible_set(
        M3,
        hopping_max=hopping_max,
        interaction_magnitude_range=interaction_magnitude_range,
        mott_ratio=mott_ratio,
    )


#: The starting magnitude of every interaction channel: the geometric mean of the reachable
#: range of the hardware.
START_INTERACTION = math.sqrt(
    DEFAULT_INTERACTION_MAGNITUDE_RANGE[0] * DEFAULT_INTERACTION_MAGNITUDE_RANGE[1]
)

#: The starting tunnelling, a quarter of the upper limit.
START_HOPPING = DEFAULT_HOPPING_MAX / 4


def device_start(
    anisotropy: float = 0.0,
    *,
    hopping: float = START_HOPPING,
    interaction: float = START_INTERACTION,
) -> dict[str, float]:
    """A deterministic starting point for the knob solve, on the intended side of the pole.

    The admissible interval of ``U_dd`` straddles ``U_dd = 0``.  With
    ``Delta + 1 = U_ud/U_uu + U_ud/U_dd`` and the other two channels attractive, an anisotropy
    below ``-1`` is reachable only with a repulsive ``U_dd``; the starting point selects that
    branch.

    Args:
        anisotropy: the requested anisotropy.
        hopping: the starting tunnelling.
        interaction: the starting magnitude of all three interaction channels.

    Returns:
        A starting value per hardware knob.

    """
    repulsive_down = anisotropy < -1.0
    return {
        ParameterNames.HOPPING: hopping,
        ParameterNames.INTERACTION_UP: -interaction,
        ParameterNames.INTERACTION_MIXED: -interaction,
        ParameterNames.INTERACTION_DOWN: interaction if repulsive_down else -interaction,
    }


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeisenbergGraph(UseCaseGraph):
    """The model graph of the use case, together with the pipelines through it.

    Attributes:
        sites: the chain length ``N``.
        device: ``M1 -> M2a -> M3``.
        fermions: ``M1 -> M2a -> M2b``.

    """

    source = SOURCE

    sites: int
    device: Pipeline
    fermions: Pipeline


def build_graph(
    sites: int,
    *,
    admissible_set: AdmissibleSet | None = None,
) -> HeisenbergGraph:
    """Assemble the model graph of the use case at a given chain length.

    Args:
        sites: the chain length ``N``.
        admissible_set: the admissible knob set of the lattice; defaults to
            [`two_component_admissible_set`][qsimod.models.hardware.two_component_admissible_set]
            on the namespace of this use case.

    Returns:
        The model graph and the two pipelines through it.

    """
    limits = admissible_set or two_component_admissible_set(M3)
    graph = ModelGraph(name=f"Heisenberg XXZ magnet graph at N = {sites}")

    graph.add_node(magnet(sites))
    graph.add_node(spin_chain(sites))
    graph.add_node(fermion_chain(sites))
    graph.add_node(lattice(sites, limits))

    step_a = anisotropy()
    step_b = jordan_wigner()
    step_c = superexchange(limits)

    graph.add_edge(SOURCE, "M2a", step_a)
    graph.add_edge("M2a", FERMION_TARGET, step_b)
    graph.add_edge("M2a", DEVICE_TARGET, step_c)

    return HeisenbergGraph(
        sites=sites,
        graph=graph,
        device=Pipeline.of([step_a, step_c], name="device: M1 -> M2a -> M3"),
        fermions=Pipeline.of([step_a, step_b], name="analytic: M1 -> M2a -> M2b"),
    )


def device_pipeline(
    sites: int = 3,
    admissible_set: AdmissibleSet | None = None,
) -> Pipeline:
    """The pipeline to the hardware model alone."""
    return build_graph(sites, admissible_set=admissible_set).device


def fermion_pipeline(sites: int = 6) -> Pipeline:
    """The analytic pipeline alone."""
    return build_graph(sites).fermions
