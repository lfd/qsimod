"""The anisotropic Heisenberg magnet, assembled from the model and transformation libraries.

The graph runs from ``xxz_magnet`` to the branch point ``xxz_chain``, whose two branches are
the exact Jordan-Wigner image ``fermion_chain`` and the second-order superexchange onto the
two-component Bose-Hubbard chain ``two_component_bose_hubbard``, after Jepsen et al. (2020).
The longitudinal field the superexchange also produces is returned by
[`field_map`][qsimod.usecases.heisenberg.field_map] and is not part of the magnet.  The guide
page of the documentation gives the Hamiltonians and the conventions.
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

XXZ_MAGNET = Namespace("xxz_magnet")
XXZ_CHAIN = Namespace("xxz_chain")
FERMION_CHAIN = Namespace("fermion_chain")
TWO_COMPONENT_BOSE_HUBBARD = Namespace("two_component_bose_hubbard")

#: Every namespace this use case owns, in pipeline order.
NAMESPACES = (XXZ_MAGNET, XXZ_CHAIN, FERMION_CHAIN, TWO_COMPONENT_BOSE_HUBBARD)

SOURCE = "xxz_magnet"
DEVICE_TARGET = "two_component_bose_hubbard"
FERMION_TARGET = "fermion_chain"


class ParameterNames:
    """The fully qualified parameter names of this use case."""

    TRANSVERSE_XXZ_MAGNET = XXZ_MAGNET(names.TRANSVERSE_COUPLING)
    ANISOTROPY = XXZ_MAGNET(names.ANISOTROPY)

    TRANSVERSE_XXZ_CHAIN = XXZ_CHAIN(names.TRANSVERSE_COUPLING)
    LONGITUDINAL_XXZ_CHAIN = XXZ_CHAIN(names.LONGITUDINAL_COUPLING)
    TRANSVERSE_FERMION_CHAIN = FERMION_CHAIN(names.TRANSVERSE_COUPLING)
    LONGITUDINAL_FERMION_CHAIN = FERMION_CHAIN(names.LONGITUDINAL_COUPLING)

    HOPPING = TWO_COMPONENT_BOSE_HUBBARD(names.HOPPING)
    INTERACTION_UP = TWO_COMPONENT_BOSE_HUBBARD(names.INTERACTION_UP)
    INTERACTION_MIXED = TWO_COMPONENT_BOSE_HUBBARD(names.INTERACTION_MIXED)
    INTERACTION_DOWN = TWO_COMPONENT_BOSE_HUBBARD(names.INTERACTION_DOWN)


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
    """``xxz_magnet``: the XXZ magnet stated by its anisotropy."""
    return heisenberg_magnet(sites, XXZ_MAGNET, name="xxz_magnet", hamiltonian_name="H_XXZ")


def spin_chain(sites: int) -> HamiltonianModel:
    """``xxz_chain``: the same chain, stated by two coupling energies."""
    return xxz_spin_chain(sites, XXZ_CHAIN, name="xxz_chain", hamiltonian_name="H_XXZ")


def fermion_chain(sites: int) -> HamiltonianModel:
    """``fermion_chain``: the Jordan-Wigner image, spinless fermions with a neighbour ``V``."""
    return interacting_fermion_chain(
        sites, FERMION_CHAIN, name="fermion_chain", hamiltonian_name="H_tV"
    )


def lattice(
    sites: int,
    admissible_set: AdmissibleSet | None = None,
) -> HamiltonianModel:
    """``two_component_bose_hubbard``: the two-component Bose-Hubbard chain, the hardware model."""
    return two_component_bose_hubbard_chain(
        sites,
        TWO_COMPONENT_BOSE_HUBBARD,
        admissible_set=admissible_set,
        name="two_component_bose_hubbard",
        hamiltonian_name="H_2BHM",
    )


# ---------------------------------------------------------------------------
# The edges
# ---------------------------------------------------------------------------


def anisotropy() -> AnisotropyResolution:
    """The anisotropy resolution, ``xxz_magnet -> xxz_chain``."""
    return anisotropy_resolution(
        XXZ_MAGNET, XXZ_CHAIN, target_name="xxz_chain", name="anisotropy resolution"
    )


def jordan_wigner() -> JordanWignerToFermions:
    """The Jordan-Wigner transformation to fermions, ``xxz_chain -> fermion_chain``."""
    return jordan_wigner_to_fermions(
        XXZ_CHAIN,
        FERMION_CHAIN,
        target_name="fermion_chain",
        name="Jordan-Wigner to spinless fermions",
    )


def superexchange(
    admissible_set: AdmissibleSet | None = None,
) -> SuperexchangeReduction:
    """Second-order superexchange solved for the knobs, to ``two_component_bose_hubbard``."""
    return superexchange_reduction(
        XXZ_CHAIN,
        TWO_COMPONENT_BOSE_HUBBARD,
        admissible_set=admissible_set or two_component_admissible_set(TWO_COMPONENT_BOSE_HUBBARD),
        target_name="two_component_bose_hubbard",
        name="second-order superexchange, inverted",
    )


def validity(**thresholds: float) -> Conjunction:
    """The validity conditions of the superexchange step, over the namespaces of this use case."""
    return superexchange_validity(XXZ_CHAIN, TWO_COMPONENT_BOSE_HUBBARD, **thresholds)


def transverse_map() -> Scalar:
    """The coupling ``Jxy`` in the forward direction of the derivation.

    The expression is stated over the hardware namespace of this use case.
    """
    return superexchange_transverse(TWO_COMPONENT_BOSE_HUBBARD)


def longitudinal_map() -> Scalar:
    """The coupling ``Jz`` in the forward direction of the derivation.

    The expression is stated over the hardware namespace of this use case.
    """
    return superexchange_longitudinal(TWO_COMPONENT_BOSE_HUBBARD)


def field_map() -> Scalar:
    """The strength of the longitudinal field that the superexchange step also produces.

    The field is not part of the target model; see
    [`superexchange_field`][qsimod.transformations.perturbative.superexchange_field].
    """
    return superexchange_field(TWO_COMPONENT_BOSE_HUBBARD)


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
        TWO_COMPONENT_BOSE_HUBBARD,
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

    The admissible interval of ``U_dd`` straddles the pole ``U_dd = 0``, and an anisotropy
    below ``-1`` is reachable only with a repulsive ``U_dd``; the starting point selects the
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
        device: the pipeline to ``two_component_bose_hubbard``.
        fermions: the pipeline to ``fermion_chain``.

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
    limits = admissible_set or two_component_admissible_set(TWO_COMPONENT_BOSE_HUBBARD)
    graph = ModelGraph(name=f"Heisenberg XXZ magnet graph at N = {sites}")

    graph.add_node(magnet(sites))
    graph.add_node(spin_chain(sites))
    graph.add_node(fermion_chain(sites))
    graph.add_node(lattice(sites, limits))

    resolve = anisotropy()
    to_fermions = jordan_wigner()
    reduce = superexchange(limits)

    graph.add_edge(SOURCE, "xxz_chain", resolve)
    graph.add_edge("xxz_chain", FERMION_TARGET, to_fermions)
    graph.add_edge("xxz_chain", DEVICE_TARGET, reduce)

    return HeisenbergGraph(
        sites=sites,
        graph=graph,
        device=Pipeline.of([resolve, reduce], name="device branch, to the two-component lattice"),
        fermions=Pipeline.of([resolve, to_fermions], name="analytic branch, to spinless fermions"),
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
