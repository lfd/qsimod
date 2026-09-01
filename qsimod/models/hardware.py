"""Models of the hardware layer: what a simulator natively realises, with its admissible knob set.

Level 3 of [`AbstractionLevel`][qsimod.levels.AbstractionLevel].  The parameters are the
hardware knobs set in the experiment, and each model declares an
[`AdmissibleSet`][qsimod.parameters.AdmissibleSet] of box bounds and coupled constraints.  The
models of this module are analogue simulator models; the digital simulator model is
[`ProductFormulaModel`][qsimod.trotter.schedule.ProductFormulaModel].
"""

from __future__ import annotations

from qsimod.artifact import HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models import names
from qsimod.models.gauge import (
    ElectricField,
    gauss_operators,
    gauss_sector,
    local_occupation_subspace,
    matter_gauge_structure,
)
from qsimod.models.magnetism import (
    down_index,
    two_component_pattern,
    two_component_structure,
    unit_filling_operators,
    unit_filling_sector,
    up_index,
)
from qsimod.parameters import (
    AdmissibleSet,
    Bound,
    ConstraintOrigin,
    InequalityConstraint,
    Namespace,
)
from qsimod.scalar import Scalar
from qsimod.structure import (
    Algebra,
    DegreeOfFreedom,
    DofRequirement,
    Lattice,
    LatticeGeometry,
    StructurePattern,
    StructureType,
    interleaved_site_count,
    open_chain_pattern,
)
from qsimod.symbolic import OperatorSum, annihilate, create, number, pauli_x, pauli_z, word
from qsimod.units import Dimension

__all__ = [
    "DEFAULT_HOPPING_MAX",
    "DEFAULT_INTERACTION_MAGNITUDE_RANGE",
    "DEFAULT_INTERACTION_RANGE",
    "DEFAULT_MOTT_RATIO",
    "DEFAULT_SUPERLATTICE_MAX",
    "DEFAULT_TILT_MAX",
    "DEFAULT_TUNNELLING_MAX",
    "SPIN_CHAIN_PATTERN",
    "TWO_COMPONENT_CHAIN_PATTERN",
    "bose_hubbard_admissible_set",
    "ising_admissible_set",
    "on_site_interaction",
    "tilted_bose_hubbard_chain",
    "transverse_field_ising_chain",
    "two_component_admissible_set",
    "two_component_bose_hubbard_chain",
]


# ---------------------------------------------------------------------------
# An optical superlattice: the tilted, staggered Bose-Hubbard chain
# ---------------------------------------------------------------------------

#: The default knob limits of the optical-superlattice simulator, in rad/ms.
DEFAULT_TUNNELLING_MAX = 0.25
DEFAULT_INTERACTION_RANGE = (0.05, 4.0)
DEFAULT_SUPERLATTICE_MAX = 2.5
DEFAULT_TILT_MAX = 0.6


def on_site_interaction(position: int, strength: Scalar) -> OperatorSum:
    """The on-site interaction ``(U/2) n (n - 1)`` at one register position, as two terms."""
    return word(strength / 2, number(position), number(position)) + word(
        -strength / 2, number(position)
    )


def bose_hubbard_admissible_set(
    namespace: Namespace,
    *,
    tunnelling_max: float = DEFAULT_TUNNELLING_MAX,
    interaction_range: tuple[float, float] = DEFAULT_INTERACTION_RANGE,
    superlattice_max: float = DEFAULT_SUPERLATTICE_MAX,
    tilt_max: float = DEFAULT_TILT_MAX,
    tilt_fraction: float = 0.5,
) -> AdmissibleSet:
    """The admissible knob set of an optical-superlattice simulator.

    The values ``J = 0`` and ``Delta = 0`` are excluded strictly.  Two coupled constraints,
    marked [`DERIVATION`][qsimod.parameters.ConstraintOrigin], bound the tilt by the fraction
    ``tilt_fraction`` of the superlattice depth and of the resonance gap ``U - delta``.  They
    belong to the superlattice derivation; a second theory on the same lattice poses its request
    against [`apparatus_only`][qsimod.parameters.AdmissibleSet.apparatus_only].

    Args:
        namespace: the namespace whose knobs are constrained.
        tunnelling_max: the largest reachable ``J``.
        interaction_range: the reachable range of ``U``.
        superlattice_max: the largest reachable ``delta``.
        tilt_max: the largest reachable ``Delta``.
        tilt_fraction: the largest fraction of ``delta`` and of ``U - delta`` the tilt may take.

    Returns:
        The admissible set.

    """
    low, high = interaction_range
    superlattice = namespace.symbol(names.SUPERLATTICE)
    interaction = namespace.symbol(names.INTERACTION)
    tilt = namespace.symbol(names.TILT)
    return AdmissibleSet(
        bounds=(
            Bound(namespace(names.TUNNELLING), 0.0, tunnelling_max, strict_lower=True),
            Bound(namespace(names.INTERACTION), low, high),
            Bound(namespace(names.SUPERLATTICE), 0.0, superlattice_max, strict_lower=True),
            Bound(namespace(names.TILT), 0.0, tilt_max, strict_lower=True),
        ),
        constraints=(
            InequalityConstraint(
                name="tilt within the superlattice depth",
                expression=tilt_fraction * superlattice - tilt,
                description="the tilt is at most a fraction of the superlattice depth",
                origin=ConstraintOrigin.DERIVATION,
            ),
            InequalityConstraint(
                name="tilt within the resonance gap",
                expression=tilt_fraction * (interaction - superlattice) - tilt,
                description="the tilt is at most a fraction of the resonance gap U - delta",
                origin=ConstraintOrigin.DERIVATION,
            ),
        ),
        description="optical superlattice simulator",
    )


def tilted_bose_hubbard_chain(
    matter_sites: int,
    namespace: Namespace,
    *,
    admissible_set: AdmissibleSet | None = None,
    name: str = "H_BHM",
    hamiltonian_name: str = "",
    tunnelling_sign: int = -1,
    staggered: bool = True,
    tilted: bool = True,
) -> HamiltonianModel:
    """A tilted, staggered Bose-Hubbard chain on an optical superlattice.

    The model is the analogue simulator model ``H_sim`` of the case study of the article, at
    the artifact ``L3a``, with the knob set ``Theta_sim = {J, U, delta, Delta}``:

    ```
    H = sign * J sum_{j=0}^{2N-3} ( b^dag_j b_{j+1} + h.c. )
      + sum_{j=0}^{2N-2} [ (U/2) n_j (n_j - 1) + eps_j n_j ],
    eps_j = (-1)**j delta/2 + j Delta
    ```

    The perturbative manifold ``|101> <-> |020>`` of each three-site block is near-degenerate at
    ``U ~= 2 delta``; it is not the ground-state manifold.

    Args:
        matter_sites: the chain length ``N``; the lattice has ``2N - 1`` sites.
        namespace: the namespace of the knobs ``J``, ``U``, ``delta`` and ``Delta``.
        admissible_set: the knob limits of the simulator; defaults to
            [`bose_hubbard_admissible_set`][qsimod.models.hardware.bose_hubbard_admissible_set].
        name: the name of the model.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.
        tunnelling_sign: ``-1`` for the tunnelling term ``-J(...)`` of Yang et al. (2020),
            ``+1`` for ``+J(...)`` of Zhou et al. (2022); the sign is a gauge choice on the ``b``
            operators.
        staggered: whether the superlattice term is present.
        tilted: whether the linear tilt is present.

    Returns:
        The model, in the hardware layer.

    """
    tunnelling = namespace.symbol(names.TUNNELLING)
    interaction = namespace.symbol(names.INTERACTION)
    superlattice = namespace.symbol(names.SUPERLATTICE)
    tilt = namespace.symbol(names.TILT)
    sites = interleaved_site_count(matter_sites)

    hamiltonian = OperatorSum()
    for site in range(sites - 1):
        hamiltonian = (
            hamiltonian
            + word(tunnelling_sign * tunnelling, create(site), annihilate(site + 1)).plus_adjoint()
        )
    for site in range(sites):
        hamiltonian = hamiltonian + on_site_interaction(site, interaction)
        offset = ((-1) ** site * superlattice / 2 if staggered else 0) + (
            site * tilt if tilted else 0
        )
        hamiltonian = hamiltonian + word(offset, number(site))

    constraints = gauss_operators(matter_sites, field=ElectricField.BOSON)
    return HamiltonianModel(
        name=name,
        structure=matter_gauge_structure(
            matter_sites, Algebra.BOSON, Algebra.BOSON, name=hamiltonian_name or name
        ),
        level=AbstractionLevel.HARDWARE,
        parameters=namespace.parameter_set(
            (names.TUNNELLING, Dimension.ENERGY, "nearest-neighbour tunnelling J"),
            (names.INTERACTION, Dimension.ENERGY, "on-site interaction U"),
            (names.SUPERLATTICE, Dimension.ENERGY, "staggered superlattice depth delta"),
            (names.TILT, Dimension.ENERGY, "linear tilt Delta"),
        ),
        admissible_set=admissible_set or bose_hubbard_admissible_set(namespace),
        hamiltonian=hamiltonian.renamed(hamiltonian_name or name),
        constraint_operators=constraints,
        sector=gauss_sector(constraints),
        local_subspace=local_occupation_subspace(matter_sites),
        origin="tilted, staggered Bose-Hubbard chain on a 1D optical superlattice",
    )


# ---------------------------------------------------------------------------
# A second, unrelated device: the transverse-field Ising chain
# ---------------------------------------------------------------------------

#: A chain of qubits with one role and no gauge sector.
SPIN_CHAIN_PATTERN: StructurePattern = open_chain_pattern(
    "a chain of qubits with no gauge sector",
    (DofRequirement("spins", frozenset({Algebra.QUBIT}), min_sites=2),),
)


def ising_admissible_set(
    namespace: Namespace,
    *,
    coupling_max: float = 5.0,
    field_max: float = 2.0,
) -> AdmissibleSet:
    """The admissible knob set of a transverse-field Ising simulator."""
    return AdmissibleSet(
        bounds=(
            Bound(namespace(names.LONGITUDINAL_COUPLING), 0.0, coupling_max, strict_lower=True),
            Bound(namespace(names.FIELD), 0.0, field_max),
        ),
        description="transverse-field Ising simulator",
    )


def transverse_field_ising_chain(
    sites: int,
    namespace: Namespace,
    *,
    admissible_set: AdmissibleSet | None = None,
    name: str = "H_TFIM",
    hamiltonian_name: str = "",
) -> HamiltonianModel:
    """A transverse-field Ising chain of qubits, ``Jz sum ZZ + h sum X``.

    Args:
        sites: the number of qubits of the chain.
        namespace: the namespace of the knobs ``Jz`` and ``h``.
        admissible_set: the knob limits of the simulator; defaults to
            [`ising_admissible_set`][qsimod.models.hardware.ising_admissible_set].
        name: the name of the model.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.

    Returns:
        The model, in the hardware layer.

    """
    coupling = namespace.symbol(names.LONGITUDINAL_COUPLING)
    field = namespace.symbol(names.FIELD)

    hamiltonian = OperatorSum()
    for site in range(sites - 1):
        hamiltonian = hamiltonian + word(coupling, pauli_z(site), pauli_z(site + 1))
    for site in range(sites):
        hamiltonian = hamiltonian + word(field, pauli_x(site))

    return HamiltonianModel(
        name=name,
        structure=StructureType(
            lattice=Lattice(LatticeGeometry.OPEN_CHAIN_1D, sites, link_count=0),
            degrees_of_freedom=(DegreeOfFreedom("spins", Algebra.QUBIT, tuple(range(sites))),),
            name=hamiltonian_name or name,
        ),
        level=AbstractionLevel.HARDWARE,
        parameters=namespace.parameter_set(
            (
                names.LONGITUDINAL_COUPLING,
                Dimension.ENERGY,
                "nearest-neighbour ZZ coupling",
            ),
            (names.FIELD, Dimension.ENERGY, "transverse field"),
        ),
        admissible_set=admissible_set or ising_admissible_set(namespace),
        hamiltonian=hamiltonian.renamed(hamiltonian_name or name),
        origin="transverse-field Ising chain",
    )


# ---------------------------------------------------------------------------
# A second optical lattice: the two-component Bose-Hubbard chain
# ---------------------------------------------------------------------------

#: Two bosonic components on an open chain with conserved magnetisation.
TWO_COMPONENT_CHAIN_PATTERN: StructurePattern = two_component_pattern(
    "two bosonic components on an open 1D chain, with conserved magnetisation"
)

#: The default knob limits of the two-component simulator, in rad/ms: lithium-7 in a lattice of
#: spacing 532 nm at 9 to 13 recoil energies (Jepsen et al. 2020), extended by a margin.
DEFAULT_HOPPING_MAX = 8.0
DEFAULT_INTERACTION_MAGNITUDE_RANGE = (5.0, 260.0)

#: The ratio ``|U| / t`` at which the unit-filling Mott lobe of the 1D Bose-Hubbard chain closes.
DEFAULT_MOTT_RATIO = 3.4


def two_component_admissible_set(
    namespace: Namespace,
    *,
    hopping_max: float = DEFAULT_HOPPING_MAX,
    interaction_magnitude_range: tuple[float, float] = DEFAULT_INTERACTION_MAGNITUDE_RANGE,
    mott_ratio: float = DEFAULT_MOTT_RATIO,
) -> AdmissibleSet:
    """The admissible knob set of a two-component optical-lattice simulator.

    The value ``t = 0`` is excluded strictly.  The interactions ``U_uu`` and ``U_ud`` are
    confined to the attractive branch of the Feshbach resonance, so that
    ``Jxy = -4 t**2 / U_ud > 0``; ``U_dd`` may take either sign, and an anisotropy below ``-1``
    requires it to be repulsive.  Three coupled constraints ``U**2 >= (mott_ratio * t)**2`` keep
    the Mott gap of every channel above the tunnelling.  The box of ``U_dd`` contains the pole
    ``U_dd = 0``, so a solve is to be given an ``initial`` point on the intended side of the
    pole.

    Args:
        namespace: the namespace whose knobs are constrained.
        hopping_max: the largest reachable ``t``.
        interaction_magnitude_range: the reachable range of ``|U|``.
        mott_ratio: the ratio ``|U| / t`` at which the unit-filling Mott lobe closes.

    Returns:
        The admissible set.

    """
    weakest, strongest = interaction_magnitude_range
    hopping = namespace.symbol(names.HOPPING)
    channels = (names.INTERACTION_UP, names.INTERACTION_MIXED, names.INTERACTION_DOWN)
    return AdmissibleSet(
        bounds=(
            Bound(namespace(names.HOPPING), 0.0, hopping_max, strict_lower=True),
            Bound(namespace(names.INTERACTION_UP), -strongest, -weakest),
            Bound(namespace(names.INTERACTION_MIXED), -strongest, -weakest),
            Bound(namespace(names.INTERACTION_DOWN), -strongest, strongest),
        ),
        constraints=tuple(
            InequalityConstraint(
                name=f"{channel} above the Mott lobe",
                # Squared: smooth at the origin and sign-blind.
                expression=namespace.symbol(channel) ** 2.0 - (mott_ratio * hopping) ** 2.0,
                description=(
                    f"|{channel}| >= {mott_ratio:g} t, so the sample is a unit-filling Mott "
                    "insulator"
                ),
            )
            for channel in channels
        ),
        description="two-component optical-lattice simulator",
    )


def two_component_bose_hubbard_chain(
    sites: int,
    namespace: Namespace,
    *,
    admissible_set: AdmissibleSet | None = None,
    name: str = "H_2BHM",
    hamiltonian_name: str = "",
) -> HamiltonianModel:
    """A two-component Bose-Hubbard chain: two hyperfine states in one optical lattice.

    ```
    H = -t sum_{sigma} sum_{j=0}^{N-2} ( b^dag_{j,sigma} b_{j+1,sigma} + h.c. )
      + (U_uu/2) sum_j n_{j,up} (n_{j,up} - 1)
      + (U_dd/2) sum_j n_{j,down} (n_{j,down} - 1)
      + U_ud     sum_j n_{j,up} n_{j,down}
    ```

    The model is defined on the register of
    [`two_component_structure`][qsimod.models.magnetism.two_component_structure], with the
    component ``up`` at ``2j`` and ``down`` at ``2j+1``.  It declares the conserved total
    particle number, fixed at ``N`` by unit filling.  On the attractive branch the doubly
    occupied states lie below the one-atom-per-site manifold.

    Args:
        sites: the chain length ``N``; the register has ``2N`` positions.
        namespace: the namespace of the knobs ``t``, ``U_uu``, ``U_ud`` and ``U_dd``.
        admissible_set: the knob limits of the simulator; defaults to
            [`two_component_admissible_set`][qsimod.models.hardware.two_component_admissible_set].
        name: the name of the model.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.

    Returns:
        The model, in the hardware layer.

    """
    hopping = namespace.symbol(names.HOPPING)
    intra_up = namespace.symbol(names.INTERACTION_UP)
    intra_down = namespace.symbol(names.INTERACTION_DOWN)
    inter = namespace.symbol(names.INTERACTION_MIXED)

    hamiltonian = OperatorSum()
    for site in range(sites - 1):
        for index in (up_index, down_index):
            hamiltonian = (
                hamiltonian
                + word(-hopping, create(index(site)), annihilate(index(site + 1))).plus_adjoint()
            )
    for site in range(sites):
        up, down = up_index(site), down_index(site)
        for position, strength in ((up, intra_up), (down, intra_down)):
            hamiltonian = hamiltonian + on_site_interaction(position, strength)
        hamiltonian = hamiltonian + word(inter, number(up), number(down))

    return HamiltonianModel(
        name=name,
        structure=two_component_structure(sites, name=hamiltonian_name or name),
        level=AbstractionLevel.HARDWARE,
        parameters=namespace.parameter_set(
            (names.HOPPING, Dimension.ENERGY, "nearest-neighbour tunnelling t"),
            (names.INTERACTION_UP, Dimension.ENERGY, "on-site interaction U_uu"),
            (names.INTERACTION_MIXED, Dimension.ENERGY, "on-site interaction U_ud"),
            (names.INTERACTION_DOWN, Dimension.ENERGY, "on-site interaction U_dd"),
        ),
        admissible_set=admissible_set or two_component_admissible_set(namespace),
        hamiltonian=hamiltonian.renamed(hamiltonian_name or name),
        constraint_operators=unit_filling_operators(sites),
        sector=unit_filling_sector(sites),
        origin="two-component Bose-Hubbard chain on a 1D optical lattice",
    )
