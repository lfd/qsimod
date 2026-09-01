"""Shared ingredients of the quantum-magnetism models: layouts, patterns, terms, observables.

Two plain-chain layouts occur: a **spin chain** of ``N`` sites at register positions
``0 .. N-1``, a [`Lattice`][qsimod.structure.Lattice] with ``link_count=0``, and a
**two-component chain** of ``N`` sites with two bosonic modes each, at positions ``2j`` and
``2j+1``.  Both carry the global U(1) symmetry of the total magnetisation ``sum_j S^z_j``.
"""

from __future__ import annotations

from collections.abc import Iterable

from qsimod.artifact import ConstraintOperator, Sector
from qsimod.scalar import Scalar, ScalarLike, as_scalar
from qsimod.structure import (
    Algebra,
    DegreeOfFreedom,
    DofRequirement,
    Lattice,
    LatticeGeometry,
    StructurePattern,
    StructureType,
    SymmetryDeclaration,
    open_chain_pattern,
)
from qsimod.symbolic import (
    OperatorSum,
    number,
    spin_minus,
    spin_plus,
    spin_z,
    word,
)

__all__ = [
    "DOWN_ROLE",
    "MAGNETISATION_SYMMETRY",
    "MOTT_FAMILY",
    "PARTICLE_FAMILY",
    "SPIN_ROLE",
    "UP_ROLE",
    "coordination_numbers",
    "down_index",
    "end_magnetisation_term",
    "ising_chain_terms",
    "magnetisation_observable",
    "mott_manifold_operators",
    "spin_chain_pattern",
    "spin_chain_structure",
    "two_component_pattern",
    "two_component_site_count",
    "two_component_structure",
    "unit_filling_operators",
    "unit_filling_sector",
    "up_index",
    "weighted_magnetisation_term",
    "xxz_chain_terms",
]

SPIN_ROLE = "spins"
"""The role of the single degree-of-freedom family of a plain chain."""

UP_ROLE = "up"
"""The role of the first component of a two-component chain."""

DOWN_ROLE = "down"
"""The role of the second component of a two-component chain."""

MOTT_FAMILY = "M"
"""The family name of the per-site occupation operators of a two-component chain."""

PARTICLE_FAMILY = "N"
"""The family name of the total particle-number operator of a two-component chain."""

MAGNETISATION_SYMMETRY = SymmetryDeclaration(name="magnetisation", group="U(1)", local=False)
"""The conservation of the total magnetisation ``sum_j S^z_j``, which reads
``(N_up - N_down) / 2`` on a two-component chain."""


# ---------------------------------------------------------------------------
# The two layouts
# ---------------------------------------------------------------------------


def spin_chain_structure(
    sites: int,
    algebra: Algebra,
    *,
    name: str,
    role: str = SPIN_ROLE,
    symmetries: Iterable[SymmetryDeclaration] = (MAGNETISATION_SYMMETRY,),
) -> StructureType:
    """A plain open chain of ``sites`` degrees of freedom of a single algebra.

    Args:
        sites: the chain length ``N``; the register positions are ``0 .. N-1``.
        algebra: the algebra at every position.
        name: the name of the structural type, used in reports.
        role: the role name of the family.
        symmetries: the symmetries declared; defaults to the magnetisation symmetry.  An
            Ising chain in a transverse field passes an empty tuple.

    Returns:
        The structural type.

    """
    return StructureType(
        lattice=Lattice(LatticeGeometry.OPEN_CHAIN_1D, sites, link_count=0),
        degrees_of_freedom=(
            DegreeOfFreedom(
                role=role,
                algebra=algebra,
                sites=tuple(range(sites)),
                label="one per chain site",
            ),
        ),
        symmetries=frozenset(symmetries),
        name=name,
    )


def spin_chain_pattern(
    description: str,
    algebras: Iterable[Algebra],
    *,
    role: str = SPIN_ROLE,
    require_magnetisation: bool = True,
    forbid_magnetisation: bool = False,
) -> StructurePattern:
    """The structural type a transformation requires of a plain chain of a single algebra.

    Args:
        description: the summary used in diagnostics.
        algebras: the algebras accepted at the positions of the chain.
        role: the role that must be present.
        require_magnetisation: whether the magnetisation symmetry must be declared.
        forbid_magnetisation: whether the magnetisation symmetry must be absent.  If both are
            false, either is accepted.

    Returns:
        The pattern.

    Raises:
        ValueError: if the symmetry is both required and forbidden.

    """
    if require_magnetisation and forbid_magnetisation:
        msg = "a pattern cannot both require and forbid the magnetisation symmetry"
        raise ValueError(msg)
    return open_chain_pattern(
        description,
        (DofRequirement(role, frozenset(algebras), min_sites=2),),
        required_symmetries=((MAGNETISATION_SYMMETRY.name,) if require_magnetisation else ()),
        forbidden_symmetries=((MAGNETISATION_SYMMETRY.name,) if forbid_magnetisation else ()),
    )


def up_index(site: int) -> int:
    """The register index of the first component at chain site ``j``, ``2*j``."""
    return 2 * site


def down_index(site: int) -> int:
    """The register index of the second component at chain site ``j``, ``2*j + 1``."""
    return 2 * site + 1


def two_component_site_count(sites: int) -> int:
    """The register size of a two-component chain of ``N`` sites, ``2N``."""
    return 2 * sites


def two_component_structure(sites: int, *, name: str) -> StructureType:
    """A chain of ``sites`` positions, each carrying two bosonic modes.

    The first component occupies position ``2j`` and the second position ``2j+1``; the
    positions of the second component are declared through the ``link_count`` of the lattice.

    Args:
        sites: the chain length ``N``.
        name: the name of the structural type, used in reports.

    Returns:
        The structural type, carrying
        [`MAGNETISATION_SYMMETRY`][qsimod.models.magnetism.MAGNETISATION_SYMMETRY].

    """
    return StructureType(
        lattice=Lattice(LatticeGeometry.OPEN_CHAIN_1D, sites, link_count=sites),
        degrees_of_freedom=(
            DegreeOfFreedom(
                role=UP_ROLE,
                algebra=Algebra.BOSON,
                sites=tuple(up_index(site) for site in range(sites)),
                label="even register positions",
            ),
            DegreeOfFreedom(
                role=DOWN_ROLE,
                algebra=Algebra.BOSON,
                sites=tuple(down_index(site) for site in range(sites)),
                label="odd register positions",
            ),
        ),
        symmetries=frozenset({MAGNETISATION_SYMMETRY}),
        name=name,
    )


def two_component_pattern(description: str) -> StructurePattern:
    """The structural type a transformation into a two-component bosonic chain requires."""
    return open_chain_pattern(
        description,
        (
            DofRequirement(UP_ROLE, frozenset({Algebra.BOSON}), min_sites=2),
            DofRequirement(DOWN_ROLE, frozenset({Algebra.BOSON}), min_sites=2),
        ),
        required_symmetries=(MAGNETISATION_SYMMETRY.name,),
    )


# ---------------------------------------------------------------------------
# Terms and observables the magnetism models share
# ---------------------------------------------------------------------------


def coordination_numbers(sites: int) -> tuple[int, ...]:
    """The nearest-neighbour bond count of each site of an open chain, ``(1, 2, ..., 2, 1)``."""
    return tuple(1 if site in {0, sites - 1} else 2 for site in range(sites))


def xxz_chain_terms(
    sites: int,
    transverse: Scalar,
    longitudinal: Scalar,
) -> OperatorSum:
    """The nearest-neighbour XXZ terms of an open spin-1/2 chain.

    ```
    sum_{j=0}^{N-2} [ (Jxy/2)( S^+_j S^-_{j+1} + h.c. ) + Jz S^z_j S^z_{j+1} ]
    ```

    Args:
        sites: the chain length ``N``.
        transverse: the spin-exchange coupling ``Jxy``.
        longitudinal: the ``ZZ`` coupling ``Jz``.

    Returns:
        The symbolic operator sum.

    """
    total = OperatorSum()
    for site in range(sites - 1):
        total = total + word(transverse / 2, spin_plus(site), spin_minus(site + 1)).plus_adjoint()
        total = total + word(longitudinal, spin_z(site), spin_z(site + 1))
    return total


def weighted_magnetisation_term(sites: int, strength: ScalarLike) -> OperatorSum:
    """The term ``strength * sum_j z_j S^z_j``, with ``z_j`` the coordination number of site ``j``.

    Since ``sum_j z_j S^z_j = 2 sum_j S^z_j - S^z_0 - S^z_{N-1}``, the term is a constant within
    a magnetisation sector plus a field on the two end spins.

    Args:
        sites: the chain length ``N``.
        strength: the field strength per unit coordination.

    Returns:
        The symbolic operator sum.

    """
    coefficient = as_scalar(strength)
    total = OperatorSum()
    for site, coordination in enumerate(coordination_numbers(sites)):
        total = total + word(coefficient * coordination, spin_z(site))
    return total


def ising_chain_terms(
    sites: int,
    coupling: Scalar,
    transverse: Scalar,
    longitudinal: Scalar,
) -> OperatorSum:
    """The nearest-neighbour Ising terms of an open spin-1/2 chain.

    ```
    Jz sum_{j=0}^{N-2} S^z_j S^z_{j+1}  -  Gamma sum_j S^x_j  -  B sum_j S^z_j
    ```

    with ``S^x = (S^+ + S^-)/2``; the chain is antiferromagnetic for ``Jz > 0``.

    Args:
        sites: the chain length ``N``.
        coupling: the ``ZZ`` coupling ``Jz``.
        transverse: the transverse field as an energy, ``Gamma``.
        longitudinal: the longitudinal field as an energy, ``B``.

    Returns:
        The symbolic operator sum.

    """
    total = OperatorSum()
    for site in range(sites - 1):
        total = total + word(coupling, spin_z(site), spin_z(site + 1))
    for site in range(sites):
        total = total + word(-transverse / 2, spin_plus(site))
        total = total + word(-transverse / 2, spin_minus(site))
        total = total + word(-longitudinal, spin_z(site))
    return total


def end_magnetisation_term(sites: int, strength: ScalarLike) -> OperatorSum:
    """The term ``strength * (S^z_0 + S^z_{N-1})``, a field on the two end spins only.

    Args:
        sites: the chain length ``N``.
        strength: the field on each end spin.

    Returns:
        The symbolic operator sum.

    """
    coefficient = as_scalar(strength)
    return word(coefficient, spin_z(0)) + word(coefficient, spin_z(sites - 1))


def magnetisation_observable(sites: int) -> OperatorSum:
    """The total magnetisation ``sum_j S^z_j`` of a spin chain."""
    total = OperatorSum()
    for site in range(sites):
        total = total + word(1.0, spin_z(site))
    return total.renamed("Sz_total")


def mott_manifold_operators(sites: int) -> tuple[ConstraintOperator, ...]:
    """The per-site occupation operators ``M_j = n_{j,up} + n_{j,down}``, targeted at one.

    The operators select the one-atom-per-site manifold in which the superexchange derivation
    expands.  The Hamiltonian does not commute with them, since a tunnelling event changes two
    of them at once, so they do not define a superselection sector;
    [`sector_projector`][qsimod.realise.build.sector_projector] nevertheless projects onto the
    manifold.

    Args:
        sites: the chain length ``N``.

    Returns:
        One operator per chain site, each with target eigenvalue one.

    """
    return tuple(
        ConstraintOperator(
            name=MOTT_FAMILY,
            index=site,
            operator=(
                word(1.0, number(up_index(site))) + word(1.0, number(down_index(site)))
            ).renamed(f"{MOTT_FAMILY}_{site}"),
            target_value=1.0,
            is_boundary=site in {0, sites - 1},
        )
        for site in range(sites)
    )


def unit_filling_operators(sites: int) -> tuple[ConstraintOperator, ...]:
    """The total particle number ``N = sum_j (n_{j,up} + n_{j,down})``, targeted at ``sites``.

    The total particle number is a conserved quantity, so the hardware Hamiltonian is built
    exactly inside this sector with [`SectorBasis`][qsimod.realise.sector.SectorBasis].

    Args:
        sites: the chain length ``N``, which equals the target particle number at unit filling.

    Returns:
        A family with a single member.

    """
    total = OperatorSum()
    for site in range(sites):
        total = total + word(1.0, number(up_index(site)))
        total = total + word(1.0, number(down_index(site)))
    return (
        ConstraintOperator(
            name=PARTICLE_FAMILY,
            index=0,
            operator=total.renamed(PARTICLE_FAMILY),
            target_value=float(sites),
        ),
    )


def unit_filling_sector(sites: int) -> Sector:
    """The superselection sector of unit filling, ``N_total = N``."""
    return Sector(
        name="unit filling",
        family=PARTICLE_FAMILY,
        values={0: float(sites)},
        description="one atom per site, of either component; the total is conserved exactly",
    )
