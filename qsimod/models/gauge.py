"""Shared ingredients of the lattice-gauge models: generators, sectors, subspaces, observables.

Every function assumes the interleaved layout of [`qsimod.structure`][qsimod.structure]: the
matter site ``l`` occupies register index ``2l`` and the gauge link ``(l, l+1)`` occupies
register index ``2l + 1``.

Conventions: the link operator ``U ~ S^+`` raises the electric field, which fixes the relative
sign of the two ``E`` in the Gauss generator ``G_l``; the charge enters ``G_l`` with the
coefficient ``e``, and the factor one half belongs to the constant alone; on an open chain the
boundary generators omit one link, take values in ``{-1/2, +1/2, +3/2}``, and the declared
background is ``+1/2``, the sector of the canonical state ``|1 0 1 0 1 ...>``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from enum import Enum
from typing import assert_never

from qsimod.artifact import ConstraintOperator, LocalSubspace, Sector
from qsimod.scalar import Scalar, ScalarLike, as_scalar
from qsimod.structure import (
    Algebra,
    DofRequirement,
    StructurePattern,
    StructureType,
    SymmetryDeclaration,
    interleaved_link_index,
    interleaved_matter_index,
    link_indices,
    matter_indices,
    open_chain_pattern,
    open_chain_structure,
)
from qsimod.symbolic import (
    OperatorSum,
    constant,
    electric_field,
    number,
    pauli_z,
    spin_z,
    word,
)

__all__ = [
    "BOUNDARY_GAUSS_VALUE",
    "GAUGE_ROLE",
    "GAUSS_FAMILY",
    "GAUSS_SYMMETRY",
    "MATTER_ROLE",
    "POLYNOMIAL_TOLERANCE",
    "ElectricField",
    "GaussForm",
    "canonical_state_configuration",
    "gauge_violation_observable",
    "gauss_operators",
    "gauss_sector",
    "link_triples",
    "local_occupation_subspace",
    "matter_gauge_pattern",
    "matter_gauge_structure",
    "matter_occupation_observable",
    "occupation_projector",
]

#: The role names of the two degree-of-freedom families of the interleaved layout.
MATTER_ROLE = "matter"
GAUGE_ROLE = "gauge"

#: The family name under which the Gauss operators of a model are registered.
GAUSS_FAMILY = "G"

#: The declared local U(1) gauge symmetry of Gauss's law.
GAUSS_SYMMETRY = SymmetryDeclaration(name="gauss", group="U(1)", local=True)

#: Coefficients of the interpolating polynomial below this magnitude are treated as exact zeros.
POLYNOMIAL_TOLERANCE = 1e-12

#: The background Gauss eigenvalue at each end of an open chain, in the homogeneous form.
BOUNDARY_GAUSS_VALUE = 0.5


class ElectricField(Enum):
    """The operator by which an abstraction level writes the electric field on a link."""

    OPERATOR = "E"
    """The untruncated compact U(1) field operator ``E``, with unbounded spectrum."""

    SPIN = "S^z"
    """The operator ``S^z`` on a spin-1/2 link, with eigenvalues ``+-1/2``."""

    QUBIT = "sigma^z / 2"
    """The operator ``sigma^z / 2`` on a qubit, with eigenvalues ``+-1/2``."""

    BOSON = "(n - 1)/2"
    """The operator ``(n - 1)/2`` on a bosonic link whose occupations are ``{0, 2}``."""

    def __str__(self) -> str:
        return self.value


class GaussForm(Enum):
    """The generator form taken by the Gauss operators of an abstraction level."""

    STAGGERED = "staggered"
    """The form ``G_l = E_{l,l+1} - E_{l-1,l} - e [ n_l - (1 - (-1)**l)/2 ]`` before a
    particle-hole transformation.  The target eigenvalue is zero in the bulk and ``(-1)**(l+1)``
    times the background at the ends, which is the image of the homogeneous sector under the
    particle-hole map ``V G_l^st V^dag = (-1)**(l+1) G_l``."""

    HOMOGENEOUS = "homogeneous"
    """The form ``G_l = S^z_{l-1,l} + S^z_{l,l+1} + n_l`` after a particle-hole
    transformation.  The target eigenvalue is zero in the bulk and the background at the ends."""

    def __str__(self) -> str:
        return self.value


def matter_gauge_structure(
    matter_sites: int,
    matter_algebra: Algebra,
    gauge_algebra: Algebra,
    *,
    name: str,
    gauge_symmetry: bool = True,
) -> StructureType:
    """The interleaved matter-gauge structural type of an open chain.

    Args:
        matter_sites: the chain length ``N``; the register has ``2N - 1`` positions.
        matter_algebra: the algebra on the even positions.
        gauge_algebra: the algebra on the odd positions.
        name: the name of the structural type, used in reports.
        gauge_symmetry: whether the local U(1) symmetry is declared.

    Returns:
        The structural type.

    """
    return open_chain_structure(
        matter_sites,
        matter_algebra=matter_algebra,
        gauge_algebra=gauge_algebra,
        name=name,
        symmetries=[GAUSS_SYMMETRY] if gauge_symmetry else [],
        matter_role=MATTER_ROLE,
        gauge_role=GAUGE_ROLE,
    )


def matter_gauge_pattern(
    description: str,
    *,
    matter_algebras: Iterable[Algebra],
    gauge_algebras: Iterable[Algebra],
    require_gauge_symmetry: bool = True,
    minimum_sites: int = 2,
) -> StructurePattern:
    """A structural pattern over the interleaved matter-gauge chain.

    Args:
        description: the summary used in diagnostics.
        matter_algebras: the algebras accepted on the matter positions.
        gauge_algebras: the algebras accepted on the gauge positions.
        require_gauge_symmetry: whether the local U(1) symmetry must be declared.
        minimum_sites: the least number of matter sites accepted.

    Returns:
        The pattern.

    """
    return open_chain_pattern(
        description,
        (
            DofRequirement(MATTER_ROLE, frozenset(matter_algebras), min_sites=minimum_sites),
            DofRequirement(GAUGE_ROLE, frozenset(gauge_algebras), min_sites=1),
        ),
        required_symmetries=[GAUSS_SYMMETRY.name] if require_gauge_symmetry else [],
    )


def link_triples(matter_sites: int) -> Iterator[tuple[int, int, int, int]]:
    """Each link of an open chain as the indices ``(link, left matter, link, right matter)``."""
    for link in range(matter_sites - 1):
        yield (
            link,
            interleaved_matter_index(link),
            interleaved_link_index(link),
            interleaved_matter_index(link + 1),
        )


def _electric_field_term(link_site: int, style: ElectricField) -> OperatorSum:
    """The electric field on a link, written in the operator language of ``style``."""
    if style is ElectricField.OPERATOR:
        return word(1, electric_field(link_site))
    if style is ElectricField.SPIN:
        return word(1, spin_z(link_site))
    if style is ElectricField.QUBIT:
        return word(0.5, pauli_z(link_site))
    if style is ElectricField.BOSON:
        return word(0.5, number(link_site)) + word(-0.5)
    assert_never(style)


def gauss_operators(
    matter_sites: int,
    *,
    field: ElectricField,
    form: GaussForm = GaussForm.HOMOGENEOUS,
    boundary_value: float = BOUNDARY_GAUSS_VALUE,
    charge: ScalarLike = 1,
) -> tuple[ConstraintOperator, ...]:
    """The Gauss operators of an abstraction level, with their declared target eigenvalues.

    On an open chain the two boundary generators omit one link and take half-integer values;
    their targets are the background ``boundary_value`` in the homogeneous form and its
    particle-hole image ``(-1)**(l+1) * boundary_value`` in the staggered form.

    Args:
        matter_sites: the chain length ``N``.
        field: the operator by which the level writes the electric field.
        form: the generator form taken by the Gauss operators of the level.
        boundary_value: the background eigenvalue at the chain ends, in the homogeneous form.
        charge: the coefficient of the charge term in the staggered form: the gauge coupling
            ``e`` in the application layer, ``1`` below the quantum-link truncation.

    Returns:
        One [`ConstraintOperator`][qsimod.artifact.ConstraintOperator] per matter site.

    """
    operators: list[ConstraintOperator] = []
    for site in range(matter_sites):
        left = interleaved_link_index(site - 1) if site > 0 else None
        right = interleaved_link_index(site) if site < matter_sites - 1 else None
        boundary = site in {0, matter_sites - 1}

        if form is GaussForm.STAGGERED:
            total = _staggered_generator(site, left, right, field, as_scalar(charge))
            target = (-1) ** (site + 1) * boundary_value if boundary else 0.0
        else:
            total = _homogeneous_generator(site, left, right, field)
            target = boundary_value if boundary else 0.0

        operators.append(
            ConstraintOperator(
                name=GAUSS_FAMILY,
                index=site,
                operator=total.collected().renamed(f"{GAUSS_FAMILY}_{site}"),
                target_value=target,
                is_boundary=boundary,
            )
        )
    return tuple(operators)


def _staggered_generator(
    site: int,
    left: int | None,
    right: int | None,
    field: ElectricField,
    charge: Scalar,
) -> OperatorSum:
    """``G_l = E_{l,l+1} - E_{l-1,l} - e [n_l - (1 - (-1)**l)/2]``, with a missing link omitted."""
    total = word(-charge, number(interleaved_matter_index(site))) + word(
        charge * (1 - (-1) ** site) / 2
    )
    if right is not None:
        total = total + _electric_field_term(right, field)
    if left is not None:
        total = total - _electric_field_term(left, field)
    return total


def _homogeneous_generator(
    site: int,
    left: int | None,
    right: int | None,
    field: ElectricField,
) -> OperatorSum:
    """``G_l = S^z_{l-1,l} + S^z_{l,l+1} + n_l``, with a missing link omitted."""
    total = word(1, number(interleaved_matter_index(site)))
    for link in (left, right):
        if link is not None:
            total = total + _electric_field_term(link, field)
    return total


def gauss_sector(operators: Sequence[ConstraintOperator]) -> Sector:
    """The superselection sector declared by the target values of a Gauss family.

    Raises:
        ValueError: if the sequence of operators is empty.

    """
    if not operators:
        msg = "cannot describe a sector without any constraint operators"
        raise ValueError(msg)
    return Sector(
        name="Gauss-law sector",
        family=GAUSS_FAMILY,
        values={operator.index: operator.target_value for operator in operators},
        description="g_l = 0 in the bulk; the boundary generators at their background values",
    )


def local_occupation_subspace(
    matter_sites: int,
    *,
    matter_occupations: tuple[int, ...] = (0, 1),
    gauge_occupations: tuple[int, ...] = (0, 2),
    name: str = "local occupation subspace P",
) -> LocalSubspace:
    """A declared per-site occupation subspace on the interleaved chain.

    The default is the doublon encoding, ``{0, 1}`` on matter positions and ``{0, 2}`` on gauge
    positions, for which a numerical realisation requires an occupation cutoff of at least two.

    Args:
        matter_sites: the chain length ``N``.
        matter_occupations: the occupations allowed on matter positions.
        gauge_occupations: the occupations allowed on gauge positions.
        name: the name of the subspace, used in reports.

    Returns:
        The subspace.

    """
    allowed: dict[int, tuple[int, ...]] = dict.fromkeys(
        matter_indices(matter_sites), matter_occupations
    )
    allowed.update(dict.fromkeys(link_indices(matter_sites), gauge_occupations))
    return LocalSubspace(
        name=name,
        allowed_occupations=allowed,
        description=(
            f"matter positions in {list(matter_occupations)}, gauge positions in "
            f"{list(gauge_occupations)}; the encoding's operator identities hold as P B P"
        ),
    )


def canonical_state_configuration(matter_sites: int) -> tuple[int, ...]:
    """The occupation pattern ``|1 0 1 0 1 ...>``: matter sites occupied, gauge links empty."""
    return tuple(1 if site % 2 == 0 else 0 for site in range(2 * matter_sites - 1))


def matter_occupation_observable(matter_sites: int) -> OperatorSum:
    """The mean matter occupation ``<n_matter> = (1/N) sum_l n_{2l}``."""
    total = OperatorSum()
    for site in matter_indices(matter_sites):
        total = total + word(1.0 / matter_sites, number(site))
    return total.renamed("<n_matter>")


def occupation_projector(
    site: int,
    occupations: Iterable[int],
    cutoff: int,
    coefficient: float = 1.0,
) -> OperatorSum:
    """The projector onto a set of occupations at one site, as a polynomial in ``n``.

    The polynomial ``P_k = prod_{j != k} (n - j) / (k - j)`` interpolates over the ladder
    ``{0, ..., cutoff}``, and ``P_A = sum_{k in A} P_k``.  For ``A = {1}`` at ``cutoff = 2`` the
    projector is ``2n - n**2``, the parity ``n mod 2``.

    Args:
        site: the register position.
        occupations: the occupations projected onto.
        cutoff: the largest occupation carried by the ladder, ``n_max``.
        coefficient: an overall prefactor.

    Returns:
        The projector, as a sum of powers of ``n`` at the site.

    Raises:
        ValueError: if an occupation lies outside the ladder.

    """
    ladder = tuple(range(cutoff + 1))
    selected = tuple(occupations)
    if any(value not in ladder for value in selected):
        msg = f"occupations {selected} are not all within the ladder 0..{cutoff}"
        raise ValueError(msg)

    coefficients = [0.0] * (cutoff + 1)
    for chosen in selected:
        polynomial = [1.0]
        for other in ladder:
            if other == chosen:
                continue
            scale = 1.0 / (chosen - other)
            raised = [0.0, *(value * scale for value in polynomial)]
            lowered = [*(-other * scale * value for value in polynomial), 0.0]
            polynomial = [left + right for left, right in zip(raised, lowered, strict=True)]
        for power, value in enumerate(polynomial):
            coefficients[power] += value

    total = OperatorSum()
    for power, value in enumerate(coefficients):
        if abs(value) < POLYNOMIAL_TOLERANCE:
            continue
        weighted = coefficient * value
        total = total + (
            constant(weighted)
            if power == 0
            else word(weighted, *(number(site) for _ in range(power)))
        )
    return total


def gauge_violation_observable(
    matter_sites: int,
    subspace: LocalSubspace | None = None,
    name: str = "<eta>",
) -> OperatorSum:
    """The mean weight on forbidden gauge-link occupations, ``(1/L_g) sum_{j in g} (1 - P_j)``.

    This observable is the gauge violation ``eta`` of the article.  Under the doublon encoding a
    forbidden link occupation is an odd one, so the observable is the mean link parity
    ``n mod 2``.

    Args:
        matter_sites: the chain length ``N``.
        subspace: the declared subspace; defaults to
            [`local_occupation_subspace`][qsimod.models.gauge.local_occupation_subspace].
        name: the display name of the observable.

    Returns:
        The observable.

    """
    subspace = subspace or local_occupation_subspace(matter_sites)
    links = link_indices(matter_sites)
    cutoff = subspace.required_cutoff()
    weight = 1.0 / len(links)

    total = OperatorSum()
    for site in links:
        allowed = subspace.allowed_occupations[site]
        forbidden = tuple(value for value in range(cutoff + 1) if value not in allowed)
        total = total + occupation_projector(site, forbidden, cutoff, weight)
    return total.renamed(name)
