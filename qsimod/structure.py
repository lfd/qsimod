"""Structural types: the geometry, degrees of freedom with their algebra, and declared symmetries.

The type is a symbolic representation of the physical and mathematical properties of the
operators without numerics; no Hilbert space is constructed.  A
[`StructureType`][qsimod.structure.StructureType] is compared structurally; a
[`StructurePattern`][qsimod.structure.StructurePattern] constrains only the aspects it lists
and yields [`StructureMismatch`][qsimod.structure.StructureMismatch] records on failure.

Site indexing is interleaved at every abstraction level: matter site ``l`` is register index
``2*l`` (``l = 0 .. N-1``) and gauge link ``(l, l+1)`` is register index ``2*l + 1``
(``l = 0 .. N-2``), so that ``N`` matter sites occupy ``2*N - 1`` register positions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Algebra",
    "DegreeOfFreedom",
    "DofRequirement",
    "Lattice",
    "LatticeGeometry",
    "StructureMismatch",
    "StructurePattern",
    "StructureType",
    "StructureTypeError",
    "SymmetryDeclaration",
    "interleaved_link_index",
    "interleaved_matter_index",
    "interleaved_site_count",
    "link_indices",
    "matter_indices",
    "open_chain_pattern",
    "open_chain_structure",
]


class Algebra(Enum):
    """The operator algebra a degree of freedom obeys."""

    FERMION = "fermion"
    """A spinless fermionic mode with canonical anticommutation relations and occupation in
    ``{0, 1}``."""

    BOSON = "boson"
    """A bosonic mode with canonical commutation relations and occupation in
    ``{0, 1, 2, ...}``."""

    SPIN_HALF = "spin-1/2"
    """A spin-1/2 degree of freedom, described by ``S^z`` and ``S^+-``."""

    QUBIT = "qubit"
    """A qubit, described by Pauli operators, with ``S^z = sigma^z / 2``; the algebra is distinct
    from ``SPIN_HALF``."""

    GAUGE_LINK_U1 = "gauge-link-U(1)"
    """An untruncated compact U(1) link with unbounded electric field and no finite-dimensional
    representation."""

    @property
    def local_dimension(self) -> int | None:
        """The dimension of the Hilbert space of one site, or ``None`` if not fixed by the algebra.

        The dimension of a bosonic mode is ``None``; it is fixed by an occupation cutoff at the
        numerical realisation.
        """
        return {
            Algebra.FERMION: 2,
            Algebra.SPIN_HALF: 2,
            Algebra.QUBIT: 2,
            Algebra.BOSON: None,
            Algebra.GAUGE_LINK_U1: None,
        }[self]

    @property
    def is_finite_dimensional(self) -> bool:
        """Whether the algebra has a finite-dimensional representation."""
        return self is not Algebra.GAUGE_LINK_U1


class LatticeGeometry(Enum):
    """The spatial arrangement of the sites."""

    OPEN_CHAIN_1D = "open 1D chain"
    PERIODIC_CHAIN_1D = "periodic 1D chain"
    OPEN_SQUARE_2D = "open 2D square lattice"

    @property
    def spatial_dimension(self) -> int:
        """The number of spatial dimensions."""
        return 2 if self is LatticeGeometry.OPEN_SQUARE_2D else 1


@dataclass(frozen=True)
class Lattice:
    """The lattice geometry of a model.

    Attributes:
        geometry: the arrangement of the sites.
        matter_sites: the number of sites carrying the primary degree of freedom.
        link_count: the number of additional link positions, or ``None`` to derive it from
            the geometry, ``N - 1`` on an open chain and ``N`` on a periodic one.  The value
            ``0`` declares a lattice without link degrees of freedom.

    """

    geometry: LatticeGeometry
    matter_sites: int
    link_count: int | None = None

    def __post_init__(self) -> None:
        if self.matter_sites < 2:
            msg = f"a chain needs at least two sites, got {self.matter_sites}"
            raise ValueError(msg)
        if self.link_count is not None and self.link_count < 0:
            msg = f"a lattice cannot have {self.link_count} links"
            raise ValueError(msg)

    @property
    def links(self) -> int:
        """The number of link positions."""
        if self.link_count is not None:
            return self.link_count
        if self.geometry is LatticeGeometry.PERIODIC_CHAIN_1D:
            return self.matter_sites
        return self.matter_sites - 1

    @property
    def register_size(self) -> int:
        """The number of interleaved register positions, ``2*N - 1`` on an open chain."""
        return self.matter_sites + self.links

    @property
    def spatial_dimension(self) -> int:
        """The number of spatial dimensions."""
        return self.geometry.spatial_dimension

    def __str__(self) -> str:
        links = "" if self.links else " (no link positions)"
        return f"{self.geometry.value} of N={self.matter_sites} sites{links}"


@dataclass(frozen=True)
class DegreeOfFreedom:
    """One family of degrees of freedom: a role, an algebra and a set of sites.

    Attributes:
        role: the physical role of the family, for instance ``"matter"`` or ``"gauge"``.
        algebra: the operator algebra obeyed on these sites.
        sites: the register indices occupied, in ascending order.
        label: a description used in reports only.

    """

    role: str
    algebra: Algebra
    sites: tuple[int, ...]
    label: str = ""

    def __post_init__(self) -> None:
        if tuple(sorted(self.sites)) != self.sites:
            msg = f"sites of {self.role!r} must be given in ascending order"
            raise ValueError(msg)

    @property
    def count(self) -> int:
        """The number of sites in the family."""
        return len(self.sites)

    def __str__(self) -> str:
        return f"{self.role}: {self.count} x {self.algebra.value}"


@dataclass(frozen=True)
class SymmetryDeclaration:
    """A declared symmetry of a model; the generators are stored on the model.

    Attributes:
        name: the identifier under which the constraint operators of the model are stored.
        group: the symmetry group, for instance ``"U(1)"``.
        local: whether there is one generator per site, as for a gauge symmetry.

    """

    name: str
    group: str
    local: bool = True

    def __str__(self) -> str:
        return f"{'local' if self.local else 'global'} {self.group} ({self.name})"


@dataclass(frozen=True)
class StructureType:
    """The structural type of a model; ``==`` is structural equality.

    Attributes:
        lattice: the geometry.
        degrees_of_freedom: the degree-of-freedom families, with distinct roles.
        symmetries: the declared symmetries.
        name: a name used in reports.

    """

    lattice: Lattice
    degrees_of_freedom: tuple[DegreeOfFreedom, ...]
    symmetries: frozenset[SymmetryDeclaration] = field(default_factory=frozenset)
    name: str = ""

    def __post_init__(self) -> None:
        roles = [dof.role for dof in self.degrees_of_freedom]
        if len(roles) != len(set(roles)):
            msg = f"duplicate degree-of-freedom roles: {roles}"
            raise ValueError(msg)

    def dof(self, role: str) -> DegreeOfFreedom:
        """The degree-of-freedom family with the given role.

        Raises:
            KeyError: if no family has that role.

        """
        for candidate in self.degrees_of_freedom:
            if candidate.role == role:
                return candidate
        msg = f"structure has no degree of freedom with role {role!r}; has {self.roles()}"
        raise KeyError(msg)

    def roles(self) -> tuple[str, ...]:
        """The roles present, in declaration order."""
        return tuple(dof.role for dof in self.degrees_of_freedom)

    def dof_at(self, site: int) -> DegreeOfFreedom:
        """The degree-of-freedom family occupying a register position.

        Raises:
            KeyError: if no declared family occupies that site.

        """
        for dof in self.degrees_of_freedom:
            if site in dof.sites:
                return dof
        msg = f"no degree of freedom occupies register site {site}"
        raise KeyError(msg)

    def algebra_at(self, site: int) -> Algebra:
        """The algebra obeyed at a register position.

        Raises:
            KeyError: if no declared family occupies that site.

        """
        return self.dof_at(site).algebra

    def role_at(self, site: int) -> str:
        """The role of the family occupying a register position.

        Raises:
            KeyError: if no declared family occupies that site.

        """
        return self.dof_at(site).role

    @property
    def sites(self) -> tuple[int, ...]:
        """Every occupied register position, in ascending order."""
        return tuple(sorted(site for dof in self.degrees_of_freedom for site in dof.sites))

    @property
    def is_finite_dimensional(self) -> bool:
        """Whether every family admits a finite-dimensional representation."""
        return all(dof.algebra.is_finite_dimensional for dof in self.degrees_of_freedom)

    def symmetry(self, name: str) -> SymmetryDeclaration | None:
        """The declared symmetry of that name, or ``None``."""
        for declaration in self.symmetries:
            if declaration.name == name:
                return declaration
        return None

    def with_symmetry(self, declaration: SymmetryDeclaration) -> StructureType:
        """A copy with one more declared symmetry."""
        return StructureType(
            lattice=self.lattice,
            degrees_of_freedom=self.degrees_of_freedom,
            symmetries=self.symmetries | {declaration},
            name=self.name,
        )

    def __str__(self) -> str:
        parts = ", ".join(str(dof) for dof in self.degrees_of_freedom)
        symmetries = ", ".join(sorted(str(s) for s in self.symmetries)) or "none declared"
        head = self.name or "structure"
        return f"{head} [{self.lattice}; {parts}; symmetries: {symmetries}]"


# ---------------------------------------------------------------------------
# Patterns: the source side of a transformation's type
# ---------------------------------------------------------------------------


#: The wording, at composition time, for an aspect that the producing transformation does
#: not declare for its target.
_UNCONSTRAINED = "unconstrained by the producing transformation"
_NOT_GUARANTEED = "not guaranteed by the producing transformation"


def _choices(values: Iterable[str]) -> str:
    """Render a set of accepted values as ``"a or b or c"``, sorted for a stable output."""
    return " or ".join(sorted(values))


def _algebra_choices(algebras: Iterable[Algebra]) -> str:
    """The accepted algebras, rendered."""
    return _choices(algebra.value for algebra in algebras)


@dataclass(frozen=True)
class StructureMismatch:
    """One reason for which a structure fails to match a pattern.

    Attributes:
        aspect: the part of the type that did not match, for instance ``"lattice.geometry"``
            or ``"dof[matter].algebra"``.
        required: the requirement of the pattern, rendered for the diagnostic.
        found: the value found in the structure.

    """

    aspect: str
    required: str
    found: str

    def __str__(self) -> str:
        return f"{self.aspect}: requires {self.required}, found {self.found}"


class StructureTypeError(TypeError):
    """The diagnostic raised when the structure of a model does not match a required source type.

    Attributes:
        subject: the name of the checked object.
        mismatches: the individual mismatches.

    """

    def __init__(self, subject: str, mismatches: Sequence[StructureMismatch]) -> None:
        self.subject = subject
        self.mismatches = tuple(mismatches)
        detail = "; ".join(str(mismatch) for mismatch in self.mismatches)
        super().__init__(f"{subject}: {detail}")


@dataclass(frozen=True)
class DofRequirement:
    """The requirement of a pattern on one degree-of-freedom family.

    Attributes:
        role: the role that must be present.
        algebras: the accepted algebras; the empty set accepts any algebra.
        min_sites: the least number of sites accepted.

    """

    role: str
    algebras: frozenset[Algebra] = field(default_factory=frozenset)
    min_sites: int = 1

    def check(self, structure: StructureType) -> list[StructureMismatch]:
        """The mismatches of this requirement against ``structure``."""
        try:
            dof = structure.dof(self.role)
        except KeyError:
            return [
                StructureMismatch(
                    aspect=f"dof[{self.role}]",
                    required="present",
                    found=f"absent (roles present: {', '.join(structure.roles())})",
                )
            ]
        problems: list[StructureMismatch] = []
        if self.algebras and dof.algebra not in self.algebras:
            problems.append(
                StructureMismatch(
                    aspect=f"dof[{self.role}].algebra",
                    required=_algebra_choices(self.algebras),
                    found=dof.algebra.value,
                )
            )
        if dof.count < self.min_sites:
            problems.append(
                StructureMismatch(
                    aspect=f"dof[{self.role}].count",
                    required=f">= {self.min_sites}",
                    found=str(dof.count),
                )
            )
        return problems


@dataclass(frozen=True)
class StructurePattern:
    """A partial description of a structural type; unmentioned aspects are accepted.

    Attributes:
        geometries: the accepted lattice geometries; the empty set accepts any geometry.
        dofs: the requirements on individual degree-of-freedom families.
        required_symmetries: the names of symmetries that must be declared.
        forbidden_roles: the roles that must not be present.
        forbidden_symmetries: the names of symmetries that must not be declared.
        description: a summary used in diagnostics.

    """

    geometries: frozenset[LatticeGeometry] = field(default_factory=frozenset)
    dofs: tuple[DofRequirement, ...] = ()
    required_symmetries: frozenset[str] = field(default_factory=frozenset)
    forbidden_roles: frozenset[str] = field(default_factory=frozenset)
    forbidden_symmetries: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    def mismatches(self, structure: StructureType) -> list[StructureMismatch]:
        """Every mismatch of ``structure`` against the pattern, or an empty list."""
        problems: list[StructureMismatch] = []
        if self.geometries and structure.lattice.geometry not in self.geometries:
            problems.append(
                StructureMismatch(
                    aspect="lattice.geometry",
                    required=_choices(g.value for g in self.geometries),
                    found=structure.lattice.geometry.value,
                )
            )
        for requirement in self.dofs:
            problems.extend(requirement.check(structure))
        for name in sorted(self.required_symmetries):
            if structure.symmetry(name) is None:
                declared = ", ".join(sorted(s.name for s in structure.symmetries)) or "none"
                problems.append(
                    StructureMismatch(
                        aspect=f"symmetry[{name}]",
                        required="declared",
                        found=f"not declared (declared: {declared})",
                    )
                )
        for role in sorted(self.forbidden_roles):
            if role in structure.roles():
                problems.append(
                    StructureMismatch(
                        aspect=f"dof[{role}]",
                        required="absent",
                        found="present",
                    )
                )
        for name in sorted(self.forbidden_symmetries):
            if structure.symmetry(name) is not None:
                problems.append(
                    StructureMismatch(
                        aspect=f"symmetry[{name}]",
                        required="not declared",
                        found="declared",
                    )
                )
        return problems

    def matches(self, structure: StructureType) -> bool:
        """Whether ``structure`` matches the pattern."""
        return not self.mismatches(structure)

    def guarantee_gaps(self, demanded: StructurePattern) -> list[StructureMismatch]:
        """The aspects in which this pattern, taken as a declaration, may fail ``demanded``.

        The pattern is taken as the structural type the producing transformation declares
        for its target.

        Args:
            demanded: the source pattern required by the next transformation.

        Returns:
            The gaps; empty if the two patterns compose.

        """
        gaps: list[StructureMismatch] = []
        gaps += self._geometry_gap(demanded)
        gaps += self._dof_gaps(demanded)
        gaps += [
            StructureMismatch(
                aspect=f"symmetry[{name}]",
                required="declared",
                found=_NOT_GUARANTEED,
            )
            for name in sorted(demanded.required_symmetries - self.required_symmetries)
        ]
        gaps += [
            StructureMismatch(
                aspect=f"dof[{role}]",
                required="absent",
                found="guaranteed present by the producing transformation",
            )
            for role in sorted(demanded.forbidden_roles & {r.role for r in self.dofs})
        ]
        gaps += [
            StructureMismatch(
                aspect=f"symmetry[{name}]",
                required="not declared",
                found="guaranteed declared by the producing transformation",
            )
            for name in sorted(demanded.forbidden_symmetries & self.required_symmetries)
        ]
        return gaps

    def _geometry_gap(self, demanded: StructurePattern) -> list[StructureMismatch]:
        """The geometry gap where the geometries of this pattern are not a subset of the demand.

        An unconstrained declaration counts as a gap.
        """
        if not demanded.geometries:
            return []
        if self.geometries and self.geometries <= demanded.geometries:
            return []
        return [
            StructureMismatch(
                aspect="lattice.geometry",
                required=_choices(g.value for g in demanded.geometries),
                found=(
                    _choices(g.value for g in self.geometries)
                    if self.geometries
                    else _UNCONSTRAINED
                ),
            )
        ]

    def _dof_gaps(self, demanded: StructurePattern) -> list[StructureMismatch]:
        """The gaps on the demanded degree-of-freedom families, role by role."""
        guaranteed = {requirement.role: requirement for requirement in self.dofs}
        gaps: list[StructureMismatch] = []
        for demand in demanded.dofs:
            mine = guaranteed.get(demand.role)
            if mine is None:
                gaps.append(
                    StructureMismatch(
                        aspect=f"dof[{demand.role}]",
                        required="present",
                        found=_NOT_GUARANTEED,
                    )
                )
                continue
            if demand.algebras and not (mine.algebras and mine.algebras <= demand.algebras):
                gaps.append(
                    StructureMismatch(
                        aspect=f"dof[{demand.role}].algebra",
                        required=_algebra_choices(demand.algebras),
                        found=(
                            _algebra_choices(mine.algebras) if mine.algebras else _UNCONSTRAINED
                        ),
                    )
                )
            if mine.min_sites < demand.min_sites:
                gaps.append(
                    StructureMismatch(
                        aspect=f"dof[{demand.role}].count",
                        required=f">= {demand.min_sites}",
                        found=f">= {mine.min_sites} guaranteed",
                    )
                )
        return gaps

    def meets(self, demanded: StructurePattern) -> bool:
        """Whether this pattern, taken as a declaration, satisfies ``demanded``."""
        return not self.guarantee_gaps(demanded)

    def check(self, structure: StructureType, subject: str) -> None:
        """Raise a diagnostic if ``structure`` does not match the pattern.

        Args:
            structure: the structure to check.
            subject: the name used in the diagnostic, for instance the name of the
                transformation.

        Raises:
            StructureTypeError: on any mismatch.

        """
        problems = self.mismatches(structure)
        if problems:
            raise StructureTypeError(subject, problems)

    def __str__(self) -> str:
        return self.description or "structure pattern"


# ---------------------------------------------------------------------------
# The interleaved index convention
# ---------------------------------------------------------------------------


def interleaved_matter_index(matter_site: int) -> int:
    """The register index of matter site ``l``: ``2*l``."""
    return 2 * matter_site


def interleaved_link_index(left_matter_site: int) -> int:
    """The register index of the link ``(l, l+1)``: ``2*l + 1``."""
    return 2 * left_matter_site + 1


def interleaved_site_count(matter_sites: int) -> int:
    """The register size for ``N`` matter sites on an open chain: ``2*N - 1``."""
    return 2 * matter_sites - 1


def matter_indices(matter_sites: int) -> tuple[int, ...]:
    """The register indices of the matter sites, in ascending order."""
    return tuple(interleaved_matter_index(site) for site in range(matter_sites))


def link_indices(matter_sites: int) -> tuple[int, ...]:
    """The register indices of the links of an open chain, in ascending order."""
    return tuple(interleaved_link_index(site) for site in range(matter_sites - 1))


def open_chain_pattern(
    description: str,
    requirements: Iterable[DofRequirement],
    *,
    required_symmetries: Iterable[str] = (),
    forbidden_symmetries: Iterable[str] = (),
) -> StructurePattern:
    """A structural pattern over an open 1D chain, with requirements on its roles.

    Args:
        description: a summary used in diagnostics.
        requirements: the requirements on individual degree-of-freedom families.
        required_symmetries: the names of symmetries that must be declared.
        forbidden_symmetries: the names of symmetries that must not be declared.

    Returns:
        The pattern.

    """
    return StructurePattern(
        geometries=frozenset({LatticeGeometry.OPEN_CHAIN_1D}),
        dofs=tuple(requirements),
        required_symmetries=frozenset(required_symmetries),
        forbidden_symmetries=frozenset(forbidden_symmetries),
        description=description,
    )


def open_chain_structure(
    matter_sites: int,
    matter_algebra: Algebra,
    gauge_algebra: Algebra,
    *,
    name: str,
    symmetries: Iterable[SymmetryDeclaration] = (),
    matter_role: str = "matter",
    gauge_role: str = "gauge",
) -> StructureType:
    """Build an interleaved open-chain structure, matter on even and gauge on odd positions.

    Args:
        matter_sites: the number of matter sites ``N``.
        matter_algebra: the algebra on the even register positions.
        gauge_algebra: the algebra on the odd register positions.
        name: a name for the structural type, used in reports.
        symmetries: declared symmetries.
        matter_role: the role name for the matter family.
        gauge_role: the role name for the gauge family.

    Returns:
        The structural type.

    """
    return StructureType(
        lattice=Lattice(LatticeGeometry.OPEN_CHAIN_1D, matter_sites),
        degrees_of_freedom=(
            DegreeOfFreedom(
                role=matter_role,
                algebra=matter_algebra,
                sites=matter_indices(matter_sites),
                label="even register positions",
            ),
            DegreeOfFreedom(
                role=gauge_role,
                algebra=gauge_algebra,
                sites=link_indices(matter_sites),
                label="odd register positions",
            ),
        ),
        symmetries=frozenset(symmetries),
        name=name,
    )
