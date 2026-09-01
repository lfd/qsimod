"""Artifacts: the nodes of the model graph.

An artifact carries an operator, a structural type that describes the physical and mathematical
properties of a model, and a configurable parameter set.
[`Artifact`][qsimod.artifact.Artifact] is the abstract artifact with a structural type, a
parameter set and an [`ArtifactKind`][qsimod.artifact.ArtifactKind];
[`HamiltonianModel`][qsimod.artifact.HamiltonianModel] and
[`ProductFormulaModel`][qsimod.trotter.schedule.ProductFormulaModel] are its subclasses.
Constraint operators and the target sector are attributes of the model, not of its structural
type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum

from qsimod.levels import AbstractionLevel
from qsimod.parameters import EMPTY_ADMISSIBLE_SET, AdmissibleSet, Binding, ParameterSet
from qsimod.scalar import Scalar
from qsimod.structure import Algebra, StructureType
from qsimod.symbolic import OperatorSum

__all__ = [
    "AbstractionLevel",
    "Artifact",
    "ArtifactKind",
    "ArtifactKindError",
    "ConstraintOperator",
    "DroppedConstant",
    "HamiltonianModel",
    "LocalSubspace",
    "Sector",
    "as_hamiltonian",
]


class ArtifactKind(Enum):
    """The kind of object an artifact of the model graph represents."""

    HAMILTONIAN = "Hamiltonian"
    """A symbolic sum of operator products under which a quantum system evolves."""

    PRODUCT_FORMULA = "product formula"
    """An ordered product of k-local unitaries on an explicit register."""

    def __str__(self) -> str:
        return self.value


class ArtifactKindError(TypeError):
    """A transformation was applied to an artifact of the wrong kind.

    Attributes:
        subject: the transformation or operation that was attempted.
        expected: the artifact kind the subject requires.
        found: the artifact kind that was supplied.
        artifact: the name of the artifact in question.

    """

    def __init__(
        self,
        subject: str,
        expected: ArtifactKind,
        found: ArtifactKind,
        artifact: str,
    ) -> None:
        self.subject = subject
        self.expected = expected
        self.found = found
        self.artifact = artifact
        super().__init__(
            f"{subject}: expects a {expected} but {artifact} is a {found}; "
            "the two artifact kinds are not interchangeable"
        )


@dataclass(frozen=True)
class ConstraintOperator:
    """One declared constraint operator, for instance a Gauss operator ``G_l``.

    Attributes:
        name: the family name, shared by every index (``"G"``).
        index: the member of the family (``l``).
        operator: the operator in symbolic form.
        target_value: the eigenvalue the physical sector requires of this member; boundary
            members of an open chain may differ from the bulk.
        is_boundary: whether this member is located at an end of the chain.

    """

    name: str
    index: int
    operator: OperatorSum
    target_value: float
    is_boundary: bool = False

    @property
    def label(self) -> str:
        """A short label, for instance ``"G_2"``."""
        return f"{self.name}_{self.index}"

    def __str__(self) -> str:
        where = " (boundary)" if self.is_boundary else ""
        return f"{self.label} = {self.target_value:+.3g}{where}"


@dataclass(frozen=True)
class Sector:
    """A declared superselection sector: the target eigenvalues of a constraint family.

    Attributes:
        name: the name of the sector, used in reports.
        family: the constraint-operator family the sector refers to.
        values: the target eigenvalue per constraint index, boundary members included.
        description: a one-line description.

    """

    name: str
    family: str
    values: Mapping[int, float]
    description: str = ""

    def __str__(self) -> str:
        body = ", ".join(
            f"{self.family}_{index}={value:+g}" for index, value in sorted(self.values.items())
        )
        return f"{self.name} [{body}]"


@dataclass(frozen=True)
class DroppedConstant:
    """An additive constant discarded by a transformation on the way to a model.

    Attributes:
        description: the term that became the constant.
        value: the constant, in the energy unit of the package.

    """

    description: str
    value: Scalar

    def __str__(self) -> str:
        return f"{self.description} = {self.value}"


@dataclass(frozen=True)
class LocalSubspace:
    """A declared per-site occupation subspace, for instance the ``{0,1}`` / ``{0,2}`` subspace.

    Attributes:
        name: a name used in reports.
        allowed_occupations: the occupations permitted at each register position.
        description: a one-line description.

    """

    name: str
    allowed_occupations: Mapping[int, tuple[int, ...]]
    description: str = ""

    @property
    def dimension(self) -> int:
        """The dimension of the subspace, the product of the per-site occupation counts."""
        total = 1
        for occupations in self.allowed_occupations.values():
            total *= len(occupations)
        return total

    def required_cutoff(self) -> int:
        """The least bosonic occupation cutoff that represents the subspace."""
        return max(
            (max(occupations) for occupations in self.allowed_occupations.values()),
            default=0,
        )

    def __str__(self) -> str:
        return f"{self.name} (dim {self.dimension}, needs n_max >= {self.required_cutoff()})"


@dataclass(frozen=True)
class Artifact(ABC):
    """An artifact of the model graph: a typed, parametrised symbolic object.

    An artifact carries a structural type that describes the physical and mathematical
    properties of a model and a configurable parameter set; the subclasses carry the operator.

    Attributes:
        name: the name of the artifact, used in every report and diagnostic.
        structure: the structural type.
        level: the abstraction level of the model.
        parameters: the named parameters, with dimensions.
        binding: the values bound to parameters; the binding may be partial or empty.
        admissible_set: the values the parameters may take; non-trivial only for a model of
            the hardware layer.
        origin: a one-line note on the derivation of the artifact, for reports.

    """

    name: str
    structure: StructureType
    level: AbstractionLevel
    parameters: ParameterSet = field(default_factory=ParameterSet)
    binding: Binding = field(default_factory=Binding)
    admissible_set: AdmissibleSet = EMPTY_ADMISSIBLE_SET
    origin: str = ""

    @property
    @abstractmethod
    def kind(self) -> ArtifactKind:
        """The kind of the artifact."""

    @abstractmethod
    def support_structure(self) -> tuple[frozenset[int], ...]:
        """The register positions each constituent acts on, in order.

        For a Hamiltonian model, the support of each term; for a product formula, the support
        of each factor in schedule order.
        """

    @property
    def free_parameters(self) -> tuple[str, ...]:
        """The declared parameters that are not bound to a value."""
        return tuple(name for name in self.parameters.names if name not in self.binding)

    @property
    def is_fully_bound(self) -> bool:
        """Whether every declared parameter is bound to a value."""
        return not self.free_parameters

    def bind(self, **values: float) -> Artifact:
        """A copy of the artifact with additional parameter values bound.

        Raises:
            KeyError: if a name is not a declared parameter of the artifact.

        """
        return self.bind_all(values)

    def bind_all(self, values: Mapping[str, float]) -> Artifact:
        """A copy of the artifact with the given parameter values bound.

        Raises:
            KeyError: if a name is not a declared parameter of the artifact.

        """
        unknown = sorted(set(values) - set(self.parameters.names))
        if unknown:
            msg = f"{self.name} has no parameter(s) {unknown}; declared: {self.parameters.names}"
            raise KeyError(msg)
        return replace(self, binding=self.binding.merged(values))

    def environment(self) -> dict[str, float]:
        """The bound values, as a plain mapping for the evaluation of expressions.

        Raises:
            ValueError: if a declared parameter is unbound.

        """
        if not self.is_fully_bound:
            msg = f"{self.name} still has free parameters: {self.free_parameters}"
            raise ValueError(msg)
        return self.binding.as_dict()

    def __str__(self) -> str:
        bound = str(self.binding) or "unbound"
        return f"{self.name} <{self.level.label} {self.kind}> {self.structure.name or ''} [{bound}]"


@dataclass(frozen=True)
class HamiltonianModel(Artifact):
    """An artifact whose operator is a Hamiltonian, a symbolic sum of operator products.

    Attributes:
        hamiltonian: the symbolic sum.
        constraint_operators: the declared constraint operators, for instance Gauss operators.
        sector: the target superselection sector, if declared.
        local_subspace: the declared per-site occupation subspace, if any.
        dropped_constants: the additive constants discarded by the transformations that
            produced the model.

    """

    hamiltonian: OperatorSum = field(default_factory=OperatorSum)
    constraint_operators: tuple[ConstraintOperator, ...] = ()
    sector: Sector | None = None
    local_subspace: LocalSubspace | None = None
    dropped_constants: tuple[DroppedConstant, ...] = ()

    def __post_init__(self) -> None:
        """Reject a model whose operators are not legal on the algebras of their sites."""
        self.validate()

    @property
    def kind(self) -> ArtifactKind:
        """Always [`ArtifactKind.HAMILTONIAN`][qsimod.artifact.ArtifactKind.HAMILTONIAN]."""
        return ArtifactKind.HAMILTONIAN

    def support_structure(self) -> tuple[frozenset[int], ...]:
        """The support of each term, in the order in which the terms are written."""
        return tuple(term.support for term in self.hamiltonian.terms)

    @property
    def max_locality(self) -> int:
        """The largest number of register positions on which a term acts."""
        return self.hamiltonian.max_locality

    def constraints_named(self, family: str) -> tuple[ConstraintOperator, ...]:
        """Every declared constraint operator of the family ``family``."""
        return tuple(c for c in self.constraint_operators if c.name == family)

    def validate(self) -> None:
        """Check that the operators of the Hamiltonian are legal on the algebras of their sites.

        Raises:
            ValueError: for an illegal operator.

        """
        algebra_at = {site: self.structure.algebra_at(site) for site in self.structure.sites}
        self.hamiltonian.validate_against(algebra_at)
        for constraint in self.constraint_operators:
            constraint.operator.validate_against(algebra_at)

    def substituted(self) -> OperatorSum:
        """The Hamiltonian with every bound parameter substituted.

        Raises:
            ValueError: if a declared parameter is unbound.

        """
        return self.hamiltonian.substitute(self.environment())

    def total_dropped_constant(self) -> float:
        """The sum of the additive constants discarded on the way to the model.

        Raises:
            KeyError: if a constant refers to a parameter of an unbound source model.

        """
        environment = self.binding.as_dict()
        return sum(dropped.value.evaluate_real(environment) for dropped in self.dropped_constants)

    def pretty(self) -> str:
        """The Hamiltonian rendered with a per-site ladder glyph taken from the structure."""
        glyphs = {
            site: "ψ" if self.structure.algebra_at(site) is Algebra.FERMION else "b"
            for site in self.structure.sites
        }
        return self.hamiltonian.renamed(self.name).pretty(glyphs)


def as_hamiltonian(artifact: Artifact, subject: str = "this operation") -> HamiltonianModel:
    """Narrow an artifact to a Hamiltonian model after checking its kind.

    Args:
        artifact: the artifact to narrow.
        subject: the name reported in the diagnostic.

    Returns:
        The artifact, typed as a Hamiltonian model.

    Raises:
        ArtifactKindError: if the artifact is not a Hamiltonian model.

    """
    if isinstance(artifact, HamiltonianModel):
        return artifact
    raise ArtifactKindError(subject, ArtifactKind.HAMILTONIAN, artifact.kind, artifact.name)
