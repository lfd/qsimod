"""Transformations between abstraction levels, and their attributes.

A transformation is a typed edge between artifacts.  Like an artifact, a transformation is a
declaration before it is a computation.  A [`Transformation`][qsimod.transform.Transformation]
declares the source it requires (``source_pattern``, ``source_kind``), whether the operator form
of the target is unitarily equivalent to that of the source (``exactness``) and, if it is
approximate, the [`ApproximationKind`][qsimod.transform.ApproximationKind]: valid within a
declared regime of the parameters (``regime-limited``) or with an error controlled by a
resource parameter (``resource-controlled``).  The realisability of the coefficients is a
matter of the parameter relation and the admissible set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from qsimod.artifact import (
    Artifact,
    ArtifactKind,
    ArtifactKindError,
    DroppedConstant,
    HamiltonianModel,
)
from qsimod.levels import AbstractionLevel
from qsimod.normal_form import difference
from qsimod.parameters import Parameter, ParameterSet
from qsimod.relations import EMPTY_RELATION, ParameterRelation, RelationClassification, RelationKind
from qsimod.scalar import Scalar
from qsimod.structure import StructurePattern, StructureType
from qsimod.symbolic import OperatorSum
from qsimod.validity import Conjunction, ValidityReport

__all__ = [
    "Approximation",
    "ApproximationKind",
    "ArtifactKindError",
    "Exactness",
    "ExactnessError",
    "Transformation",
]


class ExactnessError(ValueError):
    """The declared target of an ``EXACT`` transformation is not the symbolic image of its source.

    Raised when the source Hamiltonian, rewritten in the operators of the target by
    [`Transformation.image`][qsimod.transform.Transformation.image], differs from the
    Hamiltonian of the target model by more than an additive constant.

    Attributes:
        transformation: the name of the transformation.
        defect: the surviving terms of ``target - image``, in normal form.

    """

    def __init__(self, transformation: str, defect: OperatorSum) -> None:
        self.transformation = transformation
        self.defect = defect
        lines = "\n".join(f"    {term}" for term in defect.terms)
        super().__init__(
            f"{transformation} is declared EXACT, but its target differs from the image of "
            f"its source by {len(defect.terms)} term(s):\n{lines}"
        )


class Exactness(Enum):
    """Whether the operator form of the target is unitarily equivalent to that of the source.

    ``EXACT`` may hold only after restriction to a stated subspace and up to a stated additive
    constant.  It makes no statement about the realisability of the coefficients.
    """

    EXACT = "EXACT"
    APPROXIMATE = "APPROXIMATE"

    def __str__(self) -> str:
        return self.value


class ApproximationKind(Enum):
    """The kind of approximation of an approximate transformation."""

    REGIME_LIMITED = "regime-limited"
    """Valid only within a declared regime of the parameters; the error cannot be driven to
    zero.  Reported as the margins of the regime conditions."""

    RESOURCE_CONTROLLED = "resource-controlled"
    """The error shrinks to zero as a resource parameter grows, at a cost.  Reported as the
    error against its cost."""

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Approximation:
    """The declaration that accompanies an ``APPROXIMATE`` transformation.

    Attributes:
        kind: ``regime-limited`` or ``resource-controlled``.
        leading_error_order: the leading surviving correction, where known, for instance
            ``"fourth order in J; relative error ~ (J/U)**2"``.
        resource_parameters: the names of the resource parameters; non-empty exactly for a
            resource-controlled approximation.
        note: a further caveat to be printed in a report.

    """

    kind: ApproximationKind
    leading_error_order: str = ""
    resource_parameters: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind is ApproximationKind.RESOURCE_CONTROLLED and not self.resource_parameters:
            msg = "a resource-controlled approximation must name at least one resource parameter"
            raise ValueError(msg)
        if self.kind is ApproximationKind.REGIME_LIMITED and self.resource_parameters:
            msg = (
                "a regime-limited approximation has no resource to spend; "
                f"got resource_parameters={self.resource_parameters}"
            )
            raise ValueError(msg)

    def __str__(self) -> str:
        resources = (
            f" [resources: {', '.join(self.resource_parameters)}]"
            if self.resource_parameters
            else ""
        )
        order = f"; leading error {self.leading_error_order}" if self.leading_error_order else ""
        return f"{self.kind}{resources}{order}"


@dataclass(frozen=True)
class Transformation(ABC):
    """A typed edge from one artifact to another.

    Like an artifact, a transformation is a declaration before it is a computation.

    Attributes:
        name: a short identifier, for instance ``"spin-1/2 quantum-link truncation"``.
        source_pattern: the structural type the transformation requires of its source.
        target_pattern: the structural type the transformation declares for its target; used
            to type-check a composition.
        source_kind: the artifact kind of the source.
        target_kind: the artifact kind of the target.
        exactness: whether the operator form of the target is unitarily equivalent to that
            of the source.
        approximation: the approximation declaration, required exactly when ``exactness``
            is ``APPROXIMATE``.
        relation: the parameter relation.
        validity: the validity conditions, required for an approximate transformation.
        source_parameters: the source-side parameter names the relation involves.
        target_parameters: the target-side parameter names the relation involves.
        source_level: the abstraction level of the source; ``None`` admits any level.
        target_level: the abstraction level of the target.
        description: a longer description, printed in reports and pipeline tables.
        dropped_constant: a description of the additive constant the transformation
            discards, or empty; its value is computed by
            [`dropped_constant_at`][qsimod.transform.Transformation.dropped_constant_at].

    """

    name: str
    source_pattern: StructurePattern
    target_pattern: StructurePattern
    exactness: Exactness
    source_kind: ArtifactKind = ArtifactKind.HAMILTONIAN
    target_kind: ArtifactKind = ArtifactKind.HAMILTONIAN
    approximation: Approximation | None = None
    relation: ParameterRelation = EMPTY_RELATION
    validity: Conjunction = field(default_factory=Conjunction)
    source_parameters: tuple[str, ...] = ()
    target_parameters: tuple[str, ...] = ()
    source_level: AbstractionLevel | None = None
    target_level: AbstractionLevel | None = None
    description: str = ""
    dropped_constant: str = ""

    def __post_init__(self) -> None:
        """Check that the exactness and the approximation declaration agree.

        An approximation declaration is required exactly when the transformation is
        ``APPROXIMATE``, and a ``regime-limited`` transformation must declare at least one
        validity condition.

        Raises:
            ValueError: if the declarations do not agree.

        """
        if self.approximation is None:
            if self.exactness is Exactness.APPROXIMATE:
                msg = (
                    f"{self.name}: an APPROXIMATE transformation must declare its "
                    "approximation kind"
                )
                raise ValueError(msg)
            return
        if self.exactness is Exactness.EXACT:
            msg = f"{self.name}: an EXACT transformation must not declare an approximation kind"
            raise ValueError(msg)
        if self.approximation.kind is ApproximationKind.REGIME_LIMITED and not self.validity:
            msg = (
                f"{self.name}: a regime-limited approximation must declare at least one "
                "validity condition"
            )
            raise ValueError(msg)

    # -- type checking -----------------------------------------------------------

    def check(self, artifact: Artifact) -> None:
        """Type-check ``artifact`` against the source kind and pattern of the transformation.

        Raises:
            ArtifactKindError: if the artifact is of the wrong kind.
            StructureTypeError: if the structure of the artifact does not match; every
                mismatch is named.

        """
        if artifact.kind is not self.source_kind:
            raise ArtifactKindError(self.name, self.source_kind, artifact.kind, artifact.name)
        self.source_pattern.check(artifact.structure, self.name)

    def accepts(self, artifact: Artifact) -> bool:
        """Whether [`check`][qsimod.transform.Transformation.check] passes for ``artifact``."""
        return artifact.kind is self.source_kind and self.source_pattern.matches(artifact.structure)

    def target_structure(self, source: StructureType) -> StructureType:
        """The structural type produced from ``source``; unchanged by default."""
        return source

    def target_sites(self, source_sites: int) -> int:
        """The chain length of the target given that of the source; the same length by default.

        Args:
            source_sites: the chain length of the source artifact.

        Returns:
            The chain length of the target artifact.

        Raises:
            ValueError: if no target length corresponds to ``source_sites``.

        """
        return source_sites

    # -- application -------------------------------------------------------------

    def apply(self, artifact: Artifact) -> Artifact:
        """Type-check ``artifact`` and then transform it.

        Raises:
            ArtifactKindError: via [`check`][qsimod.transform.Transformation.check].
            StructureTypeError: via [`check`][qsimod.transform.Transformation.check].

        """
        self.check(artifact)
        return self._transform(artifact)

    @abstractmethod
    def _transform(self, artifact: Artifact) -> Artifact:
        """Produce the target artifact; called after `check` has passed."""

    def dropped_constant_at(self, source: Artifact) -> DroppedConstant | None:  # noqa: ARG002
        """The additive constant discarded from ``source``; ``None`` by default."""
        return None

    # -- exactness, made checkable -----------------------------------------------

    def image(self, source: HamiltonianModel) -> OperatorSum | None:  # noqa: ARG002
        """The source Hamiltonian rewritten in the target operators, in the source parameters.

        For an ``EXACT`` transformation that implements this method, the claim is checked on
        every application: the target model of the library must equal this image up to an
        additive constant
        ([`exactness_defect`][qsimod.transform.Transformation.exactness_defect]).  The default
        returns ``None``: no symbolic image is available and nothing is checked.
        """
        return None

    def forward_definitions(self) -> dict[str, Scalar]:
        """Each target parameter as an expression in the source parameters, where solved."""
        sources = set(self.source_parameters)
        forward: dict[str, Scalar] = {}
        for definition in self.relation.definitions:
            if definition.parameter in self.target_parameters and definition.inputs() <= sources:
                forward.setdefault(definition.parameter, definition.expression)
        return forward

    def exactness_defect(
        self, source: HamiltonianModel, target: HamiltonianModel
    ) -> OperatorSum | None:
        """``target - image(source)`` in normal form, or ``None`` if no image is available.

        The Hamiltonian of the target is first expressed in the source parameters through the
        forward solved forms of the relation, so that both sides read the same symbols.  An
        empty result means that the transformation is exact as declared; a result with an
        identity term only means exact up to that additive constant; any other result is a
        defect.
        """
        image = self.image(source)
        if image is None:
            return None
        expected = target.hamiltonian.substitute(self.forward_definitions())
        algebra_at = {site: target.structure.algebra_at(site) for site in target.structure.sites}
        return difference(expected, image, algebra_at)

    # -- the three axes, as reports ----------------------------------------------

    @property
    def is_exact(self) -> bool:
        """Whether the operator forms of source and target are unitarily equivalent."""
        return self.exactness is Exactness.EXACT

    @property
    def approximation_kind(self) -> ApproximationKind | None:
        """The kind of approximation, or ``None`` for an exact transformation."""
        return None if self.approximation is None else self.approximation.kind

    def relation_kind(self, known: frozenset[str]) -> RelationKind:
        """The case of the relation when solving for every parameter ``known`` does not fix."""
        return self.classify_relation(known).kind

    def classify_relation(self, known: frozenset[str]) -> RelationClassification:
        """The full static classification of the relation in the requested direction."""
        return self.relation.classify(known=known)

    @property
    def forward_relation_kind(self) -> RelationKind:
        """The case of the relation in the forward direction of the derivation.

        The source parameters are given.
        """
        return self.relation_kind(frozenset(self.source_parameters))

    @property
    def inverse_relation_kind(self) -> RelationKind:
        """The case of the relation in the inverse direction, the solve for the knob settings.

        The target parameters are given.
        """
        return self.relation_kind(frozenset(self.target_parameters))

    def validity_report(self, env: dict[str, float]) -> ValidityReport:
        """Evaluate the validity conditions at a parameter point."""
        return self.validity.report(env)

    def carry_parameters(self, source: Artifact, target: Artifact) -> dict[str, float]:
        """Propagate the bound values of the source through the solved forms of the relation.

        Only the parameters of the target are set, and only where a solved form reads values
        the source already binds.

        Args:
            source: the artifact being transformed.
            target: the artifact being produced.

        Returns:
            The values to bind on the target.

        """
        known = source.binding.as_dict()
        if not known:
            return {}
        values: dict[str, float] = {}
        for definition in self.relation.definitions:
            if definition.parameter not in target.parameters.names:
                continue
            if definition.parameter in values:
                continue
            if not definition.inputs() <= set(known):
                continue
            values[definition.parameter] = definition.expression.evaluate_real(known)
        return values

    def parameter_set(self) -> ParameterSet:
        """The source and target parameters, as a set of energy-dimensioned parameters."""
        names = tuple(self.source_parameters) + tuple(self.target_parameters)
        seen: list[str] = []
        for name in names:
            if name not in seen:
                seen.append(name)
        return ParameterSet(tuple(Parameter(name) for name in seen))

    @property
    def levels(self) -> tuple[AbstractionLevel, AbstractionLevel] | None:
        """The levels the transformation spans, or ``None`` if they are not declared."""
        if self.source_level is None or self.target_level is None:
            return None
        return (self.source_level, self.target_level)

    def __str__(self) -> str:
        approximation = f" ({self.approximation})" if self.approximation else ""
        span = self.levels
        levels = "" if span is None else f" [{span[0].label} -> {span[1].label}]"
        return f"{self.name}{levels}: {self.exactness}{approximation}"
