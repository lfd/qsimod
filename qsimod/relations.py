"""Parameter relations between the source and target parameters of a transformation.

A relation is a system of [`Equation`][qsimod.relations.Equation] objects over
[`Scalar`][qsimod.scalar.Scalar] expressions together with optional
[`Definition`][qsimod.relations.Definition] solved forms, each tagged with the equation it
solves.  The direction is chosen at classification time
([`ParameterRelation.classify`][qsimod.relations.ParameterRelation.classify]).  The composition
of relations is the union of their equation sets, with intermediate parameters left free.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from enum import Enum

from qsimod.scalar import Scalar, ScalarLike, Symbol, as_scalar

__all__ = [
    "EMPTY_RELATION",
    "Definition",
    "Equation",
    "ParameterRelation",
    "RelationClassification",
    "RelationKind",
    "equation",
    "identity_relation",
    "residuals_of",
]


class RelationKind(Enum):
    """The case of a relation in one direction, out of four.

    The classification is directional.  A relation that is ``CLOSED_FORM`` in both directions
    is invertible, which
    [`ParameterRelation.is_invertible`][qsimod.relations.ParameterRelation.is_invertible]
    reports.
    """

    CLOSED_FORM = "CLOSED_FORM"
    """An explicit expression per unknown, evaluable directly."""

    UNDER_DETERMINED = "UNDER_DETERMINED"
    """Fewer equations than unknowns: a solution manifold, which requires a selection policy or
    an objective."""

    OVER_DETERMINED = "OVER_DETERMINED"
    """More equations than unknowns: generically no exact solution exists, only a residual."""

    IMPLICIT = "IMPLICIT"
    """Determined, but without a closed form in this direction."""

    @property
    def needs_solver(self) -> bool:
        """Whether obtaining a point requires a numerical or symbolic solve."""
        return self is not RelationKind.CLOSED_FORM

    @property
    def can_be_exact(self) -> bool:
        """Whether an exact solution can exist; false only for ``OVER_DETERMINED``."""
        return self is not RelationKind.OVER_DETERMINED

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Equation:
    """One equation ``left = right`` over named parameters.

    Attributes:
        name: a stable identifier, referenced by [`Definition`][qsimod.relations.Definition]
            and by per-equation residuals.
        left: the left-hand side.
        right: the right-hand side.
        description: a one-line description.

    """

    name: str
    left: Scalar
    right: Scalar
    description: str = ""

    @property
    def residual(self) -> Scalar:
        """The expression ``left - right``, which a solve drives to zero."""
        return self.left - self.right

    def parameters(self) -> frozenset[str]:
        """Every parameter the equation mentions."""
        return self.left.symbols() | self.right.symbols()

    def substituted(self, env: Mapping[str, ScalarLike]) -> Equation:
        """A copy with the substitutions applied to both sides."""
        return Equation(
            name=self.name,
            left=self.left.substitute(env),
            right=self.right.substitute(env),
            description=self.description,
        )

    def __str__(self) -> str:
        return f"{self.name}: {self.left} = {self.right}"


def equation(name: str, left: ScalarLike, right: ScalarLike, description: str = "") -> Equation:
    """Convenience constructor for [`Equation`][qsimod.relations.Equation]."""
    return Equation(name, as_scalar(left), as_scalar(right), description)


@dataclass(frozen=True)
class Definition:
    """An explicit solved form of one equation for one parameter.

    Attributes:
        parameter: the parameter the expression defines.
        expression: the expression, in terms of other parameters.
        equation: the name of the [`Equation`][qsimod.relations.Equation] the definition
            solves.
        note: a caveat, typically the branch of a multi-valued inverse that was taken.

    """

    parameter: str
    expression: Scalar
    equation: str
    note: str = ""

    def inputs(self) -> frozenset[str]:
        """The parameters the expression reads."""
        return self.expression.symbols()

    def substituted(self, env: Mapping[str, ScalarLike]) -> Definition:
        """A copy with the substitutions applied to the expression."""
        return Definition(
            parameter=self.parameter,
            expression=self.expression.substitute(env),
            equation=self.equation,
            note=self.note,
        )

    def __str__(self) -> str:
        note = f"  ({self.note})" if self.note else ""
        return f"{self.parameter} := {self.expression}{note}"


@dataclass(frozen=True)
class RelationClassification:
    """The static classification of a relation in one requested direction.

    Attributes:
        kind: the case that applies, out of four.
        unknowns: the parameters solved for, sorted.
        known: the parameters treated as given, sorted.
        equations: the names of the equations involved, in order.
        elimination: the solved forms in evaluation order; present exactly when ``kind``
            is ``CLOSED_FORM``.
        note: a one-line explanation of the classification, for reports.

    """

    kind: RelationKind
    unknowns: tuple[str, ...]
    known: tuple[str, ...]
    equations: tuple[str, ...]
    elimination: tuple[Definition, ...] | None = None
    note: str = ""

    @property
    def degrees_of_freedom(self) -> int:
        """The number of unknowns minus the number of equations.

        A positive value is the rank of the solution manifold.
        """
        return len(self.unknowns) - len(self.equations)

    def __str__(self) -> str:
        return (
            f"{self.kind}: {len(self.equations)} equation(s), {len(self.unknowns)} unknown(s) "
            f"({', '.join(self.unknowns)}){'; ' + self.note if self.note else ''}"
        )


@dataclass(frozen=True)
class ParameterRelation:
    """A system of equations relating the parameters of a transformation.

    Attributes:
        name: a name used in reports.
        equations: the equations, with distinct names.
        definitions: the solved forms; each refers to an equation of the relation.  A
            relation without definitions is implicit in every direction.

    """

    name: str = ""
    equations: tuple[Equation, ...] = ()
    definitions: tuple[Definition, ...] = ()

    def __post_init__(self) -> None:
        names = [eq.name for eq in self.equations]
        if len(names) != len(set(names)):
            msg = f"duplicate equation names in relation {self.name!r}: {names}"
            raise ValueError(msg)
        known = set(names)
        for definition in self.definitions:
            if definition.equation not in known:
                msg = (
                    f"definition {definition.parameter} := ... claims to solve equation "
                    f"{definition.equation!r}, which relation {self.name!r} does not have"
                )
                raise ValueError(msg)

    # -- inspection --------------------------------------------------------------

    def parameters(self) -> frozenset[str]:
        """Every parameter mentioned by any equation."""
        if not self.equations:
            return frozenset()
        return frozenset().union(*(eq.parameters() for eq in self.equations))

    def equation_named(self, name: str) -> Equation:
        """The equation of the given name.

        Raises:
            KeyError: if the relation has no equation of that name.

        """
        for candidate in self.equations:
            if candidate.name == name:
                return candidate
        msg = f"relation {self.name!r} has no equation named {name!r}"
        raise KeyError(msg)

    def definitions_of(self, parameter: str) -> tuple[Definition, ...]:
        """Every solved form that defines ``parameter``."""
        return tuple(d for d in self.definitions if d.parameter == parameter)

    # -- composition -------------------------------------------------------------

    def compose(self, other: ParameterRelation, name: str = "") -> ParameterRelation:
        """Conjoin two relations by uniting their equations and definitions.

        Equation names must be distinct; see
        [`prefixed`][qsimod.relations.ParameterRelation.prefixed].

        Args:
            other: the relation to conjoin with.
            name: a name for the composite; by default the two names joined by ``&``.

        Returns:
            The composite relation.

        """
        return ParameterRelation(
            name=name or f"{self.name} & {other.name}".strip(" &"),
            equations=self.equations + other.equations,
            definitions=self.definitions + other.definitions,
        )

    def prefixed(self, prefix: str) -> ParameterRelation:
        """A copy with every equation name prefixed; parameter names are unchanged."""
        return ParameterRelation(
            name=self.name,
            equations=tuple(
                Equation(f"{prefix}{eq.name}", eq.left, eq.right, eq.description)
                for eq in self.equations
            ),
            definitions=tuple(
                Definition(d.parameter, d.expression, f"{prefix}{d.equation}", d.note)
                for d in self.definitions
            ),
        )

    def substituted(self, env: Mapping[str, ScalarLike]) -> ParameterRelation:
        """A copy with the substitutions applied throughout."""
        return ParameterRelation(
            name=self.name,
            equations=tuple(eq.substituted(env) for eq in self.equations),
            definitions=tuple(d.substituted(env) for d in self.definitions),
        )

    def with_equations(
        self,
        extra: Sequence[Equation],
        extra_definitions: Sequence[Definition] = (),
        name: str = "",
    ) -> ParameterRelation:
        """A copy with additional equations, for instance parameter pins or a selection policy."""
        return ParameterRelation(
            name=name or self.name,
            equations=self.equations + tuple(extra),
            definitions=self.definitions + tuple(extra_definitions),
        )

    # -- classification ----------------------------------------------------------

    def classify(
        self,
        known: Set[str],
        unknowns: Set[str] | None = None,
    ) -> RelationClassification:
        """Classify the relation in the direction of solving for the unknowns given ``known``.

        Args:
            known: the parameters treated as given.
            unknowns: the parameters to solve for; by default every parameter the equations
                mention that is not in ``known``.

        Returns:
            The classification, including an elimination order if a closed form exists.

        """
        mentioned = self.parameters()
        free = set(mentioned) - set(known) if unknowns is None else set(unknowns)
        relevant = tuple(eq for eq in self.equations if eq.parameters() & free)
        unknown_names = tuple(sorted(free))
        equation_names = tuple(eq.name for eq in relevant)
        known_names = tuple(sorted(known))

        def classified(
            kind: RelationKind,
            note: str,
            elimination: tuple[Definition, ...] | None = None,
        ) -> RelationClassification:
            return RelationClassification(
                kind=kind,
                unknowns=unknown_names,
                known=known_names,
                equations=equation_names,
                elimination=elimination,
                note=note,
            )

        if len(relevant) < len(free):
            return classified(
                RelationKind.UNDER_DETERMINED,
                f"{len(free) - len(relevant)} residual degree(s) of freedom; needs a "
                "selection policy or an objective",
            )
        if len(relevant) > len(free):
            return classified(
                RelationKind.OVER_DETERMINED,
                f"{len(relevant) - len(free)} more equation(s) than unknown(s); "
                "generically no exact solution, a best fit carries a residual",
            )
        chain = self._eliminate(known=set(known), unknowns=free, equations=relevant)
        if chain is None:
            return classified(
                RelationKind.IMPLICIT,
                "determined, but no solved form chain reaches every unknown",
            )
        return classified(
            RelationKind.CLOSED_FORM,
            "every unknown has a solved form evaluable from known values",
            elimination=chain,
        )

    def is_invertible(self, side_a: Set[str], side_b: Set[str]) -> bool:
        """Whether the relation is a closed form in both directions between the two sides."""
        forward = self.classify(known=side_a, unknowns=set(side_b))
        backward = self.classify(known=side_b, unknowns=set(side_a))
        return (
            forward.kind is RelationKind.CLOSED_FORM and backward.kind is RelationKind.CLOSED_FORM
        )

    def _eliminate(
        self,
        known: set[str],
        unknowns: set[str],
        equations: Sequence[Equation],
    ) -> tuple[Definition, ...] | None:
        """Order the solved forms greedily so that each reads only values already known."""
        available = set(known)
        pending = set(unknowns)
        unused = {eq.name for eq in equations}
        order: list[Definition] = []
        progress = True
        while pending and progress:
            progress = False
            for definition in self.definitions:
                if definition.parameter not in pending:
                    continue
                if definition.equation not in unused:
                    continue
                if not definition.inputs() <= available:
                    continue
                order.append(definition)
                available.add(definition.parameter)
                pending.discard(definition.parameter)
                unused.discard(definition.equation)
                progress = True
                break
        if pending:
            return None
        return tuple(order)

    # -- direct evaluation -------------------------------------------------------

    def evaluate_closed_form(
        self,
        elimination: Sequence[Definition],
        known: Mapping[str, float],
    ) -> dict[str, float]:
        """Evaluate an elimination chain and return the full parameter assignment.

        Args:
            elimination: the chain from [`classify`][qsimod.relations.ParameterRelation.classify].
            known: the values of the known parameters.

        Returns:
            ``known`` extended by a value for every defined parameter.

        """
        assignment = dict(known)
        for definition in elimination:
            assignment[definition.parameter] = definition.expression.evaluate_real(assignment)
        return assignment

    def residuals(self, assignment: Mapping[str, float]) -> dict[str, float]:
        """The residual of every equation whose parameters are all assigned."""
        return residuals_of(self.equations, assignment)

    def __str__(self) -> str:
        head = self.name or "relation"
        if not self.equations:
            return f"{head}: no equations (parameters carry over unchanged)"
        body = "\n".join(f"  {eq}" for eq in self.equations)
        return f"{head}:\n{body}"

    def __len__(self) -> int:
        return len(self.equations)


def residuals_of(
    equations: Iterable[Equation],
    assignment: Mapping[str, float],
) -> dict[str, float]:
    """The residual of each equation whose parameters ``assignment`` covers, by name.

    Equations that mention an unassigned parameter are omitted.
    """
    return {
        equation.name: equation.residual.evaluate_real(assignment)
        for equation in equations
        if equation.parameters() <= set(assignment)
    }


def identity_relation(
    pairs: Iterable[tuple[str, str]],
    name: str = "",
) -> ParameterRelation:
    """A relation that carries parameters over unchanged, with one equation per pair.

    Each pair ``(source, target)`` becomes an equation ``target = source`` with solved forms
    in both directions; the relation is therefore invertible.

    Args:
        pairs: the ``(source, target)`` pairs of parameter names.
        name: a name used in reports.

    Returns:
        The relation.

    """
    equations: list[Equation] = []
    definitions: list[Definition] = []
    for source, target in pairs:
        eq_name = f"{target}={source}"
        equations.append(equation(eq_name, Symbol(target), Symbol(source), "carried over"))
        definitions.append(Definition(target, Symbol(source), eq_name))
        definitions.append(Definition(source, Symbol(target), eq_name))
    return ParameterRelation(name=name, equations=tuple(equations), definitions=tuple(definitions))


EMPTY_RELATION = ParameterRelation(name="empty")
"""A relation without equations: the parameters of the two sides are unrelated."""
