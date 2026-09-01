"""Objectives for an under-determined solve.

An under-determined system of parameter relations admits multiple knob settings that realise
the same request; an objective selects among them.  The default,
[`MaximiseWeakestMargin`][qsimod.solving.objectives.MaximiseWeakestMargin], places the operating
point as deep inside the declared regime as the constraints allow.  It is posed by epigraph
transformation: an auxiliary variable ``s`` (`SLACK_VARIABLE`) is introduced, the margin of
every regime condition is constrained to be at least ``s``, and ``s`` is maximised;
[`Objective.slack_variable`][qsimod.solving.objectives.Objective.slack_variable] declares the
auxiliary variable to a backend.
[`MinimiseExpression`][qsimod.solving.objectives.MinimiseExpression] and
[`MaximiseExpression`][qsimod.solving.objectives.MaximiseExpression] accept any scalar
expression.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from qsimod.scalar import DomainError, Scalar, ScalarLike, Symbol, absolute, as_scalar
from qsimod.validity import MuchLessThan, Severity, ValidityCondition

__all__ = [
    "SLACK_VARIABLE",
    "FeasibilityOnly",
    "MaximiseExpression",
    "MaximiseWeakestMargin",
    "MinimiseExpression",
    "Objective",
    "default_objective",
    "maximise",
    "minimise",
]

#: The name of the auxiliary epigraph variable introduced by a max-min objective.
SLACK_VARIABLE = "__margin_slack"


class Objective(ABC):
    """The quantity a solve minimises, with the auxiliary variables and constraints it requires.

    A backend minimises
    [`cost_expression`][qsimod.solving.objectives.Objective.cost_expression] over the unknowns
    and any [`slack_variable`][qsimod.solving.objectives.Objective.slack_variable]; a report
    states [`value`][qsimod.solving.objectives.Objective.value], the quantity the objective is
    named after, at the solved point.
    """

    name: str
    description: str

    @abstractmethod
    def cost_expression(self) -> Scalar | None:
        """The expression a backend minimises, over the unknowns and the slack variable.

        ``None`` declares no preference: any feasible point is accepted.
        """

    @abstractmethod
    def value(self, assignment: Mapping[str, float]) -> float:
        """The quantity the objective is named after, at a point.

        For a maximisation this is the maximised quantity itself, which is the quantity a
        report states; [`cost`][qsimod.solving.objectives.Objective.cost] is the quantity the
        backend minimised.
        """

    def cost(self, assignment: Mapping[str, float]) -> float:
        """The quantity the backend minimises, at a point; a lower value is preferred."""
        return self.value(assignment)

    @abstractmethod
    def parameters(self) -> frozenset[str]:
        """The names of the parameters the objective depends on."""

    @property
    def slack_variable(self) -> str | None:
        """The auxiliary variable the objective requires, if any."""
        return None

    def epigraph_constraints(self) -> tuple[tuple[str, Scalar], ...]:
        """The named expressions that must be ``>= 0`` for the epigraph form to be valid."""
        return ()

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class FeasibilityOnly(Objective):
    """No preference: any admissible point that satisfies the relation is accepted."""

    name: str = field(default="feasibility only", init=False)
    description: str = field(
        default="no objective; the first admissible point that satisfies the relation",
        init=False,
    )

    def cost_expression(self) -> None:
        """``None``, declaring no preference."""

    # `assignment` is unused: this objective has no preference.
    def value(self, assignment: Mapping[str, float]) -> float:  # noqa: ARG002
        """The constant zero."""
        return 0.0

    def parameters(self) -> frozenset[str]:
        """The empty set."""
        return frozenset()


@dataclass(frozen=True)
class MinimiseExpression(Objective):
    """Minimise a scalar expression over the parameters of the pipeline."""

    expression: Scalar
    name: str = "minimise expression"
    description: str = ""

    def cost_expression(self) -> Scalar:
        """The expression itself."""
        return self.expression

    def value(self, assignment: Mapping[str, float]) -> float:
        """The expression's value."""
        return self.expression.evaluate_real(assignment)

    def parameters(self) -> frozenset[str]:
        """The expression's symbols."""
        return self.expression.symbols()


@dataclass(frozen=True)
class MaximiseExpression(Objective):
    """Maximise a scalar expression over the parameters of the pipeline."""

    expression: Scalar
    name: str = "maximise expression"
    description: str = ""

    def cost_expression(self) -> Scalar:
        """The negated expression."""
        return -self.expression

    def value(self, assignment: Mapping[str, float]) -> float:
        """The expression's value."""
        return self.expression.evaluate_real(assignment)

    def cost(self, assignment: Mapping[str, float]) -> float:
        """The negated value of the expression."""
        return -self.value(assignment)

    def parameters(self) -> frozenset[str]:
        """The expression's symbols."""
        return self.expression.symbols()


@dataclass(frozen=True)
class MaximiseWeakestMargin(Objective):
    """Maximise the margin of the weakest regime condition.

    Attributes:
        conditions: the validity conditions whose margins are traded off; only the regime
            conditions enter, as the domain conditions are hard constraints.
        normalise: whether each margin surrogate is divided by the scale of its own
            requirement, which makes conditions in different units comparable (default
            ``True``).

    """

    conditions: tuple[ValidityCondition, ...]
    normalise: bool = True
    name: str = field(default="maximise the weakest validity margin", init=False)
    description: str = field(
        default=(
            "place the operating point as deep inside the validity window as the "
            "constraints allow; posed as an epigraph problem over the regime conditions"
        ),
        init=False,
    )

    @property
    def regime_conditions(self) -> tuple[ValidityCondition, ...]:
        """The regime conditions among ``conditions``, which carry the maximised margins."""
        return tuple(c for c in self.conditions if c.severity is Severity.REGIME)

    def cost_expression(self) -> Scalar:
        """The negated epigraph variable, which the constraints hold below every margin."""
        return -Symbol(SLACK_VARIABLE)

    def value(self, assignment: Mapping[str, float]) -> float:
        """The least normalised surrogate margin at a point: the quantity that is maximised.

        Each regime condition contributes ``margin_expression / scale``, the expression the
        epigraph constraints bound below by the slack variable; the value is therefore the
        optimum in the units of the backend: dimensionless, linear, and positive inside the
        declared regime.  The logarithmic margins in decades are stated in the validity report.
        """
        margins: list[float] = []
        for condition in self.regime_conditions:
            surrogate = condition.margin_expression()
            if surrogate is None:
                continue
            scale = _surrogate_scale(condition) if self.normalise else as_scalar(1.0)
            try:
                margins.append((surrogate / scale).evaluate_real(assignment))
            except (DomainError, KeyError):
                margins.append(-math.inf)
        return min(margins) if margins else 0.0

    def cost(self, assignment: Mapping[str, float]) -> float:
        """The negated least normalised surrogate margin."""
        return -self.value(assignment)

    def parameters(self) -> frozenset[str]:
        """Every parameter that any regime condition depends on."""
        conditions = self.regime_conditions
        if not conditions:
            return frozenset()
        return frozenset().union(*(c.parameters() for c in conditions))

    @property
    def slack_variable(self) -> str:
        """The name of the epigraph variable."""
        return SLACK_VARIABLE

    def epigraph_constraints(self) -> tuple[tuple[str, Scalar], ...]:
        """The constraints ``surrogate_margin(theta) - s >= 0``, one per regime condition."""
        slack = Symbol(SLACK_VARIABLE)
        constraints: list[tuple[str, Scalar]] = []
        for condition in self.regime_conditions:
            surrogate = condition.margin_expression()
            if surrogate is None:
                continue
            scale = _surrogate_scale(condition) if self.normalise else as_scalar(1.0)
            constraints.append((f"margin >= s: {condition.name}", surrogate / scale - slack))
        return tuple(constraints)


def _surrogate_scale(condition: ValidityCondition) -> Scalar:
    """The scale that makes the margin surrogate of a condition dimensionless.

    For ``a << b`` the surrogate ``threshold*|b| - |a|`` is divided by ``|b|``; every other kind
    of condition is scaled by unity.  The magnitude of the denominator is taken, so that a
    negative denominator cannot reverse the epigraph constraint.
    """
    if isinstance(condition, MuchLessThan):
        return absolute(condition.denominator)
    return as_scalar(1.0)


def default_objective(conditions: Sequence[ValidityCondition]) -> Objective:
    """The default objective for an under-determined solve.

    [`MaximiseWeakestMargin`][qsimod.solving.objectives.MaximiseWeakestMargin] when regime
    conditions are declared, [`FeasibilityOnly`][qsimod.solving.objectives.FeasibilityOnly]
    otherwise.
    """
    regime = [c for c in conditions if c.severity is Severity.REGIME]
    if not regime:
        return FeasibilityOnly()
    return MaximiseWeakestMargin(tuple(conditions))


def minimise(expression: ScalarLike, name: str = "", description: str = "") -> MinimiseExpression:
    """Construct a minimisation objective.

    Returns a [`MinimiseExpression`][qsimod.solving.objectives.MinimiseExpression] over
    ``expression``.
    """
    scalar = as_scalar(expression)
    return MinimiseExpression(scalar, name or f"minimise {scalar}", description)


def maximise(expression: ScalarLike, name: str = "", description: str = "") -> MaximiseExpression:
    """Construct a maximisation objective.

    Returns a [`MaximiseExpression`][qsimod.solving.objectives.MaximiseExpression] over
    ``expression``.
    """
    scalar = as_scalar(expression)
    return MaximiseExpression(scalar, name or f"maximise {scalar}", description)
