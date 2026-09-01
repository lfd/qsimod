"""Validity conditions: the declared regime within which an approximate transformation holds.

A regime condition ``a << b`` is evaluated as ``|a / b| <= threshold`` with a default threshold
of 0.1; a domain condition ``x != 0`` is evaluated as ``|x| / scale > tolerance`` with a default
tolerance of 1e-3.  Margins are reported in decades of slack, the ``log10`` of the ratio to the
boundary: zero at the boundary, positive inside the regime.  The outcome of a condition is
valid, out of regime, domain error, or undetermined if a parameter was unbound.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from qsimod.scalar import DomainError, Scalar, ScalarLike, absolute, as_scalar

__all__ = [
    "DEFAULT_MUCH_LESS_THAN_THRESHOLD",
    "DEFAULT_NONZERO_TOLERANCE",
    "ConditionOutcome",
    "ConditionStatus",
    "Conjunction",
    "MuchLessThan",
    "NonZero",
    "Severity",
    "ValidityCondition",
    "ValidityReport",
    "conjunction",
    "much_less_than",
    "non_zero",
]

#: The default threshold of ``<<``: a ratio of at most one tenth.
DEFAULT_MUCH_LESS_THAN_THRESHOLD = 0.1

#: The default tolerance of ``!= 0``: a separation of at least this fraction of the stated scale.
DEFAULT_NONZERO_TOLERANCE = 1e-3


class Severity(Enum):
    """Whether the failure of a condition is a mathematical or a physical failure."""

    DOMAIN = "domain"
    """The parameter map is undefined (a pole).  The condition cannot be overridden."""

    REGIME = "regime"
    """The parameter map is defined, but the assumptions of the derivation are violated.  The
    condition can be overridden on explicit request."""

    def __str__(self) -> str:
        return self.value


class ConditionStatus(Enum):
    """The outcome of the evaluation of one condition."""

    VALID = "valid"
    OUT_OF_REGIME = "out of regime"
    DOMAIN_ERROR = "domain error"
    UNDETERMINED = "undetermined"
    """A parameter read by the condition was unbound."""

    @property
    def is_failure(self) -> bool:
        """Whether the outcome means that the transformation is not valid at the point."""
        return self in {ConditionStatus.OUT_OF_REGIME, ConditionStatus.DOMAIN_ERROR}

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ConditionOutcome:
    """The evaluation of one validity condition at one parameter point.

    Attributes:
        name: the name of the condition.
        quantity: the compared quantity, rendered for a report, for instance ``"J/delta"``.
        value: the value of that quantity.
        requirement: the requirement, rendered, for instance ``"<< 1 (ratio <= 0.1)"``.
        margin: the decades of slack; positive inside the regime, negative outside.
        status: the outcome.
        severity: whether a failure of the condition is a domain or a regime failure.
        rationale: the reason for the condition, for the report.

    """

    name: str
    quantity: str
    value: float
    requirement: str
    margin: float
    status: ConditionStatus
    severity: Severity
    rationale: str = ""

    @property
    def passed(self) -> bool:
        """Whether the condition holds."""
        return self.status is ConditionStatus.VALID

    def __str__(self) -> str:
        margin = "   n/a" if math.isnan(self.margin) else f"{self.margin:+6.2f}"
        return (
            f"{self.name:<24} {self.quantity:<14} = {self.value:< 12.6g} "
            f"requires {self.requirement:<26} margin {margin} dec  [{self.status}]"
        )


class ValidityCondition(ABC):
    """A predicate over parameter values, with a margin and a severity.

    Attributes:
        name: the name of the condition.
        severity: whether a failure is a domain or a regime failure.
        rationale: the reason for the condition, for reports.

    """

    name: str
    severity: Severity
    rationale: str

    @abstractmethod
    def evaluate(self, env: Mapping[str, float]) -> ConditionOutcome:
        """Evaluate the condition at a parameter point.

        Args:
            env: the parameter values.

        Returns:
            The outcome; a missing parameter gives
            [`ConditionStatus.UNDETERMINED`][qsimod.validity.ConditionStatus.UNDETERMINED].

        """

    @abstractmethod
    def parameters(self) -> frozenset[str]:
        """The names of the parameters the condition reads."""

    @abstractmethod
    def substituted(self, env: Mapping[str, ScalarLike]) -> ValidityCondition:
        """A copy with the parameter substitutions applied to its expressions."""

    def margin_expression(self) -> Scalar | None:
        """A smooth expression with the sign and zero set of the margin; ``None`` if none exists."""
        return None

    # -- outcome construction, shared by every condition kind --------------------

    def _outcome(
        self,
        *,
        quantity: str,
        value: float,
        requirement: str,
        margin: float,
        status: ConditionStatus,
        rationale: str = "",
    ) -> ConditionOutcome:
        """An outcome that carries the identity of this condition."""
        return ConditionOutcome(
            name=self.name,
            quantity=quantity,
            value=value,
            requirement=requirement,
            margin=margin,
            status=status,
            severity=self.severity,
            rationale=rationale or self.rationale,
        )

    def _undetermined(
        self,
        missing: Sequence[str],
        *,
        quantity: str,
        requirement: str,
    ) -> ConditionOutcome:
        """The undetermined outcome for a point with missing parameters.

        The missing parameters are named in the rationale.
        """
        return self._outcome(
            quantity=quantity,
            value=math.nan,
            requirement=requirement,
            margin=math.nan,
            status=ConditionStatus.UNDETERMINED,
            rationale=f"unbound: {', '.join(missing)}",
        )


def _decades(ratio: float, boundary: float, *, larger_is_safer: bool) -> float:
    """The signed decades of slack between a dimensionless ``ratio`` and its ``boundary``.

    A positive value lies inside the regime; ``larger_is_safer`` states on which side of the
    boundary the regime lies.  A zero or infinite ratio gives ``+-inf``.
    """
    if ratio == 0.0:
        return -math.inf if larger_is_safer else math.inf
    if math.isinf(ratio):
        return math.inf if larger_is_safer else -math.inf
    if larger_is_safer:
        return math.log10(ratio / boundary)
    return math.log10(boundary / ratio)


@dataclass(frozen=True)
class MuchLessThan(ValidityCondition):
    """The regime condition ``|numerator / denominator| <= threshold``, that is ``a << b``.

    Attributes:
        numerator: the small quantity.
        denominator: the large quantity.
        threshold: the ratio at which the condition lies exactly on the boundary.
        name: the name of the condition.
        quantity_label: the rendering of the ratio, for instance ``"J/delta"``.
        rationale: the reason for the condition.

    """

    numerator: Scalar
    denominator: Scalar
    name: str = "much less than"
    quantity_label: str = ""
    threshold: float = DEFAULT_MUCH_LESS_THAN_THRESHOLD
    rationale: str = ""
    severity: Severity = field(default=Severity.REGIME, init=False)

    def parameters(self) -> frozenset[str]:
        """The parameters of both expressions."""
        return self.numerator.symbols() | self.denominator.symbols()

    def _label(self) -> str:
        return self.quantity_label or f"|{self.numerator}/{self.denominator}|"

    def evaluate(self, env: Mapping[str, float]) -> ConditionOutcome:
        """Evaluate the ratio and report its margin in decades."""
        requirement = f"<< 1 (ratio <= {self.threshold:g})"
        missing = sorted(self.parameters() - set(env))
        if missing:
            return self._undetermined(missing, quantity=self._label(), requirement=requirement)
        try:
            denominator = abs(self.denominator.evaluate_real(env))
            numerator = abs(self.numerator.evaluate_real(env))
        except DomainError:
            # A pole: the ratio is unbounded.
            denominator, numerator = 0.0, math.inf
        ratio = math.inf if denominator == 0.0 else numerator / denominator
        margin = _decades(ratio, self.threshold, larger_is_safer=False)
        return self._outcome(
            quantity=self._label(),
            value=ratio,
            requirement=requirement,
            margin=margin,
            status=(ConditionStatus.VALID if margin >= 0.0 else ConditionStatus.OUT_OF_REGIME),
        )

    def substituted(self, env: Mapping[str, ScalarLike]) -> MuchLessThan:
        """A copy with the substitutions applied to numerator and denominator."""
        return MuchLessThan(
            numerator=self.numerator.substitute(env),
            denominator=self.denominator.substitute(env),
            name=self.name,
            quantity_label=self.quantity_label,
            threshold=self.threshold,
            rationale=self.rationale,
        )

    def margin_expression(self) -> Scalar:
        """The expression ``threshold * |denominator| - |numerator|``, positive inside the regime.

        The expression is a linear surrogate with the same sign and zero set as the
        logarithmic margin.
        """
        return self.threshold * absolute(self.denominator) - absolute(self.numerator)


@dataclass(frozen=True)
class NonZero(ValidityCondition):
    """The domain condition ``|expression| / scale > tolerance``, that is, the absence of a pole.

    Attributes:
        expression: the quantity that must not vanish.
        scale: the scale against which the separation is measured.
        tolerance: the dimensionless separation required.
        name: the name of the condition.
        quantity_label: the rendering of the ratio.
        rationale: the reason for the condition.

    """

    expression: Scalar
    scale: Scalar
    name: str = "non-zero"
    quantity_label: str = ""
    tolerance: float = DEFAULT_NONZERO_TOLERANCE
    rationale: str = ""
    severity: Severity = field(default=Severity.DOMAIN, init=False)

    def parameters(self) -> frozenset[str]:
        """The parameters of the expression and of the scale."""
        return self.expression.symbols() | self.scale.symbols()

    def _label(self) -> str:
        return self.quantity_label or f"|{self.expression}|/{self.scale}"

    def evaluate(self, env: Mapping[str, float]) -> ConditionOutcome:
        """Evaluate the separation from the pole."""
        requirement = f"!= 0 (|.|/scale > {self.tolerance:g})"
        missing = sorted(self.parameters() - set(env))
        if missing:
            return self._undetermined(missing, quantity=self._label(), requirement=requirement)
        try:
            scale = abs(self.scale.evaluate_real(env))
            separation = abs(self.expression.evaluate_real(env))
        except DomainError:
            # A pole inside the expression itself: no separation at all.
            scale, separation = 1.0, 0.0
        ratio = math.inf if scale == 0.0 else separation / scale
        margin = _decades(ratio, self.tolerance, larger_is_safer=True)
        return self._outcome(
            quantity=self._label(),
            value=ratio,
            requirement=requirement,
            margin=margin,
            status=(ConditionStatus.VALID if margin > 0.0 else ConditionStatus.DOMAIN_ERROR),
        )

    def substituted(self, env: Mapping[str, ScalarLike]) -> NonZero:
        """A copy with the substitutions applied to expression and scale."""
        return NonZero(
            expression=self.expression.substitute(env),
            scale=self.scale.substitute(env),
            name=self.name,
            quantity_label=self.quantity_label,
            tolerance=self.tolerance,
            rationale=self.rationale,
        )

    def margin_expression(self) -> Scalar:
        """``expression**2 - (tolerance * scale)**2``, which is positive away from the pole."""
        return self.expression**2.0 - (self.tolerance * self.scale) ** 2.0


@dataclass(frozen=True)
class ValidityReport:
    """The evaluation of a conjunction of conditions at one parameter point.

    Attributes:
        outcomes: one outcome per condition, in declaration order.
        point: the parameter point evaluated, for the reproducibility of the report.

    """

    outcomes: tuple[ConditionOutcome, ...]
    point: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Whether every condition holds."""
        return all(outcome.passed for outcome in self.outcomes)

    @property
    def has_domain_error(self) -> bool:
        """Whether a condition failed as a domain error, that is, at a pole."""
        return any(o.status is ConditionStatus.DOMAIN_ERROR for o in self.outcomes)

    @property
    def domain_errors(self) -> tuple[ConditionOutcome, ...]:
        """The outcomes that failed as domain errors."""
        return tuple(o for o in self.outcomes if o.status is ConditionStatus.DOMAIN_ERROR)

    @property
    def regime_violations(self) -> tuple[ConditionOutcome, ...]:
        """The outcomes that failed as regime violations."""
        return tuple(o for o in self.outcomes if o.status is ConditionStatus.OUT_OF_REGIME)

    @property
    def failures(self) -> tuple[ConditionOutcome, ...]:
        """Every failing outcome, domain errors first."""
        return self.domain_errors + self.regime_violations

    @property
    def undetermined(self) -> tuple[ConditionOutcome, ...]:
        """The outcomes that could not be decided because a parameter value was missing."""
        return tuple(o for o in self.outcomes if o.status is ConditionStatus.UNDETERMINED)

    def weakest(self, severity: Severity | None = Severity.REGIME) -> ConditionOutcome | None:
        """The outcome with the least margin, optionally restricted to one severity."""
        candidates = [
            outcome
            for outcome in self.outcomes
            if (severity is None or outcome.severity is severity) and not math.isnan(outcome.margin)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda outcome: outcome.margin)

    @property
    def weakest_margin(self) -> float:
        """The least regime margin, in decades; ``nan`` if no condition was decidable."""
        weakest = self.weakest()
        return math.nan if weakest is None else weakest.margin

    def margins(self) -> dict[str, float]:
        """The margin of each condition, by name."""
        return {outcome.name: outcome.margin for outcome in self.outcomes}

    def __str__(self) -> str:
        if not self.outcomes:
            return "validity: no conditions declared (transformation is EXACT)"
        head = "VALID" if self.is_valid else "INVALID"
        lines = [f"validity: {head}"]
        lines += [f"  {outcome}" for outcome in self.outcomes]
        weakest = self.weakest()
        if weakest is not None:
            lines.append(
                f"  weakest regime margin: {weakest.name} at {weakest.margin:+.2f} decades"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class Conjunction:
    """A conjunction of validity conditions.

    Attributes:
        conditions: the conditions, in declaration order.

    """

    conditions: tuple[ValidityCondition, ...] = ()

    def __len__(self) -> int:
        return len(self.conditions)

    def __iter__(self) -> Iterator[ValidityCondition]:
        return iter(self.conditions)

    def __bool__(self) -> bool:
        return bool(self.conditions)

    def parameters(self) -> frozenset[str]:
        """Every parameter read by any condition."""
        if not self.conditions:
            return frozenset()
        return frozenset().union(*(c.parameters() for c in self.conditions))

    def report(self, env: Mapping[str, float]) -> ValidityReport:
        """Evaluate every condition at ``env``."""
        return ValidityReport(
            outcomes=tuple(condition.evaluate(env) for condition in self.conditions),
            point=dict(env),
        )

    def is_valid(self, env: Mapping[str, float]) -> bool:
        """Whether every condition holds at ``env``."""
        return self.report(env).is_valid

    def and_also(self, other: Conjunction) -> Conjunction:
        """The conjunction with another conjunction, preserving order and omitting duplicates."""
        seen = list(self.conditions)
        for condition in other.conditions:
            if condition not in seen:
                seen.append(condition)
        return Conjunction(tuple(seen))

    def substituted(self, env: Mapping[str, ScalarLike]) -> Conjunction:
        """A copy with the substitutions applied to every condition."""
        return Conjunction(tuple(c.substituted(env) for c in self.conditions))

    def by_severity(self, severity: Severity) -> Conjunction:
        """The sub-conjunction of conditions of one severity."""
        return Conjunction(tuple(c for c in self.conditions if c.severity is severity))

    def __str__(self) -> str:
        if not self.conditions:
            return "no validity conditions"
        return "; ".join(condition.name for condition in self.conditions)


def conjunction(conditions: Sequence[ValidityCondition]) -> Conjunction:
    """Build a [`Conjunction`][qsimod.validity.Conjunction] from a sequence of conditions."""
    return Conjunction(tuple(conditions))


def much_less_than(
    numerator: ScalarLike,
    denominator: ScalarLike,
    *,
    name: str,
    label: str = "",
    threshold: float = DEFAULT_MUCH_LESS_THAN_THRESHOLD,
    rationale: str = "",
) -> MuchLessThan:
    """Convenience constructor for [`MuchLessThan`][qsimod.validity.MuchLessThan]."""
    return MuchLessThan(
        numerator=as_scalar(numerator),
        denominator=as_scalar(denominator),
        name=name,
        quantity_label=label,
        threshold=threshold,
        rationale=rationale,
    )


def non_zero(
    expression: ScalarLike,
    scale: ScalarLike,
    *,
    name: str,
    label: str = "",
    tolerance: float = DEFAULT_NONZERO_TOLERANCE,
    rationale: str = "",
) -> NonZero:
    """Convenience constructor for [`NonZero`][qsimod.validity.NonZero]."""
    return NonZero(
        expression=as_scalar(expression),
        scale=as_scalar(scale),
        name=name,
        quantity_label=label,
        tolerance=tolerance,
        rationale=rationale,
    )
