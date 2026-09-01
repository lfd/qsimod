"""Parameters, bindings, and the admissible set of knob settings a hardware model declares.

Parameters are named symbols with a physical dimension, namespaced by the owning model, for
instance ``"L3a.J"``.  An [`AdmissibleSet`][qsimod.parameters.AdmissibleSet] holds
[`Bound`][qsimod.parameters.Bound] records and
[`InequalityConstraint`][qsimod.parameters.InequalityConstraint] expressions of the form
``expression >= 0``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum

from qsimod.scalar import Interval, Scalar, Symbol
from qsimod.units import Dimension

__all__ = [
    "EMPTY_ADMISSIBLE_SET",
    "AdmissibleSet",
    "Binding",
    "Bound",
    "ConstraintOrigin",
    "ConstraintViolation",
    "InequalityConstraint",
    "Namespace",
    "Parameter",
    "ParameterSet",
    "box",
]


@dataclass(frozen=True)
class Parameter:
    """A named physical parameter.

    Attributes:
        name: the symbol name used in coefficient expressions, namespaced by the owning
            model (``"L3a.J"``).
        dimension: the physical dimension of the parameter.
        description: a one-line description, used in reports.

    """

    name: str
    dimension: Dimension = Dimension.ENERGY
    description: str = ""

    @property
    def symbol(self) -> Symbol:
        """The scalar symbol for this parameter."""
        return Symbol(self.name)

    @property
    def local_name(self) -> str:
        """The part of the name after the last dot, that is, the name without its namespace."""
        return self.name.rsplit(".", 1)[-1]

    def __str__(self) -> str:
        unit = f" [{self.dimension.unit}]" if self.dimension.unit else ""
        return f"{self.name}{unit}"


@dataclass(frozen=True)
class ParameterSet:
    """An ordered, name-indexed collection of parameters with distinct names.

    Attributes:
        parameters: the parameters, in declaration order.

    """

    parameters: tuple[Parameter, ...] = ()

    def __post_init__(self) -> None:
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            msg = f"duplicate parameter names: {names}"
            raise ValueError(msg)

    @property
    def names(self) -> tuple[str, ...]:
        """The parameter names, in declaration order."""
        return tuple(parameter.name for parameter in self.parameters)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self.parameters)

    def __len__(self) -> int:
        return len(self.parameters)

    def get(self, name: str) -> Parameter:
        """The parameter of that name.

        Raises:
            KeyError: if no parameter has that name.

        """
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        msg = f"no parameter named {name!r}; have {self.names}"
        raise KeyError(msg)

    def union(self, other: ParameterSet) -> ParameterSet:
        """The union with another set, keeping this set's entry on a name clash."""
        known = set(self.names)
        extra = tuple(p for p in other.parameters if p.name not in known)
        return ParameterSet(self.parameters + extra)

    def symbols(self) -> dict[str, Symbol]:
        """A mapping from local name to scalar symbol, used to build expressions."""
        return {parameter.local_name: parameter.symbol for parameter in self.parameters}

    def __str__(self) -> str:
        return ", ".join(str(parameter) for parameter in self.parameters)


@dataclass(frozen=True)
class Namespace:
    """The prefix carried by the parameter names of a model.

    ```python
    hardware = Namespace("device")
    hardware("J")  # 'device.J'
    hardware.parameter("J", description="tunnelling")
    ```

    Attributes:
        prefix: the namespace, without the separating dot.

    """

    prefix: str

    def __post_init__(self) -> None:
        if not self.prefix or "." in self.prefix:
            msg = f"a namespace prefix must be a non-empty dotless string, got {self.prefix!r}"
            raise ValueError(msg)

    def __call__(self, local: str) -> str:
        """The fully qualified name of a local parameter name."""
        return f"{self.prefix}.{local}"

    def symbol(self, local: str) -> Symbol:
        """The scalar symbol of a local parameter name."""
        return Symbol(self(local))

    def parameter(
        self,
        local: str,
        dimension: Dimension = Dimension.ENERGY,
        description: str = "",
    ) -> Parameter:
        """A [`Parameter`][qsimod.parameters.Parameter] in this namespace."""
        return Parameter(self(local), dimension, description)

    def parameter_set(self, *locals_: tuple[str, Dimension, str]) -> ParameterSet:
        """A parameter set from ``(local name, dimension, description)`` triples."""
        return ParameterSet(
            tuple(self.parameter(local, dimension, text) for local, dimension, text in locals_)
        )

    def __str__(self) -> str:
        return self.prefix


@dataclass(frozen=True)
class Bound:
    """A box bound on one knob, with optional strictness.

    ``J in (0, J_max]`` is ``Bound("J", 0.0, j_max, strict_lower=True)``.

    Attributes:
        parameter: the name of the knob.
        lower: the lower endpoint; may be ``-inf``.
        upper: the upper endpoint; may be ``+inf``.
        strict_lower: whether the lower endpoint is excluded.
        strict_upper: whether the upper endpoint is excluded.
        margin: the distance inside a strict endpoint at which a point satisfies the bound.

    """

    parameter: str
    lower: float = -math.inf
    upper: float = math.inf
    strict_lower: bool = False
    strict_upper: bool = False
    margin: float = 0.0

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            msg = f"empty bound for {self.parameter}: [{self.lower}, {self.upper}]"
            raise ValueError(msg)

    @property
    def interval(self) -> Interval:
        """The bound as an interval, ignoring strictness."""
        return Interval(self.lower, self.upper)

    @property
    def effective_lower(self) -> float:
        """The lower endpoint after the strictness margin."""
        return self.lower + (self.margin if self.strict_lower else 0.0)

    @property
    def effective_upper(self) -> float:
        """The upper endpoint after the strictness margin."""
        return self.upper - (self.margin if self.strict_upper else 0.0)

    def violation(self, value: float) -> float:
        """The distance by which ``value`` lies outside the bound.

        The distance is zero at a satisfied bound and at an excluded endpoint alike;
        [`contains`][qsimod.parameters.Bound.contains] decides satisfaction.
        """
        low = self.effective_lower
        high = self.effective_upper
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0

    def contains(self, value: float) -> bool:
        """Whether ``value`` satisfies the bound, honouring strict endpoints."""
        low = self.effective_lower
        high = self.effective_upper
        below = value <= low if self.strict_lower else value < low
        above = value >= high if self.strict_upper else value > high
        return not (below or above)

    def intersect(self, other: Bound) -> Bound:
        """The tighter of two bounds on the same knob.

        Raises:
            ValueError: if the bounds name different knobs, or their intersection is empty.

        """
        if other.parameter != self.parameter:
            msg = f"cannot intersect bounds on {self.parameter!r} and {other.parameter!r}"
            raise ValueError(msg)
        if self.lower > other.lower:
            lower, strict_lower = self.lower, self.strict_lower
        elif other.lower > self.lower:
            lower, strict_lower = other.lower, other.strict_lower
        else:
            lower, strict_lower = self.lower, self.strict_lower or other.strict_lower
        if self.upper < other.upper:
            upper, strict_upper = self.upper, self.strict_upper
        elif other.upper < self.upper:
            upper, strict_upper = other.upper, other.strict_upper
        else:
            upper, strict_upper = self.upper, self.strict_upper or other.strict_upper
        return Bound(
            self.parameter,
            lower,
            upper,
            strict_lower=strict_lower,
            strict_upper=strict_upper,
            margin=max(self.margin, other.margin),
        )

    def __str__(self) -> str:
        left = "(" if self.strict_lower else "["
        right = ")" if self.strict_upper else "]"
        return f"{self.parameter} in {left}{self.lower:.6g}, {self.upper:.6g}{right}"


class ConstraintOrigin(Enum):
    """The origin of a coupled constraint.

    See [`AdmissibleSet.apparatus_only`][qsimod.parameters.AdmissibleSet.apparatus_only].
    """

    APPARATUS = "apparatus"
    """A limit of the apparatus, independent of the theory derived on it."""

    DERIVATION = "derivation"
    """A requirement of one particular derivation, which may be dropped for a different
    theory."""

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class InequalityConstraint:
    """A coupled constraint between knobs, in the form ``expression >= 0``.

    Attributes:
        name: a short name, used in reports.
        expression: the constrained quantity, satisfied where it is non-negative.
        description: a one-line description.
        origin: whether the constraint is a limit of the apparatus, the default, or a
            requirement of one derivation.

    """

    name: str
    expression: Scalar
    description: str = ""
    origin: ConstraintOrigin = ConstraintOrigin.APPARATUS

    def slack(self, assignment: Mapping[str, float]) -> float:
        """The value of the expression; the constraint is satisfied where it is ``>= 0``."""
        return self.expression.evaluate_real(assignment)

    def satisfied(self, assignment: Mapping[str, float]) -> bool:
        """Whether the constraint holds at ``assignment``."""
        return self.slack(assignment) >= 0.0

    def __str__(self) -> str:
        return f"{self.name} [{self.origin}]: {self.expression} >= 0"


@dataclass(frozen=True)
class ConstraintViolation:
    """A record of one admissibility constraint that a point fails.

    Attributes:
        constraint: a rendering of the violated constraint.
        parameter: the knob involved, if the constraint is a box bound.
        value: the offending value, or the value of the constraint expression.
        amount: the amount by which the constraint is violated.

    """

    constraint: str
    parameter: str | None
    value: float
    amount: float

    def __str__(self) -> str:
        where = (
            f" ({self.parameter} = {self.value:.6g})"
            if self.parameter
            else f" (= {self.value:.6g})"
        )
        return f"{self.constraint} violated by {self.amount:.6g}{where}"


@dataclass(frozen=True)
class AdmissibleSet:
    """The set of knob settings to which a hardware model can be tuned.

    Attributes:
        bounds: the box bounds, at most one per knob.
        constraints: the coupled constraints between knobs.
        description: a name for the operating envelope of the hardware model.

    """

    bounds: tuple[Bound, ...] = ()
    constraints: tuple[InequalityConstraint, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        names = [bound.parameter for bound in self.bounds]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = (
                f"admissible set {self.description!r} bounds {duplicates} more than once; "
                "intersect the bounds instead"
            )
            raise ValueError(msg)

    def bound_for(self, parameter: str) -> Bound | None:
        """The box bound on ``parameter``, if declared."""
        for bound in self.bounds:
            if bound.parameter == parameter:
                return bound
        return None

    @property
    def parameters(self) -> tuple[str, ...]:
        """The knobs constrained by the set, box-bounded knobs first, then those in constraints."""
        seen: list[str] = [bound.parameter for bound in self.bounds]
        for constraint in self.constraints:
            for name in sorted(constraint.expression.symbols()):
                if name not in seen:
                    seen.append(name)
        return tuple(seen)

    def interval_environment(self) -> dict[str, Interval]:
        """The box bounds as an interval environment, used in reachability analysis."""
        return {bound.parameter: bound.interval for bound in self.bounds}

    def violations(self, assignment: Mapping[str, float]) -> list[ConstraintViolation]:
        """Every constraint that ``assignment`` fails, the box bounds first."""
        found: list[ConstraintViolation] = []
        for bound in self.bounds:
            if bound.parameter not in assignment:
                continue
            value = assignment[bound.parameter]
            if not bound.contains(value):
                amount = bound.violation(value)
                found.append(ConstraintViolation(str(bound), bound.parameter, value, amount))
        for constraint in self.constraints:
            if not constraint.expression.symbols() <= set(assignment):
                continue
            slack = constraint.slack(assignment)
            if slack < 0.0:
                found.append(ConstraintViolation(str(constraint), None, slack, -slack))
        return found

    def admits(self, assignment: Mapping[str, float]) -> bool:
        """Whether ``assignment`` lies in the admissible set."""
        return not self.violations(assignment)

    def constraints_from(self, origin: ConstraintOrigin) -> tuple[InequalityConstraint, ...]:
        """The coupled constraints of one origin."""
        return tuple(c for c in self.constraints if c.origin is origin)

    def apparatus_only(self) -> AdmissibleSet:
        """A copy carrying the box bounds and the apparatus constraints alone.

        Constraints marked [`DERIVATION`][qsimod.parameters.ConstraintOrigin] are dropped.

        Returns:
            The reach of the apparatus alone.

        """
        kept = self.constraints_from(ConstraintOrigin.APPARATUS)
        if len(kept) == len(self.constraints):
            return self
        return replace(self, constraints=kept, description=f"{self.description}, as a device")

    def union(self, other: AdmissibleSet) -> AdmissibleSet:
        """Conjoin two admissible sets: every constraint, and the tighter of the bounds per knob."""
        merged = {bound.parameter: bound for bound in self.bounds}
        for bound in other.bounds:
            existing = merged.get(bound.parameter)
            merged[bound.parameter] = bound if existing is None else existing.intersect(bound)
        return AdmissibleSet(
            bounds=tuple(merged.values()),
            constraints=self.constraints + other.constraints,
            description=f"{self.description} & {other.description}".strip(" &"),
        )

    def with_pinned(self, pinned: Mapping[str, float]) -> AdmissibleSet:
        """A copy in which each named knob is pinned to a single value."""
        extra = tuple(
            Bound(parameter=name, lower=value, upper=value)
            for name, value in sorted(pinned.items())
        )
        kept = tuple(bound for bound in self.bounds if bound.parameter not in pinned)
        return AdmissibleSet(kept + extra, self.constraints, self.description)

    def __str__(self) -> str:
        parts = [str(bound) for bound in self.bounds]
        parts += [str(constraint) for constraint in self.constraints]
        head = self.description or "admissible set"
        return f"{head}: " + "; ".join(parts)


def box(**bounds: tuple[float, float]) -> AdmissibleSet:
    """Build an [`AdmissibleSet`][qsimod.parameters.AdmissibleSet] of plain box bounds.

    Args:
        **bounds: a ``(low, high)`` pair per knob name.

    Returns:
        The admissible set.

    """
    return AdmissibleSet(
        bounds=tuple(Bound(name, low, high) for name, (low, high) in sorted(bounds.items()))
    )


EMPTY_ADMISSIBLE_SET = AdmissibleSet(description="unconstrained")
"""The admissible set of a model that declares no knob limits."""


@dataclass(frozen=True)
class Binding:
    """An assignment of values to parameter names.

    Attributes:
        values: the bound values by name; may be partial or empty.

    """

    values: Mapping[str, float] = field(default_factory=dict)

    def __contains__(self, name: object) -> bool:
        return name in self.values

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def get(self, name: str, default: float | None = None) -> float | None:
        """The value bound to ``name``, or ``default``."""
        return self.values.get(name, default)

    def with_values(self, **values: float) -> Binding:
        """A copy with additional or overridden values."""
        return Binding({**self.values, **values})

    def merged(self, other: Mapping[str, float]) -> Binding:
        """A copy with the values of ``other`` merged in, overriding those of this binding."""
        return Binding({**self.values, **other})

    def as_dict(self) -> dict[str, float]:
        """A plain mutable copy of the assignment."""
        return dict(self.values)

    def covers(self, names: Iterable[str]) -> bool:
        """Whether every name in ``names`` is bound."""
        return all(name in self.values for name in names)

    def missing(self, names: Iterable[str]) -> tuple[str, ...]:
        """The names in ``names`` that are not bound, sorted."""
        return tuple(sorted(name for name in names if name not in self.values))

    def __str__(self) -> str:
        return ", ".join(f"{name}={value:.6g}" for name, value in sorted(self.values.items()))
