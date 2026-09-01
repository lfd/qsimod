"""Affine arithmetic: interval bounds that retain first-order correlations between symbols.

An affine form is ``x = x0 + sum_i x_i e_i + e * eta``, where each ``e_i`` is an unknown in
``[-1, 1]`` shared between forms, one per bounded symbol, and ``eta`` is a private unknown
carrying the non-linear remainder.  Every operation over-approximates, so that
[`bound`][qsimod.affine.bound] returns a sound enclosure, which may be intersected with the
plain interval bound.  Sub-expressions that admit no affine approximation enter as their
interval bound, as a form without shared dependence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce

from qsimod.scalar import Abs, Add, Const, Interval, Mul, Pow, Scalar, Symbol

__all__ = ["AffineForm", "bound"]


@dataclass(frozen=True)
class AffineForm:
    """A value as a centre, a linear part over shared unknowns, and a private error.

    Attributes:
        center: the constant term.
        deviations: the coefficient of each shared unknown, keyed by symbol index; absent
            keys are zero.
        error: the non-negative radius of the private unknown.

    """

    center: float
    deviations: Mapping[int, float] = field(default_factory=dict)
    error: float = 0.0

    @classmethod
    def of_interval(cls, interval: Interval, index: int | None = None) -> AffineForm:
        """A form for a quantity known only to lie in ``interval``.

        Args:
            interval: the bounds, which must be finite.
            index: the shared unknown that carries the width; ``None`` places the whole width
                in the private error.

        Returns:
            The form.

        """
        center = 0.5 * (interval.lo + interval.hi)
        radius = 0.5 * (interval.hi - interval.lo)
        if index is None:
            return cls(center, {}, radius)
        return cls(center, {index: radius})

    @property
    def radius(self) -> float:
        """The largest deviation of the form from its centre."""
        return sum(abs(value) for value in self.deviations.values()) + self.error

    @property
    def interval(self) -> Interval:
        """The interval that contains the form."""
        return Interval(self.center - self.radius, self.center + self.radius)

    def __add__(self, other: AffineForm) -> AffineForm:
        deviations = dict(self.deviations)
        for index, value in other.deviations.items():
            deviations[index] = deviations.get(index, 0.0) + value
        return AffineForm(
            self.center + other.center,
            {index: value for index, value in deviations.items() if value != 0.0},
            self.error + other.error,
        )

    def scaled(self, factor: float) -> AffineForm:
        """The form multiplied by a constant."""
        return AffineForm(
            self.center * factor,
            {index: value * factor for index, value in self.deviations.items()},
            self.error * abs(factor),
        )

    def shifted(self, offset: float) -> AffineForm:
        """The form with a constant added."""
        return AffineForm(self.center + offset, dict(self.deviations), self.error)

    def widened(self, extra: float) -> AffineForm:
        """The form with additional private error."""
        return AffineForm(self.center, dict(self.deviations), self.error + abs(extra))

    def __mul__(self, other: AffineForm) -> AffineForm:
        """The product, with the second-order remainder folded into the private error."""
        linear = {index: value * other.center for index, value in self.deviations.items()}
        for index, value in other.deviations.items():
            linear[index] = linear.get(index, 0.0) + value * self.center
        spread = sum(abs(value) for value in self.deviations.values())
        other_spread = sum(abs(value) for value in other.deviations.values())
        remainder = (
            spread * other_spread
            + self.error * (abs(other.center) + other_spread + other.error)
            + other.error * (abs(self.center) + spread)
        )
        return AffineForm(
            self.center * other.center,
            {index: value for index, value in linear.items() if value != 0.0},
            remainder,
        )


def bound(expression: Scalar, env: Mapping[str, Interval]) -> Interval | None:
    """Bound ``expression`` by affine arithmetic over an interval environment.

    Args:
        expression: the expression to bound.
        env: an interval per symbol.  A symbol absent from ``env``, or one with an unbounded
            interval, causes the traversal to abandon the affine bound.

    Returns:
        A sound interval, or ``None`` if no affine bound can be formed.

    """
    indices = {name: index for index, name in enumerate(sorted(env))}
    form = _form(expression, env, indices)
    if form is None:
        return None
    result = form.interval
    return result if result.is_bounded else None


def _form(
    expression: Scalar,
    env: Mapping[str, Interval],
    indices: Mapping[str, int],
) -> AffineForm | None:
    """The affine form of an expression, or ``None`` if it has an unbounded symbol."""
    if isinstance(expression, Const):
        value = complex(expression.value)
        return None if value.imag else AffineForm(value.real)
    if isinstance(expression, Symbol):
        interval = env.get(expression.name)
        if interval is None or not interval.is_bounded:
            return None
        return AffineForm.of_interval(interval, indices[expression.name])
    if isinstance(expression, Add):
        forms = _forms(expression.terms, env, indices)
        return None if forms is None else reduce(AffineForm.__add__, forms)
    if isinstance(expression, Mul):
        forms = _forms(expression.factors, env, indices)
        return None if forms is None else reduce(AffineForm.__mul__, forms)
    if isinstance(expression, Pow):
        return _power(expression, env, indices)
    if isinstance(expression, Abs):
        return _magnitude(expression, env, indices)
    return _fallback(expression, env)


def _forms(
    parts: tuple[Scalar, ...],
    env: Mapping[str, Interval],
    indices: Mapping[str, int],
) -> list[AffineForm] | None:
    """The affine form of every part, or ``None`` if any part has none."""
    found: list[AffineForm] = []
    for part in parts:
        form = _form(part, env, indices)
        if form is None:
            return None
        found.append(form)
    return found


def _power(
    expression: Pow,
    env: Mapping[str, Interval],
    indices: Mapping[str, int],
) -> AffineForm | None:
    """The affine form of a power, where the exponent admits one."""
    base = _form(expression.base, env, indices)
    if base is None:
        return None
    exponent = expression.exponent
    if exponent == 0.0:
        return AffineForm(1.0)
    if exponent < 0.0:
        inverted = _reciprocal(base)
        if inverted is None:
            return None
        return inverted if exponent == -1.0 else _power_of(inverted, -exponent)
    return _power_of(base, exponent)


def _power_of(base: AffineForm, exponent: float) -> AffineForm | None:
    """A non-negative power: repeated multiplication, or a square root."""
    if exponent == int(exponent):
        total = AffineForm(1.0)
        for _ in range(int(exponent)):
            total = total * base
        return total
    if exponent == 0.5:
        return _square_root(base)
    return _from_interval_power(base, exponent)


def _from_interval_power(base: AffineForm, exponent: float) -> AffineForm | None:
    """The interval bound of a power without an affine approximation, entered as a bare form."""
    interval = base.interval**exponent
    if not interval.is_bounded:
        return None
    return AffineForm.of_interval(interval)


def _reciprocal(form: AffineForm) -> AffineForm | None:
    """``1/x`` by the min-range affine approximation on a sign-definite interval.

    On ``[a, b]`` with ``0 < a <= b`` the slope is ``-1/b**2``, the offset is the midpoint of the
    residual ``1/x - slope*x`` at the endpoints, and the error is half the spread of the
    residual.  Negative intervals are treated by symmetry; an interval that contains zero has
    no bounded reciprocal.
    """
    interval = form.interval
    if interval.straddles_zero():
        return None
    if interval.hi < 0.0:
        negated = _reciprocal(form.scaled(-1.0))
        return None if negated is None else negated.scaled(-1.0)
    low, high = interval.lo, interval.hi
    slope = -1.0 / (high * high)
    at_low = 1.0 / low - slope * low
    at_high = 1.0 / high - slope * high
    offset = 0.5 * (at_low + at_high)
    return form.scaled(slope).shifted(offset).widened(0.5 * abs(at_low - at_high))


def _square_root(form: AffineForm) -> AffineForm | None:
    """``sqrt(x)`` on a strictly positive interval, by the min-range construction."""
    interval = form.interval
    if interval.lo <= 0.0:
        return _from_interval_power(form, 0.5)
    low, high = interval.lo, interval.hi
    slope = 0.5 / math.sqrt(high)
    at_low = math.sqrt(low) - slope * low
    at_high = math.sqrt(high) - slope * high
    offset = 0.5 * (at_low + at_high)
    return form.scaled(slope).shifted(offset).widened(0.5 * abs(at_low - at_high))


def _magnitude(
    expression: Abs,
    env: Mapping[str, Interval],
    indices: Mapping[str, int],
) -> AffineForm | None:
    """``|x|``, exact where the argument has a definite sign and an interval where it does not."""
    inner = _form(expression.argument, env, indices)
    if inner is None:
        return None
    interval = inner.interval
    if interval.lo >= 0.0:
        return inner
    if interval.hi <= 0.0:
        return inner.scaled(-1.0)
    return AffineForm.of_interval(Interval(0.0, max(-interval.lo, interval.hi)))


def _fallback(expression: Scalar, env: Mapping[str, Interval]) -> AffineForm | None:
    """An unrecognised node, entered as its interval without shared dependence."""
    interval = expression.interval(env)
    if not interval.is_bounded:
        return None
    return AffineForm.of_interval(interval)
