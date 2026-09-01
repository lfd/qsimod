"""A symbolic scalar algebra for parameters and coefficients.

The algebra has six node kinds, [`Const`][qsimod.scalar.Const],
[`Symbol`][qsimod.scalar.Symbol], [`Add`][qsimod.scalar.Add], [`Mul`][qsimod.scalar.Mul],
[`Pow`][qsimod.scalar.Pow] and [`Abs`][qsimod.scalar.Abs], with evaluation, substitution,
interval arithmetic and [`simplify`][qsimod.scalar.simplify].  Evaluation uses only ``+``,
``*``, ``**`` and ``abs``, so that it applies to Python numbers and to JAX scalars alike.
``str()`` renders the simplified expression; the stored expression is unchanged.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

__all__ = [
    "Abs",
    "Add",
    "Const",
    "DomainError",
    "Interval",
    "Mul",
    "Pow",
    "Scalar",
    "ScalarLike",
    "Symbol",
    "absolute",
    "as_scalar",
    "envelope",
    "simplify",
    "sqrt",
    "summation",
]

#: Any value acceptable where a [`Scalar`][qsimod.scalar.Scalar] is expected.
ScalarLike: TypeAlias = "Scalar | int | float | complex"

#: A value of an evaluation environment: a Python number, or a JAX/NumPy scalar.
NumericValue: TypeAlias = Any


class DomainError(ZeroDivisionError):
    """The error raised when a scalar expression is evaluated at a division by zero.

    Attributes:
        expression: the vanishing sub-expression.

    """

    def __init__(self, expression: Scalar, message: str | None = None) -> None:
        self.expression = expression
        super().__init__(message or f"undefined: division by zero in {expression}")


class Scalar(ABC):
    """A symbolic scalar expression over named parameters."""

    # -- interface ---------------------------------------------------------------

    @abstractmethod
    def evaluate(self, env: Mapping[str, NumericValue]) -> NumericValue:
        """Evaluate the expression.

        Args:
            env: the values of the free symbols.  Python numbers yield a Python number;
                JAX scalars yield a traceable JAX scalar.

        Returns:
            The value, in the numeric type supplied by ``env``.

        Raises:
            KeyError: if a free symbol is missing from ``env``.
            DomainError: on a division by zero.

        """

    @abstractmethod
    def symbols(self) -> frozenset[str]:
        """The names of the free symbols occurring in the expression."""

    @abstractmethod
    def substitute(self, env: Mapping[str, ScalarLike]) -> Scalar:
        """Replace symbols by expressions or values and return a new expression."""

    @abstractmethod
    def interval(self, env: Mapping[str, Interval]) -> Interval:
        """Bound the expression by interval arithmetic, given interval bounds on its symbols.

        Args:
            env: an interval per symbol; unbounded symbols default to the whole real line.

        Returns:
            A sound enclosure, not necessarily tight.

        """

    @abstractmethod
    def _render(self) -> str: ...

    # -- conveniences ------------------------------------------------------------

    def evaluate_real(self, env: Mapping[str, NumericValue]) -> float:
        """Evaluate the expression and return a real ``float``.

        Args:
            env: the values of the free symbols.

        Returns:
            The real part of the value.

        Raises:
            ValueError: if the value has a non-negligible imaginary part.

        """
        value = complex(self.evaluate(env))
        if abs(value.imag) > 1e-12 * max(1.0, abs(value.real)):
            msg = f"{self} evaluated to a complex value {value!r} where a real was expected"
            raise ValueError(msg)
        return value.real

    def is_constant(self) -> bool:
        """Whether the expression has no free symbols."""
        return not self.symbols()

    def simplified(self) -> Scalar:
        """The expression with constants folded and nesting flattened.

        See [`simplify`][qsimod.scalar.simplify].
        """
        return simplify(self)

    def __str__(self) -> str:
        return _strip_outer_brackets(self.simplified()._render())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._render()})"

    # -- arithmetic --------------------------------------------------------------

    def __add__(self, other: ScalarLike) -> Scalar:
        return Add((self, as_scalar(other)))

    def __radd__(self, other: ScalarLike) -> Scalar:
        return Add((as_scalar(other), self))

    def __neg__(self) -> Scalar:
        return Mul((Const(-1), self))

    def __sub__(self, other: ScalarLike) -> Scalar:
        return Add((self, -as_scalar(other)))

    def __rsub__(self, other: ScalarLike) -> Scalar:
        return Add((as_scalar(other), -self))

    def __mul__(self, other: ScalarLike) -> Scalar:
        return Mul((self, as_scalar(other)))

    def __rmul__(self, other: ScalarLike) -> Scalar:
        return Mul((as_scalar(other), self))

    def __truediv__(self, other: ScalarLike) -> Scalar:
        return Mul((self, Pow(as_scalar(other), -1.0)))

    def __rtruediv__(self, other: ScalarLike) -> Scalar:
        numerator = as_scalar(other)
        if isinstance(numerator, Const) and numerator.value == 1:
            return Pow(self, -1.0)
        return Mul((numerator, Pow(self, -1.0)))

    def __pow__(self, exponent: float) -> Scalar:
        return Pow(self, exponent)


def as_scalar(value: ScalarLike) -> Scalar:
    """Coerce a number to a [`Const`][qsimod.scalar.Const]; a `Scalar` is passed through.

    Args:
        value: a `Scalar` or a number.

    Returns:
        The scalar expression.

    Raises:
        TypeError: if ``value`` is neither a number nor a [`Scalar`][qsimod.scalar.Scalar].

    """
    if isinstance(value, Scalar):
        return value
    try:
        return Const(complex(value))
    except (TypeError, ValueError) as error:
        msg = f"cannot use {value!r} of type {type(value).__name__} as a scalar expression"
        raise TypeError(msg) from error


@dataclass(frozen=True)
class Const(Scalar):
    """A numeric literal."""

    value: complex

    def evaluate(self, env: Mapping[str, NumericValue]) -> NumericValue:  # noqa: ARG002
        """Return the literal value."""
        if self.value.imag == 0.0:
            return self.value.real
        return self.value

    def symbols(self) -> frozenset[str]:
        """The empty set; a literal has no free symbols."""
        return frozenset()

    def substitute(self, env: Mapping[str, ScalarLike]) -> Scalar:  # noqa: ARG002
        """A literal is unchanged by substitution."""
        return self

    def interval(self, env: Mapping[str, Interval]) -> Interval:  # noqa: ARG002
        """A literal is a degenerate interval.

        Raises:
            ValueError: if the literal is complex.

        """
        real = complex(self.value)
        if real.imag != 0.0:
            msg = f"cannot bound the complex constant {self.value!r} on the real line"
            raise ValueError(msg)
        return Interval(real.real, real.real)

    def _render(self) -> str:
        val = complex(self.value)
        if val.imag == 0.0:
            return _render_real(val.real)
        if val.real == 0.0:
            magnitude = _render_real(abs(val.imag))
            sign = "-" if val.imag < 0 else ""
            return f"{sign}i" if magnitude == "1" else f"{sign}{magnitude}i"
        sign = "+" if val.imag > 0 else "-"
        return f"({_render_real(val.real)}{sign}{_render_real(abs(val.imag))}i)"


def _is_integral(value: float) -> bool:
    """Whether ``value`` is finite and a whole number."""
    return math.isfinite(value) and value == int(value)


def _render_real(value: float) -> str:
    """Render a real number compactly, as an integer where it is one."""
    if _is_integral(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


@dataclass(frozen=True)
class Symbol(Scalar):
    """A named free parameter."""

    name: str

    def evaluate(self, env: Mapping[str, NumericValue]) -> NumericValue:
        """Look the symbol up in ``env``.

        Raises:
            KeyError: if the symbol is unbound.

        """
        try:
            return env[self.name]
        except KeyError:
            msg = f"unbound parameter {self.name!r}"
            raise KeyError(msg) from None

    def symbols(self) -> frozenset[str]:
        """The symbol itself."""
        return frozenset({self.name})

    def substitute(self, env: Mapping[str, ScalarLike]) -> Scalar:
        """Replace the symbol if ``env`` binds it."""
        if self.name in env:
            return as_scalar(env[self.name])
        return self

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        """Look up the bounds of the symbol in ``env``, defaulting to the whole real line."""
        return env.get(self.name, Interval(-math.inf, math.inf))

    def _render(self) -> str:
        return self.name


@dataclass(frozen=True)
class Add(Scalar):
    """A sum of two or more terms."""

    terms: tuple[Scalar, ...]

    def evaluate(self, env: Mapping[str, NumericValue]) -> NumericValue:
        """Sum the terms."""
        total = self.terms[0].evaluate(env)
        for term in self.terms[1:]:
            total = total + term.evaluate(env)
        return total

    def symbols(self) -> frozenset[str]:
        """The union of the symbols of the terms."""
        return frozenset().union(*(term.symbols() for term in self.terms))

    def substitute(self, env: Mapping[str, ScalarLike]) -> Scalar:
        """Substitute into every term."""
        return Add(tuple(term.substitute(env) for term in self.terms))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        """Sum the intervals of the terms."""
        result = self.terms[0].interval(env)
        for term in self.terms[1:]:
            result = result + term.interval(env)
        return result

    def _render(self) -> str:
        parts = [self.terms[0]._render()]
        for term in self.terms[1:]:
            rendered = term._render()
            if rendered.startswith("-"):
                parts.append(f" - {rendered[1:]}")
            else:
                parts.append(f" + {rendered}")
        return "(" + "".join(parts) + ")"


@dataclass(frozen=True)
class Mul(Scalar):
    """A product of two or more factors."""

    factors: tuple[Scalar, ...]

    def evaluate(self, env: Mapping[str, NumericValue]) -> NumericValue:
        """Multiply the factors."""
        total = self.factors[0].evaluate(env)
        for factor in self.factors[1:]:
            total = total * factor.evaluate(env)
        return total

    def symbols(self) -> frozenset[str]:
        """The union of the symbols of the factors."""
        return frozenset().union(*(factor.symbols() for factor in self.factors))

    def substitute(self, env: Mapping[str, ScalarLike]) -> Scalar:
        """Substitute into every factor."""
        return Mul(tuple(factor.substitute(env) for factor in self.factors))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        """Multiply the intervals of the factors."""
        result = self.factors[0].interval(env)
        for factor in self.factors[1:]:
            result = result * factor.interval(env)
        return result

    def _render(self) -> str:
        numerators: list[str] = []
        denominators: list[str] = []
        sign = ""
        for factor in self.factors:
            if isinstance(factor, Pow) and factor.exponent == -1.0:
                base = factor.base._render()
                needs_parens = any(char in base for char in "*+- ") and not base.startswith("(")
                denominators.append(f"({base})" if needs_parens else base)
                continue
            rendered = factor._render()
            if rendered == "-1":
                sign = "-"
                continue
            if rendered != "1":
                numerators.append(rendered)
        head = "*".join(numerators) or "1"
        if denominators:
            head = f"{head}/{'/'.join(denominators)}"
        return sign + head


@dataclass(frozen=True)
class Pow(Scalar):
    """A power with a fixed real exponent."""

    base: Scalar
    exponent: float

    def evaluate(self, env: Mapping[str, NumericValue]) -> NumericValue:
        """Raise the base to the exponent.

        Raises:
            DomainError: if the exponent is negative and the base vanishes.

        """
        base = self.base.evaluate(env)
        if self.exponent < 0.0 and _is_definitely_zero(base):
            raise DomainError(self.base)
        return base**self.exponent

    def symbols(self) -> frozenset[str]:
        """The symbols of the base."""
        return self.base.symbols()

    def substitute(self, env: Mapping[str, ScalarLike]) -> Scalar:
        """Substitute into the base."""
        return Pow(self.base.substitute(env), self.exponent)

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        """Bound the power."""
        return self.base.interval(env) ** self.exponent

    def _render(self) -> str:
        base = self.base._render()
        if self.exponent == -1.0:
            return f"1/{base}"
        if self.exponent == 0.5:
            return f"sqrt({base})"
        if _needs_brackets(base):
            base = f"({base})"
        return f"{base}**{self.exponent:g}"


@dataclass(frozen=True)
class Abs(Scalar):
    """The absolute value of an expression."""

    argument: Scalar

    def evaluate(self, env: Mapping[str, NumericValue]) -> NumericValue:
        """The magnitude of the argument."""
        return abs(self.argument.evaluate(env))

    def symbols(self) -> frozenset[str]:
        """The symbols of the argument."""
        return self.argument.symbols()

    def substitute(self, env: Mapping[str, ScalarLike]) -> Scalar:
        """Substitute into the argument."""
        return Abs(self.argument.substitute(env))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        """Bound the magnitude; an interval that contains zero has the least magnitude ``0``."""
        bounds = self.argument.interval(env)
        magnitudes = (abs(bounds.lo), abs(bounds.hi))
        lo = 0.0 if bounds.straddles_zero() else min(magnitudes)
        return Interval(lo, max(magnitudes))

    def _render(self) -> str:
        return f"|{self.argument._render()}|"


def _needs_brackets(rendered: str) -> bool:
    """Whether a rendered sub-expression binds more loosely than exponentiation.

    ``Add`` and ``Abs`` render their own delimiters; a bare literal or symbol requires none.
    """
    if rendered.startswith(("(", "|")):
        return False
    return any(character in rendered for character in "*/ +-")


def _is_definitely_zero(value: NumericValue) -> bool:
    """Whether ``value`` is a concrete (non-traced) zero."""
    try:
        return complex(value) == 0
    except (TypeError, ValueError):
        # A JAX tracer has no concrete value.
        return False


def simplify(expression: ScalarLike) -> Scalar:
    """Fold constants and flatten nesting without changing the value or domain of the expression.

    Nested sums and products are flattened, constant terms and factors combined, additive
    zeros and multiplicative ones dropped, unit exponents removed, and constants raised to
    integer powers evaluated.  Constants raised to fractional powers are not folded, nor are
    products with a zero factor whose other factors can raise a
    [`DomainError`][qsimod.scalar.DomainError].

    Args:
        expression: the expression to fold.

    Returns:
        An expression with the same value and domain as the original.

    """
    scalar = as_scalar(expression)
    if isinstance(scalar, Add):
        return _simplify_add(scalar)
    if isinstance(scalar, Mul):
        return _simplify_mul(scalar)
    if isinstance(scalar, Pow):
        return _simplify_pow(scalar)
    if isinstance(scalar, Abs):
        argument = simplify(scalar.argument)
        if isinstance(argument, Const):
            return Const(abs(argument.value))
        return Abs(argument.argument if isinstance(argument, Abs) else argument)
    return scalar


def _simplify_add(node: Add) -> Scalar:
    """Flatten nested sums, combine the constant terms and drop the additive zero."""
    terms: list[Scalar] = []
    total = 0j
    for term in node.terms:
        simplified = simplify(term)
        flattened = simplified.terms if isinstance(simplified, Add) else (simplified,)
        for part in flattened:
            if isinstance(part, Const):
                total += part.value
            else:
                terms.append(part)
    if total != 0 or not terms:
        terms.append(Const(total))
    if len(terms) == 1:
        return terms[0]
    return Add(tuple(terms))


def _simplify_mul(node: Mul) -> Scalar:
    """Flatten nested products, combine the constant factors and drop the multiplicative one."""
    factors: list[Scalar] = []
    total = 1 + 0j
    for factor in node.factors:
        simplified = simplify(factor)
        flattened = simplified.factors if isinstance(simplified, Mul) else (simplified,)
        for part in flattened:
            if isinstance(part, Const):
                total *= part.value
            else:
                factors.append(part)
    if total == 0 and not all(_is_total(factor) for factor in factors):
        # Keep the product: another factor may still raise a DomainError.
        return Mul((Const(total), *factors)) if factors else Const(total)
    if total == 0 or not factors:
        return Const(total)
    if total != 1:
        factors.insert(0, Const(total))
    if len(factors) == 1:
        return factors[0]
    return Mul(tuple(factors))


def _simplify_pow(node: Pow) -> Scalar:
    """Drop a unit exponent and evaluate a constant raised to an integral exponent."""
    base = simplify(node.base)
    if node.exponent == 1.0:
        return base
    if isinstance(base, Const) and _is_integral(node.exponent):
        if base.value == 0 and node.exponent < 0:
            return Pow(base, node.exponent)
        return Const(base.value ** int(node.exponent))
    return Pow(base, node.exponent)


def _is_total(expression: Scalar) -> bool:
    """Whether the expression is defined everywhere, that is, contains no negative power."""
    if isinstance(expression, Pow):
        return expression.exponent >= 0.0 and _is_total(expression.base)
    if isinstance(expression, Add):
        return all(_is_total(term) for term in expression.terms)
    if isinstance(expression, Mul):
        return all(_is_total(factor) for factor in expression.factors)
    if isinstance(expression, Abs):
        return _is_total(expression.argument)
    return True


def _strip_outer_brackets(rendered: str) -> str:
    """Drop one enclosing pair of brackets where it encloses the whole expression."""
    if not (rendered.startswith("(") and rendered.endswith(")")):
        return rendered
    depth = 0
    for position, character in enumerate(rendered):
        depth += (character == "(") - (character == ")")
        if depth == 0 and position < len(rendered) - 1:
            return rendered
    return rendered[1:-1]


def sqrt(value: ScalarLike) -> Scalar:
    """The square root of an expression."""
    return Pow(as_scalar(value), 0.5)


def absolute(value: ScalarLike) -> Scalar:
    """The magnitude of an expression."""
    return Abs(as_scalar(value))


def summation(terms: Iterable[ScalarLike]) -> Scalar:
    """The sum of an iterable of expressions; the empty sum is zero."""
    collected = tuple(as_scalar(term) for term in terms)
    if not collected:
        return Const(0)
    if len(collected) == 1:
        return collected[0]
    return Add(collected)


@dataclass(frozen=True)
class Interval:
    """A closed real interval with sound arithmetic.

    Division by an interval that contains zero yields the whole real line.

    Attributes:
        lo: the lower endpoint; may be ``-inf``.
        hi: the upper endpoint; may be ``+inf``.

    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            msg = f"empty interval [{self.lo}, {self.hi}]"
            raise ValueError(msg)

    @property
    def is_bounded(self) -> bool:
        """Whether both endpoints are finite."""
        return math.isfinite(self.lo) and math.isfinite(self.hi)

    def contains(self, value: float, tolerance: float = 0.0) -> bool:
        """Whether ``value`` lies in the interval, within ``tolerance``."""
        return self.lo - tolerance <= value <= self.hi + tolerance

    def straddles_zero(self) -> bool:
        """Whether the interval contains zero."""
        return self.lo <= 0.0 <= self.hi

    def __add__(self, other: Interval) -> Interval:
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self) -> Interval:
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other: Interval) -> Interval:
        return self + (-other)

    def __mul__(self, other: Interval) -> Interval:
        candidates = [
            _mul(self.lo, other.lo),
            _mul(self.lo, other.hi),
            _mul(self.hi, other.lo),
            _mul(self.hi, other.hi),
        ]
        return Interval(min(candidates), max(candidates))

    def __pow__(self, exponent: float) -> Interval:
        if exponent == -1.0:
            return self._reciprocal()
        if exponent < 0.0:
            return self._reciprocal() ** (-exponent)
        if _is_integral(exponent) and int(exponent) % 2 == 0:
            magnitudes = (abs(self.lo), abs(self.hi))
            hi = max(magnitudes) ** exponent
            lo = 0.0 if self.straddles_zero() else min(magnitudes) ** exponent
            return Interval(lo, hi)
        if self.lo < 0.0 and not _is_integral(exponent):
            return Interval(-math.inf, math.inf)
        return Interval(_signed_pow(self.lo, exponent), _signed_pow(self.hi, exponent))

    def _reciprocal(self) -> Interval:
        if self.straddles_zero():
            return Interval(-math.inf, math.inf)
        return Interval(1.0 / self.hi, 1.0 / self.lo)

    def __str__(self) -> str:
        return f"[{self.lo:.6g}, {self.hi:.6g}]"


def _mul(left: float, right: float) -> float:
    """Multiply two endpoints, treating ``0 * inf`` as ``0`` for degenerate intervals."""
    if left == 0.0 or right == 0.0:
        return 0.0
    return left * right


def _signed_pow(base: float, exponent: float) -> float:
    """``base ** exponent`` extended to negative bases with integral exponents."""
    if base < 0.0:
        magnitude = float((-base) ** exponent)
        return -magnitude if int(exponent) % 2 else magnitude
    if base == 0.0 and exponent < 0.0:
        return math.inf
    return float(base**exponent)


def envelope(intervals: Sequence[Interval]) -> Interval:
    """The smallest interval containing all of ``intervals``.

    Raises:
        ValueError: if ``intervals`` is empty.

    """
    if not intervals:
        msg = "envelope of no intervals is undefined"
        raise ValueError(msg)
    return Interval(min(i.lo for i in intervals), max(i.hi for i in intervals))
