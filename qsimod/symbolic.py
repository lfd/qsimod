"""Symbolic operators: site operators, terms and operator sums.

A Hamiltonian is a tuple of [`Term`][qsimod.symbolic.Term], each a
[`Scalar`][qsimod.scalar.Scalar] coefficient times an ordered tuple of
[`SiteOperator`][qsimod.symbolic.SiteOperator] factors.  Operator order within a term is
preserved and significant; no reordering or normal ordering is performed symbolically.
Exact commutators are computed in the Pauli representation of [`qsimod.pauli`][qsimod.pauli].
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from qsimod.scalar import Add, Const, Mul, Pow, Scalar, ScalarLike, as_scalar, summation
from qsimod.structure import Algebra

__all__ = [
    "ALLOWED_OPERATORS",
    "ZERO",
    "OpSymbol",
    "OperatorSum",
    "SiteOperator",
    "Term",
    "annihilate",
    "constant",
    "create",
    "electric_field",
    "identity_term",
    "link_u",
    "number",
    "parity",
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "sigma_minus",
    "sigma_plus",
    "spin_minus",
    "spin_plus",
    "spin_z",
    "word",
]


class OpSymbol(Enum):
    """A primitive single-site operator."""

    CREATE = "create"
    """The creation operator, fermionic or bosonic according to the algebra of the site."""
    ANNIHILATE = "annihilate"
    """The annihilation operator, fermionic or bosonic according to the algebra of the site."""
    NUMBER = "number"
    """The occupation number.  On a qubit it is ``(sigma^z + 1) / 2``."""
    PARITY = "parity"
    """``1 - 2n``, the fermionic parity of a two-level site, ``-Z`` on a qubit.  It is the factor
    of a Jordan-Wigner string; as a single operator, a string of length ``L`` is one word of
    ``L`` factors rather than ``2**L`` words."""
    SPIN_Z = "spin-z"
    """``S^z``, with eigenvalues ``+-1/2``."""
    SPIN_PLUS = "spin-plus"
    """``S^+``, the spin-1/2 raising operator."""
    SPIN_MINUS = "spin-minus"
    """``S^-``, the spin-1/2 lowering operator."""
    PAULI_X = "pauli-x"
    PAULI_Y = "pauli-y"
    PAULI_Z = "pauli-z"
    SIGMA_PLUS = "sigma-plus"
    """``sigma^+ = |n=1><n=0| = (X + iY)/2``."""
    SIGMA_MINUS = "sigma-minus"
    """``sigma^- = |n=0><n=1| = (X - iY)/2``."""
    LINK_U = "link-U"
    """The compact U(1) link operator of the Kogut-Susskind Hamiltonian."""
    LINK_U_DAGGER = "link-U-dagger"
    ELECTRIC_FIELD = "electric-field"
    """The link electric field ``E``, with unbounded spectrum."""

    @property
    def adjoint(self) -> OpSymbol:
        """The Hermitian adjoint of this operator."""
        return _ADJOINT[self]

    @property
    def is_self_adjoint(self) -> bool:
        """Whether the operator equals its own adjoint."""
        return self.adjoint is self

    def render(self, site: int, ladder_glyph: str = "b") -> str:
        """A compact unicode rendering of the operator at ``site``.

        Args:
            site: the register position.
            ladder_glyph: the letter used for the creation, annihilation and number operators,
                for instance the Greek letter psi for a fermionic site and ``"b"`` for a
                bosonic one.

        """
        return _RENDER[self].format(site=site, ladder=ladder_glyph)


_ADJOINT: dict[OpSymbol, OpSymbol] = {
    OpSymbol.CREATE: OpSymbol.ANNIHILATE,
    OpSymbol.ANNIHILATE: OpSymbol.CREATE,
    OpSymbol.NUMBER: OpSymbol.NUMBER,
    OpSymbol.PARITY: OpSymbol.PARITY,
    OpSymbol.SPIN_Z: OpSymbol.SPIN_Z,
    OpSymbol.SPIN_PLUS: OpSymbol.SPIN_MINUS,
    OpSymbol.SPIN_MINUS: OpSymbol.SPIN_PLUS,
    OpSymbol.PAULI_X: OpSymbol.PAULI_X,
    OpSymbol.PAULI_Y: OpSymbol.PAULI_Y,
    OpSymbol.PAULI_Z: OpSymbol.PAULI_Z,
    OpSymbol.SIGMA_PLUS: OpSymbol.SIGMA_MINUS,
    OpSymbol.SIGMA_MINUS: OpSymbol.SIGMA_PLUS,
    OpSymbol.LINK_U: OpSymbol.LINK_U_DAGGER,
    OpSymbol.LINK_U_DAGGER: OpSymbol.LINK_U,
    OpSymbol.ELECTRIC_FIELD: OpSymbol.ELECTRIC_FIELD,
}

_RENDER: dict[OpSymbol, str] = {
    OpSymbol.CREATE: "{ladder}†_{site}",
    OpSymbol.ANNIHILATE: "{ladder}_{site}",
    OpSymbol.NUMBER: "n_{site}",
    OpSymbol.PARITY: "P_{site}",
    OpSymbol.SPIN_Z: "Sᶻ_{site}",
    OpSymbol.SPIN_PLUS: "S⁺_{site}",
    OpSymbol.SPIN_MINUS: "S⁻_{site}",
    OpSymbol.PAULI_X: "X_{site}",
    OpSymbol.PAULI_Y: "Y_{site}",
    OpSymbol.PAULI_Z: "Z_{site}",
    OpSymbol.SIGMA_PLUS: "σ⁺_{site}",  # noqa: RUF001 - the Pauli ladder glyph, not a letter o
    OpSymbol.SIGMA_MINUS: "σ⁻_{site}",  # noqa: RUF001 - the Pauli ladder glyph, not a letter o
    OpSymbol.LINK_U: "U_{site}",
    OpSymbol.LINK_U_DAGGER: "U†_{site}",
    OpSymbol.ELECTRIC_FIELD: "E_{site}",
}

#: The primitive operators admissible on a site, by the algebra of the site.  Enforced by
#: [`OperatorSum.validate_against`][qsimod.symbolic.OperatorSum.validate_against].
ALLOWED_OPERATORS: dict[Algebra, frozenset[OpSymbol]] = {
    Algebra.FERMION: frozenset(
        {OpSymbol.CREATE, OpSymbol.ANNIHILATE, OpSymbol.NUMBER, OpSymbol.PARITY}
    ),
    Algebra.BOSON: frozenset({OpSymbol.CREATE, OpSymbol.ANNIHILATE, OpSymbol.NUMBER}),
    Algebra.SPIN_HALF: frozenset({OpSymbol.SPIN_Z, OpSymbol.SPIN_PLUS, OpSymbol.SPIN_MINUS}),
    Algebra.QUBIT: frozenset(
        {
            OpSymbol.PAULI_X,
            OpSymbol.PAULI_Y,
            OpSymbol.PAULI_Z,
            OpSymbol.SIGMA_PLUS,
            OpSymbol.SIGMA_MINUS,
            OpSymbol.NUMBER,
            OpSymbol.PARITY,
        }
    ),
    Algebra.GAUGE_LINK_U1: frozenset(
        {OpSymbol.LINK_U, OpSymbol.LINK_U_DAGGER, OpSymbol.ELECTRIC_FIELD}
    ),
}


@dataclass(frozen=True, order=True)
class SiteOperator:
    """A primitive operator acting on one register position."""

    symbol: OpSymbol
    site: int

    @property
    def adjoint(self) -> SiteOperator:
        """The Hermitian adjoint, on the same site."""
        return SiteOperator(self.symbol.adjoint, self.site)

    def render(self, ladder_glyph: str = "b") -> str:
        """A compact unicode rendering, with ``ladder_glyph`` for the ladder operators."""
        return self.symbol.render(self.site, ladder_glyph)

    def __str__(self) -> str:
        return self.symbol.render(self.site)


def create(site: int) -> SiteOperator:
    """A creation operator on ``site``."""
    return SiteOperator(OpSymbol.CREATE, site)


def annihilate(site: int) -> SiteOperator:
    """An annihilation operator on ``site``."""
    return SiteOperator(OpSymbol.ANNIHILATE, site)


def number(site: int) -> SiteOperator:
    """An occupation-number operator on ``site``."""
    return SiteOperator(OpSymbol.NUMBER, site)


def parity(site: int) -> SiteOperator:
    """``1 - 2 n_site``, the parity of a fermionic or qubit site."""
    return SiteOperator(OpSymbol.PARITY, site)


def spin_z(site: int) -> SiteOperator:
    """``S^z`` on ``site``."""
    return SiteOperator(OpSymbol.SPIN_Z, site)


def spin_plus(site: int) -> SiteOperator:
    """``S^+`` on ``site``."""
    return SiteOperator(OpSymbol.SPIN_PLUS, site)


def spin_minus(site: int) -> SiteOperator:
    """``S^-`` on ``site``."""
    return SiteOperator(OpSymbol.SPIN_MINUS, site)


def pauli_x(site: int) -> SiteOperator:
    """``X`` on ``site``."""
    return SiteOperator(OpSymbol.PAULI_X, site)


def pauli_y(site: int) -> SiteOperator:
    """``Y`` on ``site``."""
    return SiteOperator(OpSymbol.PAULI_Y, site)


def pauli_z(site: int) -> SiteOperator:
    """``Z`` on ``site``."""
    return SiteOperator(OpSymbol.PAULI_Z, site)


def sigma_plus(site: int) -> SiteOperator:
    """``sigma^+`` on ``site``."""
    return SiteOperator(OpSymbol.SIGMA_PLUS, site)


def sigma_minus(site: int) -> SiteOperator:
    """``sigma^-`` on ``site``."""
    return SiteOperator(OpSymbol.SIGMA_MINUS, site)


def link_u(site: int, *, dagger: bool = False) -> SiteOperator:
    """The compact link operator ``U``, or ``U^dagger``, on ``site``."""
    return SiteOperator(OpSymbol.LINK_U_DAGGER if dagger else OpSymbol.LINK_U, site)


def electric_field(site: int) -> SiteOperator:
    """The link electric field ``E`` on ``site``."""
    return SiteOperator(OpSymbol.ELECTRIC_FIELD, site)


@dataclass(frozen=True)
class Term:
    """A coefficient times an ordered product of site operators.

    Attributes:
        coefficient: the scalar coefficient.
        operators: the factors, in order; the empty tuple denotes the identity.

    """

    coefficient: Scalar
    operators: tuple[SiteOperator, ...] = ()

    @property
    def support(self) -> frozenset[int]:
        """The register positions on which the term acts."""
        return frozenset(operator.site for operator in self.operators)

    @property
    def locality(self) -> int:
        """The number of distinct register positions on which the term acts."""
        return len(self.support)

    @property
    def is_constant(self) -> bool:
        """Whether the term is a multiple of the identity."""
        return not self.operators

    def adjoint(self) -> Term:
        """The Hermitian adjoint, with the factors reversed and each conjugated."""
        return Term(
            coefficient=_conjugate(self.coefficient),
            operators=tuple(operator.adjoint for operator in reversed(self.operators)),
        )

    def scaled(self, factor: ScalarLike) -> Term:
        """The term with its coefficient multiplied by ``factor``."""
        return Term(self.coefficient * as_scalar(factor), self.operators)

    def substitute(self, env: Mapping[str, ScalarLike]) -> Term:
        """The term with parameter values substituted into its coefficient."""
        return Term(self.coefficient.substitute(env), self.operators)

    def render(self, glyphs: Mapping[int, str] | None = None) -> str:
        """Render the term, optionally with a ladder-operator glyph per site."""
        glyphs = glyphs or {}
        rendered = " ".join(
            operator.render(glyphs.get(operator.site, "b")) for operator in self.operators
        )
        coefficient = str(self.coefficient)
        if not rendered:
            return coefficient
        if coefficient == "1":
            return rendered
        return f"{coefficient} {rendered}"

    def __str__(self) -> str:
        return self.render()


def identity_term(coefficient: ScalarLike) -> Term:
    """A purely additive constant."""
    return Term(as_scalar(coefficient), ())


def _conjugate(expression: Scalar) -> Scalar:
    """The complex conjugate of a coefficient expression.

    Numeric literals are conjugated; symbols are taken to be real and remain unchanged.
    """
    if isinstance(expression, Const):
        return Const(complex(expression.value).conjugate())
    if isinstance(expression, Add):
        return Add(tuple(_conjugate(term) for term in expression.terms))
    if isinstance(expression, Mul):
        return Mul(tuple(_conjugate(factor) for factor in expression.factors))
    if isinstance(expression, Pow):
        return Pow(_conjugate(expression.base), expression.exponent)
    return expression


@dataclass(frozen=True)
class OperatorSum:
    """A symbolic sum of operator products; the representation of a Hamiltonian.

    Attributes:
        terms: the terms, in the order written.
        name: a display name.

    """

    terms: tuple[Term, ...] = ()
    name: str = ""

    # -- algebra -----------------------------------------------------------------

    def __add__(self, other: OperatorSum) -> OperatorSum:
        return OperatorSum(self.terms + other.terms, self.name)

    def __mul__(self, factor: ScalarLike) -> OperatorSum:
        return OperatorSum(tuple(term.scaled(factor) for term in self.terms), self.name)

    def __rmul__(self, factor: ScalarLike) -> OperatorSum:
        return self.__mul__(factor)

    def __neg__(self) -> OperatorSum:
        return self.__mul__(-1)

    def __sub__(self, other: OperatorSum) -> OperatorSum:
        return self + (-other)

    def adjoint(self) -> OperatorSum:
        """The Hermitian adjoint of the whole sum."""
        return OperatorSum(tuple(term.adjoint() for term in self.terms), self.name)

    def plus_adjoint(self) -> OperatorSum:
        """``self + self.adjoint()``, the ``+ h.c.`` of the printed Hamiltonians."""
        return self + self.adjoint()

    def substitute(self, env: Mapping[str, ScalarLike]) -> OperatorSum:
        """Substitute parameter values into every coefficient."""
        return OperatorSum(tuple(term.substitute(env) for term in self.terms), self.name)

    def renamed(self, name: str) -> OperatorSum:
        """A copy carrying a different display name."""
        return OperatorSum(self.terms, name)

    # -- structure ---------------------------------------------------------------

    @property
    def support(self) -> frozenset[int]:
        """Every register position on which some term acts."""
        if not self.terms:
            return frozenset()
        return frozenset().union(*(term.support for term in self.terms))

    @property
    def max_locality(self) -> int:
        """The largest number of positions on which a single term acts."""
        return max((term.locality for term in self.terms), default=0)

    @property
    def parameters(self) -> frozenset[str]:
        """The free parameter names occurring in the coefficients."""
        if not self.terms:
            return frozenset()
        return frozenset().union(*(term.coefficient.symbols() for term in self.terms))

    def constant_part(self) -> Scalar:
        """The sum of the coefficients of the identity terms."""
        return summation(term.coefficient for term in self.terms if term.is_constant)

    def without_constants(self) -> OperatorSum:
        """A copy with the additive constants removed."""
        return OperatorSum(tuple(t for t in self.terms if not t.is_constant), self.name)

    def collected(self) -> OperatorSum:
        """A copy with terms of identical operator tuples merged, in order of first occurrence.

        Terms are merged by exact identity of the operator tuple; ``b_0 b_1`` and ``b_1 b_0``
        remain distinct.
        """
        buckets: dict[tuple[SiteOperator, ...], list[Scalar]] = {}
        for term in self.terms:
            buckets.setdefault(term.operators, []).append(term.coefficient)
        merged = tuple(
            Term(summation(coefficients), operators) for operators, coefficients in buckets.items()
        )
        return OperatorSum(merged, self.name)

    def validate_against(self, algebra_at: Mapping[int, Algebra]) -> None:
        """Check that every factor is admissible on the algebra of the site on which it acts.

        Args:
            algebra_at: the algebra of each register position.

        Raises:
            ValueError: if a factor acts on an undeclared site, or on a site whose algebra
                does not admit that operator.

        """
        for term in self.terms:
            for operator in term.operators:
                algebra = algebra_at.get(operator.site)
                if algebra is None:
                    msg = f"{operator} acts on register site {operator.site}, which is not declared"
                    raise ValueError(msg)
                if operator.symbol not in ALLOWED_OPERATORS[algebra]:
                    msg = (
                        f"{operator} is not a legal operator on a {algebra.value} site "
                        f"(site {operator.site})"
                    )
                    raise ValueError(msg)

    # -- rendering ---------------------------------------------------------------

    def pretty(self, glyphs: Mapping[int, str] | None = None) -> str:
        """A multi-line rendering, one term per line.

        Args:
            glyphs: the ladder-operator letter per register position, for instance the Greek
                letter psi on fermionic sites and ``"b"`` on bosonic ones.

        Returns:
            The rendered Hamiltonian.

        """
        if not self.terms:
            return f"{self.name or 'H'} = 0"
        lines = [
            f"  {'+' if index else ' '} {term.render(glyphs)}"
            for index, term in enumerate(self.terms)
        ]
        return f"{self.name or 'H'} =\n" + "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty()

    def __repr__(self) -> str:
        return f"OperatorSum(name={self.name!r}, terms={len(self.terms)})"

    def __len__(self) -> int:
        return len(self.terms)


def word(coefficient: ScalarLike, *operators: SiteOperator) -> OperatorSum:
    """A one-term operator sum: ``coefficient`` times the given ordered product."""
    return OperatorSum((Term(as_scalar(coefficient), operators),))


ZERO = OperatorSum(())
"""The empty sum."""


def constant(coefficient: ScalarLike) -> OperatorSum:
    """A purely additive constant, as a one-term operator sum."""
    return OperatorSum((identity_term(coefficient),))
