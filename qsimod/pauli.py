"""Pauli strings and Pauli sums: exact Pauli algebra without operator construction.

Coefficients are complex numbers; the Pauli algebra is entered after the parameters have been
bound.  Two Pauli strings commute if and only if they differ in an even number of tensor
positions, the product of two strings is a string times a phase, and the commutator of two Pauli
sums is a Pauli sum.  No Hilbert space is constructed.

The single-qubit conventions are shared with [`qsimod.realise.hilbert`][qsimod.realise.hilbert]:

* the local basis index is the occupation number, ``|0>`` is empty and ``|1>`` is occupied;
* ``Z = 2n - 1 = diag(-1, +1)``, ``X = [[0,1],[1,0]]``, ``Y = [[0,i],[-i,0]]``;
* hence ``n = (Z + 1)/2``, ``sigma^+ = |1><0| = (X + iY)/2``, ``sigma^- = (X - iY)/2`` and
  ``XY = iZ``; the axes are right-handed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from qsimod.structure import Algebra, StructureType
from qsimod.symbolic import OperatorSum, OpSymbol, SiteOperator

__all__ = [
    "COEFFICIENT_TOLERANCE",
    "IDENTITY_STRING",
    "PAULI_ALGEBRAS",
    "PauliAxis",
    "PauliString",
    "PauliSum",
    "commutator",
    "nested_commutator",
    "pauli_sum_from_operator",
]


class PauliAxis(Enum):
    """One of the three Pauli axes."""

    X = "X"
    Y = "Y"
    Z = "Z"

    def __str__(self) -> str:
        return self.value


#: The single-qubit product table ``axis_left * axis_right -> (resulting axis, phase)``, where
#: ``None`` in place of an axis denotes the identity.  The axes are right-handed: ``XY = iZ``,
#: ``YZ = iX``, ``ZX = iY``.
_PRODUCT: dict[tuple[PauliAxis, PauliAxis], tuple[PauliAxis | None, complex]] = {
    (PauliAxis.X, PauliAxis.X): (None, 1),
    (PauliAxis.Y, PauliAxis.Y): (None, 1),
    (PauliAxis.Z, PauliAxis.Z): (None, 1),
    (PauliAxis.X, PauliAxis.Y): (PauliAxis.Z, 1j),
    (PauliAxis.Y, PauliAxis.X): (PauliAxis.Z, -1j),
    (PauliAxis.Y, PauliAxis.Z): (PauliAxis.X, 1j),
    (PauliAxis.Z, PauliAxis.Y): (PauliAxis.X, -1j),
    (PauliAxis.Z, PauliAxis.X): (PauliAxis.Y, 1j),
    (PauliAxis.X, PauliAxis.Z): (PauliAxis.Y, -1j),
}


@dataclass(frozen=True)
class PauliString:
    """A tensor product of Pauli operators that acts as the identity on every unlisted qubit.

    Attributes:
        axes: the ``(qubit, axis)`` pairs on which the string acts non-trivially, ascending by
            qubit.  The empty tuple is the identity.

    """

    axes: tuple[tuple[int, PauliAxis], ...] = ()

    def __post_init__(self) -> None:
        qubits = [qubit for qubit, _ in self.axes]
        if qubits != sorted(qubits) or len(qubits) != len(set(qubits)):
            msg = f"Pauli string axes must be one per qubit, ascending; got {self.axes}"
            raise ValueError(msg)

    @classmethod
    def of(cls, axes: Mapping[int, PauliAxis]) -> PauliString:
        """Construct a string from a ``{qubit: axis}`` mapping."""
        return cls(tuple(sorted(axes.items())))

    @classmethod
    def single(cls, qubit: int, axis: PauliAxis) -> PauliString:
        """The string that acts with ``axis`` on a single qubit."""
        return cls(((qubit, axis),))

    @property
    def mapping(self) -> dict[int, PauliAxis]:
        """The string as a ``{qubit: axis}`` mapping."""
        return dict(self.axes)

    @property
    def support(self) -> frozenset[int]:
        """The qubits on which the string acts non-trivially."""
        return frozenset(qubit for qubit, _ in self.axes)

    @property
    def weight(self) -> int:
        """The locality of the string, that is, the number of qubits on which it acts."""
        return len(self.axes)

    @property
    def is_identity(self) -> bool:
        """Whether the string is the identity."""
        return not self.axes

    @property
    def sort_key(self) -> tuple[int, tuple[tuple[int, str], ...]]:
        """A total order on strings, by weight, then by qubit and axis letter."""
        return (self.weight, tuple((qubit, axis.value) for qubit, axis in self.axes))

    def commutes_with(self, other: PauliString) -> bool:
        """Whether the two strings commute.

        Two strings anticommute on every position where both act with different axes; they
        commute if and only if the number of such positions is even.
        """
        mine = self.mapping
        disagreements = sum(
            1 for qubit, axis in other.axes if qubit in mine and mine[qubit] is not axis
        )
        return disagreements % 2 == 0

    def times(self, other: PauliString) -> tuple[PauliString, complex]:
        """The product ``self * other``, as a string and a phase."""
        result: dict[int, PauliAxis] = self.mapping
        phase: complex = 1
        for qubit, axis in other.axes:
            existing = result.get(qubit)
            if existing is None:
                result[qubit] = axis
                continue
            combined, factor = _PRODUCT[(existing, axis)]
            phase *= factor
            if combined is None:
                del result[qubit]
            else:
                result[qubit] = combined
        return PauliString.of(result), phase

    def __str__(self) -> str:
        if not self.axes:
            return "I"
        return " ".join(f"{axis}{qubit}" for qubit, axis in self.axes)


IDENTITY_STRING = PauliString()
"""The identity Pauli string."""

#: Coefficients below this magnitude are dropped when a Pauli sum is assembled.
COEFFICIENT_TOLERANCE = 1e-14


@dataclass(frozen=True)
class PauliSum:
    """A real or complex linear combination of Pauli strings.

    Attributes:
        terms: the coefficient of each string.  Strings with a coefficient below
            `COEFFICIENT_TOLERANCE` are dropped on construction.
        name: a display name, used in reports.

    """

    terms: Mapping[PauliString, complex]
    name: str = ""

    @classmethod
    def from_terms(
        cls,
        terms: Iterable[tuple[PauliString, complex]],
        name: str = "",
    ) -> PauliSum:
        """Construct a sum; repeated strings are merged and cancelled strings are dropped."""
        merged: dict[PauliString, complex] = {}
        for string, coefficient in terms:
            merged[string] = merged.get(string, 0j) + coefficient
        pruned = {
            string: coefficient
            for string, coefficient in merged.items()
            if abs(coefficient) > COEFFICIENT_TOLERANCE
        }
        return cls(pruned, name)

    @classmethod
    def zero(cls, name: str = "") -> PauliSum:
        """The empty sum."""
        return cls({}, name)

    # -- algebra -----------------------------------------------------------------

    def __add__(self, other: PauliSum) -> PauliSum:
        return PauliSum.from_terms([*self.terms.items(), *other.terms.items()], self.name)

    def __sub__(self, other: PauliSum) -> PauliSum:
        return self + other.scaled(-1)

    def scaled(self, factor: complex) -> PauliSum:
        """The sum with every coefficient multiplied by ``factor``."""
        return PauliSum.from_terms(
            [(string, coefficient * factor) for string, coefficient in self.terms.items()],
            self.name,
        )

    def times(self, other: PauliSum) -> PauliSum:
        """The operator product ``self * other``."""
        products: list[tuple[PauliString, complex]] = []
        for left, left_coefficient in self.terms.items():
            for right, right_coefficient in other.terms.items():
                string, phase = left.times(right)
                products.append((string, left_coefficient * right_coefficient * phase))
        return PauliSum.from_terms(products)

    def adjoint(self) -> PauliSum:
        """The Hermitian adjoint.

        Pauli strings are self-adjoint, so the coefficients are conjugated.
        """
        return PauliSum.from_terms(
            [(string, coefficient.conjugate()) for string, coefficient in self.terms.items()],
            self.name,
        )

    def renamed(self, name: str) -> PauliSum:
        """A copy carrying a different display name."""
        return PauliSum(self.terms, name)

    # -- structure ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.terms)

    def __bool__(self) -> bool:
        return bool(self.terms)

    @property
    def is_zero(self) -> bool:
        """Whether the sum is empty."""
        return not self.terms

    @property
    def strings(self) -> tuple[PauliString, ...]:
        """The strings present, in `PauliString.sort_key` order."""
        return tuple(sorted(self.terms, key=lambda string: string.sort_key))

    @property
    def support(self) -> frozenset[int]:
        """The union of the supports of all strings."""
        return frozenset().union(*(s.support for s in self.terms)) if self.terms else frozenset()

    @property
    def max_weight(self) -> int:
        """The largest locality of any string."""
        return max((string.weight for string in self.terms), default=0)

    @property
    def one_norm(self) -> float:
        """The one-norm ``sum |coefficient|``, an upper bound on the spectral norm.

        Every Pauli string has spectral norm one, so ``||sum c_P P|| <= sum |c_P|``.
        """
        return float(sum(abs(coefficient) for coefficient in self.terms.values()))

    def is_hermitian(self, tolerance: float = 1e-12) -> bool:
        """Whether every coefficient is real to within ``tolerance``."""
        return all(abs(c.imag) <= tolerance for c in self.terms.values())

    def internally_commuting(self) -> bool:
        """Whether every pair of strings in the sum commutes."""
        strings = self.strings
        return all(
            strings[i].commutes_with(strings[j])
            for i in range(len(strings))
            for j in range(i + 1, len(strings))
        )

    def commutes_with(self, other: PauliSum) -> bool:
        """Whether ``[self, other] = 0`` exactly, decided by computing the commutator."""
        return commutator(self, other).is_zero

    def identity_coefficient(self) -> complex:
        """The coefficient of the identity string, that is, the additive constant."""
        return self.terms.get(IDENTITY_STRING, 0j)

    def weight_histogram(self) -> dict[int, int]:
        """The number of strings of each locality."""
        histogram: dict[int, int] = {}
        for string in self.terms:
            histogram[string.weight] = histogram.get(string.weight, 0) + 1
        return dict(sorted(histogram.items()))

    def __str__(self) -> str:
        if not self.terms:
            return f"{self.name or 'P'} = 0"
        lines = [
            f"  {coefficient.real:+.6g}{coefficient.imag:+.6g}j  {string}"
            if abs(coefficient.imag) > COEFFICIENT_TOLERANCE
            else f"  {coefficient.real:+.6g}  {string}"
            for string, coefficient in ((s, self.terms[s]) for s in self.strings)
        ]
        return f"{self.name or 'P'} =\n" + "\n".join(lines)

    def __repr__(self) -> str:
        return f"PauliSum(name={self.name!r}, strings={len(self.terms)})"


def commutator(left: PauliSum, right: PauliSum) -> PauliSum:
    """The commutator ``[left, right] = left*right - right*left``, exactly."""
    return left.times(right) - right.times(left)


def nested_commutator(operands: list[PauliSum]) -> PauliSum:
    """The nested commutator ``[operands[-1], [..., [operands[1], operands[0]]]]``.

    The brackets are evaluated from the innermost outwards.

    Raises:
        ValueError: if fewer than two operands are given.

    """
    if len(operands) < 2:
        msg = f"a nested commutator needs at least two operands, got {len(operands)}"
        raise ValueError(msg)
    current = commutator(operands[1], operands[0])
    for operand in operands[2:]:
        current = commutator(operand, current)
    return current


# ---------------------------------------------------------------------------
# Conversion from the symbolic layer
# ---------------------------------------------------------------------------

#: The Pauli expansion of each primitive single-site operator on a two-level site.
#: ``None`` in place of an axis denotes the identity string, that is, an additive constant.
_SINGLE_SITE: dict[OpSymbol, tuple[tuple[PauliAxis | None, complex], ...]] = {
    OpSymbol.PAULI_X: ((PauliAxis.X, 1.0),),
    OpSymbol.PAULI_Y: ((PauliAxis.Y, 1.0),),
    OpSymbol.PAULI_Z: ((PauliAxis.Z, 1.0),),
    OpSymbol.SIGMA_PLUS: ((PauliAxis.X, 0.5), (PauliAxis.Y, 0.5j)),
    OpSymbol.SIGMA_MINUS: ((PauliAxis.X, 0.5), (PauliAxis.Y, -0.5j)),
    OpSymbol.NUMBER: ((PauliAxis.Z, 0.5), (None, 0.5)),
    OpSymbol.PARITY: ((PauliAxis.Z, -1.0),),
    # Spin-1/2: S^z = Z/2, S^+- = sigma^+-.
    OpSymbol.SPIN_Z: ((PauliAxis.Z, 0.5),),
    OpSymbol.SPIN_PLUS: ((PauliAxis.X, 0.5), (PauliAxis.Y, 0.5j)),
    OpSymbol.SPIN_MINUS: ((PauliAxis.X, 0.5), (PauliAxis.Y, -0.5j)),
}

#: The algebras whose sites are two-level and therefore admit a Pauli expansion.
PAULI_ALGEBRAS = frozenset({Algebra.QUBIT, Algebra.SPIN_HALF})


def _site_operator_to_pauli(operator: SiteOperator) -> PauliSum:
    """The Pauli expansion of one primitive site operator.

    Raises:
        ValueError: if the operator has no two-level Pauli expansion.

    """
    expansion = _SINGLE_SITE.get(operator.symbol)
    if expansion is None:
        msg = (
            f"{operator} has no Pauli expansion: {operator.symbol.value} is not an "
            "operator on a two-level site"
        )
        raise ValueError(msg)
    return PauliSum.from_terms(
        [
            (
                IDENTITY_STRING if axis is None else PauliString.single(operator.site, axis),
                coefficient,
            )
            for axis, coefficient in expansion
        ]
    )


def pauli_sum_from_operator(
    hamiltonian: OperatorSum,
    environment: Mapping[str, float],
    structure: StructureType | None = None,
    name: str = "",
) -> PauliSum:
    """Expand a bound symbolic Hamiltonian into a Pauli sum.

    Args:
        hamiltonian: the symbolic sum; its coefficients are evaluated at ``environment``.
        environment: parameter values.
        structure: if given, every site in the support of the Hamiltonian is checked to be
            two-level.
        name: a display name for the result.

    Returns:
        The Pauli sum.

    Raises:
        ValueError: if the model has a site whose algebra is not two-level, or an
            operator with no Pauli expansion.

    """
    if structure is not None:
        offenders = sorted(
            site for site in hamiltonian.support if structure.algebra_at(site) not in PAULI_ALGEBRAS
        )
        if offenders:
            algebras = {structure.algebra_at(site).value for site in offenders}
            msg = (
                f"cannot expand {hamiltonian.name or 'the Hamiltonian'} into Pauli strings: "
                f"register site(s) {offenders} have algebra {sorted(algebras)}, which is not "
                "two-level"
            )
            raise ValueError(msg)

    total = PauliSum.zero(name)
    for term in hamiltonian.terms:
        coefficient = complex(term.coefficient.evaluate(environment))
        product = PauliSum.from_terms([(IDENTITY_STRING, coefficient)])
        for operator in term.operators:
            product = product.times(_site_operator_to_pauli(operator))
        total = total + product
    return total.renamed(name or hamiltonian.name)
