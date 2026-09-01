"""A normal form for symbolic operator sums, and the equivalence of two sums.

Two [`OperatorSum`][qsimod.symbolic.OperatorSum] objects that denote the same operator may be
written differently: the factors in another order, ``psi^dag psi`` for ``n``, ``sigma^+`` for
``(X + iY)/2``, or the product ``(1 - 2n)(1 - 2n)`` for the identity.
[`normal_form`][qsimod.normal_form.normal_form] rewrites a sum into a canonical one: every term
is a product of at most one canonical operator per site, with the sites in ascending order,
the fermionic signs carried by the reordering, and same-site products reduced by the algebra
of the site.  Two sums are [`equivalent`][qsimod.normal_form.equivalent] when their normal
forms have the same terms with the same coefficients, the coefficients being compared as
expressions in the parameters.

The canonical single-site operators are ``c^dag, c, n`` on a fermion, ``X, Y, Z`` on a qubit,
``S^z, S^+, S^-`` on a spin-1/2, and the normal-ordered ``(a^dag)^p a^q`` on a boson; the parity
``P = 1 - 2n`` expands to ``1 - 2n`` on a fermion and to ``-Z`` on a qubit.  A gauge link is
left as written, except that ``U U^dag`` and ``U^dag U`` cancel.

No matrix is constructed; the algebra tables are those satisfied by the local matrices of the
numerical realisation ([`qsimod.realise.hilbert`][qsimod.realise.hilbert]).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence

from qsimod.scalar import Const, Scalar, ScalarLike, as_scalar, simplify, summation
from qsimod.structure import Algebra
from qsimod.symbolic import OperatorSum, OpSymbol, SiteOperator, Term

__all__ = [
    "DEFAULT_SAMPLE_SEED",
    "OperatorImage",
    "constant_word",
    "difference",
    "equivalent",
    "normal_form",
    "product",
    "rewrite",
    "sample_environments",
]

#: The seed of the deterministic parameter samples at which coefficient equality is decided.
DEFAULT_SAMPLE_SEED = 20260909

#: A per-site rewriting rule: the image of one site operator as a sum of operator words.
OperatorImage = Callable[[SiteOperator], OperatorSum]

#: A linear combination of canonical operators on one site; ``None`` is the identity.
_Local = dict[OpSymbol | None, complex]

_FERMION_ODD = frozenset({OpSymbol.CREATE, OpSymbol.ANNIHILATE})
_LINK_INVERSES = frozenset(
    {(OpSymbol.LINK_U, OpSymbol.LINK_U_DAGGER), (OpSymbol.LINK_U_DAGGER, OpSymbol.LINK_U)}
)

# -- the algebra tables ---------------------------------------------------------------

#: ``a * b`` for the two-level algebras, as a combination of canonical operators.  Pairs
#: absent from the table multiply to zero.
_TWO_LEVEL_PRODUCTS: dict[Algebra, dict[tuple[OpSymbol, OpSymbol], _Local]] = {
    Algebra.FERMION: {
        (OpSymbol.CREATE, OpSymbol.ANNIHILATE): {OpSymbol.NUMBER: 1},
        (OpSymbol.ANNIHILATE, OpSymbol.CREATE): {None: 1, OpSymbol.NUMBER: -1},
        (OpSymbol.NUMBER, OpSymbol.NUMBER): {OpSymbol.NUMBER: 1},
        (OpSymbol.NUMBER, OpSymbol.CREATE): {OpSymbol.CREATE: 1},
        (OpSymbol.ANNIHILATE, OpSymbol.NUMBER): {OpSymbol.ANNIHILATE: 1},
    },
    Algebra.SPIN_HALF: {
        (OpSymbol.SPIN_PLUS, OpSymbol.SPIN_MINUS): {None: 0.5, OpSymbol.SPIN_Z: 1},
        (OpSymbol.SPIN_MINUS, OpSymbol.SPIN_PLUS): {None: 0.5, OpSymbol.SPIN_Z: -1},
        (OpSymbol.SPIN_Z, OpSymbol.SPIN_Z): {None: 0.25},
        (OpSymbol.SPIN_Z, OpSymbol.SPIN_PLUS): {OpSymbol.SPIN_PLUS: 0.5},
        (OpSymbol.SPIN_PLUS, OpSymbol.SPIN_Z): {OpSymbol.SPIN_PLUS: -0.5},
        (OpSymbol.SPIN_Z, OpSymbol.SPIN_MINUS): {OpSymbol.SPIN_MINUS: -0.5},
        (OpSymbol.SPIN_MINUS, OpSymbol.SPIN_Z): {OpSymbol.SPIN_MINUS: 0.5},
    },
    Algebra.QUBIT: {
        (OpSymbol.PAULI_X, OpSymbol.PAULI_X): {None: 1},
        (OpSymbol.PAULI_Y, OpSymbol.PAULI_Y): {None: 1},
        (OpSymbol.PAULI_Z, OpSymbol.PAULI_Z): {None: 1},
        (OpSymbol.PAULI_X, OpSymbol.PAULI_Y): {OpSymbol.PAULI_Z: 1j},
        (OpSymbol.PAULI_Y, OpSymbol.PAULI_X): {OpSymbol.PAULI_Z: -1j},
        (OpSymbol.PAULI_Y, OpSymbol.PAULI_Z): {OpSymbol.PAULI_X: 1j},
        (OpSymbol.PAULI_Z, OpSymbol.PAULI_Y): {OpSymbol.PAULI_X: -1j},
        (OpSymbol.PAULI_Z, OpSymbol.PAULI_X): {OpSymbol.PAULI_Y: 1j},
        (OpSymbol.PAULI_X, OpSymbol.PAULI_Z): {OpSymbol.PAULI_Y: -1j},
    },
}

#: Non-canonical qubit operators in the Pauli basis: ``sigma^+ = (X + iY)/2``,
#: ``sigma^- = (X - iY)/2``, ``n = (1 + Z)/2`` (``Z = 2n - 1`` in the occupation basis).
_QUBIT_BASIS: dict[OpSymbol, _Local] = {
    OpSymbol.SIGMA_PLUS: {OpSymbol.PAULI_X: 0.5, OpSymbol.PAULI_Y: 0.5j},
    OpSymbol.SIGMA_MINUS: {OpSymbol.PAULI_X: 0.5, OpSymbol.PAULI_Y: -0.5j},
    OpSymbol.NUMBER: {None: 0.5, OpSymbol.PAULI_Z: 0.5},
    OpSymbol.PARITY: {OpSymbol.PAULI_Z: -1},
}

#: The non-canonical fermionic operator: the parity ``P = 1 - 2n``.
_FERMION_BASIS: dict[OpSymbol, _Local] = {
    OpSymbol.PARITY: {None: 1, OpSymbol.NUMBER: -2},
}

_NON_CANONICAL: dict[Algebra, dict[OpSymbol, _Local]] = {
    Algebra.QUBIT: _QUBIT_BASIS,
    Algebra.FERMION: _FERMION_BASIS,
}


def _to_canonical(operator: OpSymbol, algebra: Algebra) -> _Local:
    """One primitive operator as a combination of the algebra's canonical operators."""
    basis = _NON_CANONICAL.get(algebra, {})
    if operator in basis:
        return dict(basis[operator])
    return {operator: 1}


def _multiply_two_level(left: _Local, right: _Local, algebra: Algebra) -> _Local:
    """The product of two combinations on one two-level site."""
    table = _TWO_LEVEL_PRODUCTS[algebra]
    result: _Local = {}
    for a, x in left.items():
        for b, y in right.items():
            if a is None:
                _accumulate(result, {b: x * y})
            elif b is None:
                _accumulate(result, {a: x * y})
            else:
                for c, z in table.get((a, b), {}).items():
                    _accumulate(result, {c: x * y * z})
    return result


def _accumulate(into: _Local, more: _Local) -> None:
    for key, value in more.items():
        into[key] = into.get(key, 0) + value


def _reduce_two_level(operators: Sequence[OpSymbol], algebra: Algebra) -> _Local:
    """A same-site product of primitive operators as a combination of canonical ones."""
    result: _Local = {None: 1}
    for operator in operators:
        result = _multiply_two_level(result, _to_canonical(operator, algebra), algebra)
    return {key: value for key, value in result.items() if value != 0}


def _normal_order_bosons(operators: Sequence[OpSymbol]) -> dict[tuple[OpSymbol, ...], complex]:
    """A same-site bosonic product as a combination of normal-ordered words.

    ``n`` is written as ``a^dag a``, and every ``a a^dag`` is replaced by ``a^dag a + 1``.
    """
    expanded: list[OpSymbol] = []
    for operator in operators:
        if operator is OpSymbol.NUMBER:
            expanded += [OpSymbol.CREATE, OpSymbol.ANNIHILATE]
        else:
            expanded.append(operator)
    pending: dict[tuple[OpSymbol, ...], complex] = {tuple(expanded): 1}
    finished: dict[tuple[OpSymbol, ...], complex] = {}
    while pending:
        word, weight = pending.popitem()
        for index in range(len(word) - 1):
            if word[index] is OpSymbol.ANNIHILATE and word[index + 1] is OpSymbol.CREATE:
                swapped = (*word[:index], OpSymbol.CREATE, OpSymbol.ANNIHILATE, *word[index + 2 :])
                contracted = (*word[:index], *word[index + 2 :])
                pending[swapped] = pending.get(swapped, 0) + weight
                pending[contracted] = pending.get(contracted, 0) + weight
                break
        else:
            finished[word] = finished.get(word, 0) + weight
    return {word: weight for word, weight in finished.items() if weight != 0}


def _cancel_link_inverses(operators: Sequence[OpSymbol]) -> tuple[OpSymbol, ...]:
    """Cancel adjacent ``U U^dag`` and ``U^dag U`` pairs on a gauge link."""
    stack: list[OpSymbol] = []
    for operator in operators:
        if stack and (stack[-1], operator) in _LINK_INVERSES:
            stack.pop()
        else:
            stack.append(operator)
    return tuple(stack)


def _reduce_site(
    site: int, operators: Sequence[OpSymbol], algebra: Algebra
) -> dict[tuple[SiteOperator, ...], complex]:
    """A same-site product as a combination of canonical words on that site."""
    if algebra in _TWO_LEVEL_PRODUCTS:
        return {
            (() if symbol is None else (SiteOperator(symbol, site),)): weight
            for symbol, weight in _reduce_two_level(operators, algebra).items()
        }
    if algebra is Algebra.BOSON:
        return {
            tuple(SiteOperator(symbol, site) for symbol in word): weight
            for word, weight in _normal_order_bosons(operators).items()
        }
    return {tuple(SiteOperator(s, site) for s in _cancel_link_inverses(operators)): 1}


# -- reordering ------------------------------------------------------------------------


def _sorted_by_site(
    operators: Sequence[SiteOperator], algebra_at: Mapping[int, Algebra]
) -> tuple[tuple[SiteOperator, ...], int]:
    """A stable sort by site, with the fermionic sign of the reordering.

    Two factors on different sites commute unless both are fermionic ladder operators, which
    anticommute.  Factors on the same site keep their order.
    """

    def is_odd(operator: SiteOperator) -> bool:
        return algebra_at.get(operator.site) is Algebra.FERMION and operator.symbol in _FERMION_ODD

    ordered = list(operators)
    sign = 1
    for index in range(1, len(ordered)):
        position = index
        while position > 0 and ordered[position - 1].site > ordered[position].site:
            if is_odd(ordered[position - 1]) and is_odd(ordered[position]):
                sign = -sign
            ordered[position - 1], ordered[position] = ordered[position], ordered[position - 1]
            position -= 1
    return tuple(ordered), sign


def _normal_terms(term: Term, algebra_at: Mapping[int, Algebra]) -> list[Term]:
    """The normal form of one term: canonical words, each with its coefficient."""
    for operator in term.operators:
        if operator.site not in algebra_at:
            msg = f"{operator} acts on register site {operator.site}, which is not declared"
            raise ValueError(msg)
    ordered, sign = _sorted_by_site(term.operators, algebra_at)
    # Group the ordered factors by site and reduce each group.
    groups: list[tuple[int, list[OpSymbol]]] = []
    for operator in ordered:
        if groups and groups[-1][0] == operator.site:
            groups[-1][1].append(operator.symbol)
        else:
            groups.append((operator.site, [operator.symbol]))
    words: dict[tuple[SiteOperator, ...], complex] = {(): sign}
    for site, symbols in groups:
        reduced = _reduce_site(site, symbols, algebra_at[site])
        words = {
            prefix + suffix: weight * more
            for prefix, weight in words.items()
            for suffix, more in reduced.items()
        }
    return [
        Term(term.coefficient * _weight(weight), word)
        for word, weight in words.items()
        if weight != 0
    ]


def _weight(value: complex) -> Scalar:
    """A numeric weight as a scalar constant, real where it is real."""
    return Const(value.real if value.imag == 0 else value)


# -- the public functions --------------------------------------------------------------


def normal_form(operator: OperatorSum, algebra_at: Mapping[int, Algebra]) -> OperatorSum:
    """The canonical form of an operator sum.

    Args:
        operator: the sum to normalise.
        algebra_at: the algebra of every register site the sum may act on.

    Returns:
        A sum with one term per distinct canonical word, the sites in ascending order within
        each word, the coefficients simplified, and terms whose coefficient folds to zero
        removed.  The identity term, if present, is first; the others follow in order of
        first occurrence.

    Raises:
        ValueError: if a factor acts on a site ``algebra_at`` does not declare.

    """
    buckets: dict[tuple[SiteOperator, ...], list[Scalar]] = {}
    for term in operator.terms:
        for normal in _normal_terms(term, algebra_at):
            buckets.setdefault(normal.operators, []).append(normal.coefficient)
    terms: list[Term] = []
    for word, coefficients in sorted(buckets.items(), key=lambda item: item[0] != ()):
        coefficient = simplify(summation(coefficients))
        if isinstance(coefficient, Const) and coefficient.value == 0:
            continue
        terms.append(Term(coefficient, word))
    return OperatorSum(tuple(terms), operator.name)


def product(left: OperatorSum, right: OperatorSum) -> OperatorSum:
    """The operator product ``left * right``: every pair of words concatenated, in order."""
    return OperatorSum(
        tuple(
            Term(a.coefficient * b.coefficient, a.operators + b.operators)
            for a in left.terms
            for b in right.terms
        ),
        left.name,
    )


def rewrite(operator: OperatorSum, image: OperatorImage) -> OperatorSum:
    """Replace every site operator by its image, multiplying the images out in order.

    The rule is applied factor by factor, so that it must be an algebra homomorphism on the
    factors it is applied to: the image of a product is the product of the images.  A rule may
    map one factor to a sum, for instance ``n -> 1 - n``, or to a word on several sites, for
    instance a Jordan-Wigner string.

    Args:
        operator: the sum to rewrite.
        image: the per-factor rule.

    Returns:
        The rewritten sum, not normalised.

    """
    terms: list[Term] = []
    for term in operator.terms:
        current = OperatorSum((Term(term.coefficient, ()),))
        for factor in term.operators:
            current = product(current, image(factor))
        terms.extend(current.terms)
    return OperatorSum(tuple(terms), operator.name)


def sample_environments(
    symbols: Iterable[str],
    count: int = 3,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[dict[str, float], ...]:
    """Deterministic parameter points at which coefficient expressions are compared.

    The values lie in ``[0.5, 1.5]``, away from zero, so that a ratio of parameters is finite.
    """
    names = sorted(set(symbols))
    generator = random.Random(seed)  # noqa: S311 - reproducible samples, not security
    return tuple({name: generator.uniform(0.5, 1.5) for name in names} for _ in range(count))


def difference(
    left: OperatorSum,
    right: OperatorSum,
    algebra_at: Mapping[int, Algebra],
    *,
    tolerance: float = 1e-9,
    samples: Sequence[Mapping[str, float]] | None = None,
) -> OperatorSum:
    """``left - right`` in normal form, with the terms whose coefficients vanish removed.

    A coefficient vanishes when it folds to the constant zero, or when it evaluates to at most
    ``tolerance`` times the largest coefficient magnitude of either operand at every sample
    point.

    Args:
        left: one operand.
        right: the other.
        algebra_at: the algebra of every register site either may act on.
        tolerance: the relative tolerance on a coefficient.
        samples: the parameter points; defaults to
            [`sample_environments`][qsimod.normal_form.sample_environments] over both operands'
            parameters.

    Returns:
        The surviving terms of the difference; empty exactly when the operands are equivalent.

    """
    normal_left = normal_form(left, algebra_at)
    normal_right = normal_form(right, algebra_at)
    delta = normal_form(normal_left - normal_right, algebra_at)
    if not delta.terms:
        return delta
    points = sample_environments(left.parameters | right.parameters) if samples is None else samples
    scales = [
        max(
            1.0,
            *(abs(complex(t.coefficient.evaluate(point))) for t in normal_left.terms),
            *(abs(complex(t.coefficient.evaluate(point))) for t in normal_right.terms),
        )
        for point in points
    ]
    kept: list[Term] = []
    for term in delta.terms:
        for point, scale in zip(points, scales, strict=True):
            if abs(complex(term.coefficient.evaluate(point))) > tolerance * scale:
                kept.append(term)
                break
    return OperatorSum(tuple(kept), f"{left.name} - {right.name}".strip(" -"))


def equivalent(
    left: OperatorSum,
    right: OperatorSum,
    algebra_at: Mapping[int, Algebra],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Whether two sums denote the same operator on the given algebras."""
    return not difference(left, right, algebra_at, tolerance=tolerance).terms


def constant_word(coefficient: ScalarLike) -> OperatorSum:
    """A one-term identity sum, used to build images."""
    return OperatorSum((Term(as_scalar(coefficient), ()),))
