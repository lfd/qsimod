"""Numerical realisation inside a declared sector, without constructing the full space.

[`SectorBasis`][qsimod.realise.sector.SectorBasis] enumerates the configurations admitted by a
[`LocalSubspace`][qsimod.artifact.LocalSubspace] and a family of diagonal
[`ConstraintOperator`][qsimod.artifact.ConstraintOperator]s by a pruned depth-first search, and
[`build_operator_in`][qsimod.realise.sector.build_operator_in] assembles ``P H P`` in that basis
by lookup.  Terms that leave the sector are dropped.  The realisation is therefore exact for a
model that is confined to the sector and lossy for a model whose leakage out of the sector is
the observable; the latter is realised densely.  Fermionic degrees of freedom are rejected: a
Jordan-Wigner string is a property of the ordering and not a per-configuration amplitude.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from qsimod.artifact import ConstraintOperator, LocalSubspace
from qsimod.jax_setup import COMPLEX_DTYPE, require_x64
from qsimod.realise.hilbert import HilbertSpace, RealisationRequest, local_matrix
from qsimod.structure import Algebra
from qsimod.symbolic import OperatorSum, OpSymbol, Term

if TYPE_CHECKING:  # pragma: no cover -- imported for typing only
    from jax import Array

    from qsimod.structure import StructureType

__all__ = [
    "SECTOR_TOLERANCE",
    "SectorBasis",
    "build_operator_in",
]

#: The largest distance between the value of a constraint and its declared target for which a
#: configuration is counted as inside the sector.
SECTOR_TOLERANCE = 1e-9


def _amplitude_of(matrix: Array, column: int) -> tuple[int, complex] | None:
    """The single nonzero entry of one column as ``(row, amplitude)``, or ``None`` if there is none.

    Args:
        matrix: the local matrix of a single-site operator.
        column: the occupation on which the operator acts.

    Returns:
        The occupation reached and the amplitude, or ``None`` if the column vanishes.

    Raises:
        ValueError: if the column has more than one nonzero.

    """
    entries = np.asarray(matrix)[:, column]
    rows = np.flatnonzero(np.abs(entries) > SECTOR_TOLERANCE)
    if rows.size == 0:
        return None
    if rows.size > 1:
        msg = (
            f"a sector realisation needs each operator to map one occupation to one other, "
            f"but column {column} has {rows.size} nonzero entries"
        )
        raise ValueError(msg)
    row = int(rows[0])
    return row, complex(entries[row])


def _apply(
    term: Term,
    configuration: tuple[int, ...],
    space: HilbertSpace,
) -> tuple[tuple[int, ...], complex] | None:
    """Apply one term to one configuration.

    The operators of a [`Term`][qsimod.symbolic.Term] act on a ket, so the rightmost acts
    first.

    Args:
        term: the term; its coefficient is not applied.
        configuration: the occupations, in site order.
        space: the space against which the configuration is indexed.

    Returns:
        The configuration reached and the amplitude, or ``None`` if the term annihilates it.

    """
    amplitude = 1.0 + 0.0j
    current = list(configuration)
    for operator in reversed(term.operators):
        local = space.space_at(operator.site)
        position = space.position_of(operator.site)
        columns = _column_map(operator.symbol, local.algebra, local.dimension)
        reached = columns[current[position]]
        if reached is None:
            return None
        current[position], factor = reached
        amplitude *= factor
    return tuple(current), amplitude


@cache
def _column_map(
    symbol: OpSymbol, algebra: Algebra, dimension: int
) -> tuple[tuple[int, complex] | None, ...]:
    """The occupation and amplitude to which a primitive operator maps each occupation.

    The local matrix is constructed once per ``(symbol, algebra, dimension)`` and read column by
    column; the enumeration of a sector does not dispatch to JAX per configuration.

    Args:
        symbol: the primitive operator.
        algebra: the algebra of the site.
        dimension: the local dimension of the site.

    Returns:
        One entry per occupation: the occupation reached and the amplitude, or ``None`` if
        the operator annihilates it.

    """
    matrix = np.asarray(local_matrix(symbol, algebra, dimension))
    return tuple(_amplitude_of(matrix, column) for column in range(dimension))


def _diagonal_value(
    operator: OperatorSum,
    configuration: tuple[int, ...],
    space: HilbertSpace,
    environment: Mapping[str, float],
) -> float:
    """The eigenvalue of a diagonal operator on one configuration.

    Args:
        operator: the operator, which must be diagonal in the occupation basis.
        configuration: the occupations, in site order.
        space: the space against which the configuration is indexed.
        environment: values for the parameters of the coefficients.

    Returns:
        The eigenvalue.

    Raises:
        ValueError: if the operator maps the configuration to a different one, that is, if it
            is not diagonal.

    """
    total = 0.0 + 0.0j
    for term in operator.terms:
        coefficient = complex(term.coefficient.evaluate(environment))
        if term.is_constant:
            total += coefficient
            continue
        reached = _apply(term, configuration, space)
        if reached is None:
            continue
        landed, amplitude = reached
        if landed != configuration:
            msg = (
                "a sector constraint must be diagonal in the occupation basis so that a "
                f"partial configuration can be judged; {operator.name or 'this operator'} "
                "moves it"
            )
            raise ValueError(msg)
        total += coefficient * amplitude
    return float(total.real)


@dataclass(frozen=True)
class SectorBasis:
    """The configurations admitted by a declared subspace and a family of constraints.

    Attributes:
        space: the Hilbert space against which the configurations are indexed; it is never
            materialised as an array.
        configurations: the admitted configurations, in enumeration order.
        index: the position of each configuration in ``configurations``.

    """

    space: HilbertSpace
    configurations: tuple[tuple[int, ...], ...]
    index: Mapping[tuple[int, ...], int]

    @classmethod
    def of(
        cls,
        structure: StructureType,
        subspace: LocalSubspace | None = None,
        constraints: Sequence[ConstraintOperator] = (),
        request: RealisationRequest | None = None,
        environment: Mapping[str, float] | None = None,
    ) -> SectorBasis:
        """Enumerate the sector by depth-first search with pruning.

        Sites are assigned in order, and each constraint is checked as soon as the last site
        of its support is assigned.

        Args:
            structure: the structural type of the model.
            subspace: the declared per-site occupations; every occupation is admitted if
                omitted.
            constraints: the declared constraint operators, with their target eigenvalues.
            request: the cutoffs and orderings; defaults are used if omitted.
            environment: values for the coefficients of the constraints.

        Returns:
            The basis.

        Raises:
            ValueError: if any degree of freedom is fermionic.

        """
        space = HilbertSpace.of(structure, request)
        fermionic = [
            dof.role for dof in structure.degrees_of_freedom if dof.algebra is Algebra.FERMION
        ]
        if fermionic:
            msg = (
                f"a sector realisation cannot carry fermionic degree(s) of freedom {fermionic}:"
                " a Jordan-Wigner string is a property of the ordering, not of the local "
                "occupation, so it is not a per-configuration amplitude"
            )
            raise ValueError(msg)

        values = dict(environment or {})
        allowed = [
            tuple(
                (subspace.allowed_occupations.get(site) if subspace else None)
                or range(space.space_at(site).dimension)
            )
            for site in space.sites
        ]
        # A constraint can be judged once the last site it touches has been assigned.
        closing: dict[int, list[ConstraintOperator]] = {}
        for constraint in constraints:
            support = constraint.operator.support
            if not support:  # pragma: no cover -- a constraint always names sites
                continue
            closing.setdefault(space.position_of(max(support)), []).append(constraint)

        found: list[tuple[int, ...]] = []
        partial: list[int] = []

        def walk(position: int) -> None:
            if position == len(allowed):
                found.append(tuple(partial))
                return
            for occupation in allowed[position]:
                partial.append(occupation)
                # Padding sites lie beyond this constraint's support and cannot affect it.
                padded = tuple(partial) + (0,) * (len(allowed) - len(partial))
                if all(
                    abs(
                        _diagonal_value(constraint.operator, padded, space, values)
                        - constraint.target_value
                    )
                    <= SECTOR_TOLERANCE
                    for constraint in closing.get(position, ())
                ):
                    walk(position + 1)
                partial.pop()

        walk(0)
        return cls(
            space=space,
            configurations=tuple(found),
            index={configuration: where for where, configuration in enumerate(found)},
        )

    @property
    def dimension(self) -> int:
        """The number of configurations in the sector."""
        return len(self.configurations)

    def state(self, configuration: Sequence[int]) -> Array:
        """The basis vector of one configuration, in the ordering of the sector.

        Args:
            configuration: the occupations, in site order.

        Returns:
            A unit vector of length [`dimension`][qsimod.realise.sector.SectorBasis.dimension].

        Raises:
            KeyError: if the configuration is not in the sector.

        """
        require_x64()
        import jax.numpy as jnp  # noqa: PLC0415 -- deferred so the module imports cheaply

        where = self.index[tuple(configuration)]
        vector = np.zeros((self.dimension,), dtype=np.complex128)
        vector[where] = 1.0
        return jnp.asarray(vector, dtype=COMPLEX_DTYPE)

    def __str__(self) -> str:
        return (
            f"SectorBasis: {len(self.space.sites)} site(s), sector dimension "
            f"{self.dimension} of {self.space.dimension}"
        )


def build_operator_in(
    hamiltonian: OperatorSum,
    basis: SectorBasis,
    environment: Mapping[str, float],
) -> Array:
    """Realise a symbolic operator sum directly in the basis of a sector.

    The result equals ``restrict(sandwich(build_operator(...), projector), indices)`` and is
    constructed at the cost of the sector; terms that leave the sector are dropped.

    Args:
        hamiltonian: the symbolic sum.
        basis: the sector in which the operator is constructed.
        environment: values for every parameter occurring in the coefficients.

    Returns:
        A ``(dimension, dimension)`` ``complex128`` array over the sector.

    Raises:
        KeyError: if a coefficient refers to an unbound parameter.

    """
    require_x64()
    import jax.numpy as jnp  # noqa: PLC0415 -- deferred so the module imports cheaply

    matrix = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    for term in hamiltonian.terms:
        coefficient = complex(term.coefficient.evaluate(environment))
        if term.is_constant:
            matrix[np.diag_indices(basis.dimension)] += coefficient
            continue
        for column, configuration in enumerate(basis.configurations):
            reached = _apply(term, configuration, basis.space)
            if reached is None:
                continue
            landed, amplitude = reached
            row = basis.index.get(landed)
            if row is not None:
                matrix[row, column] += coefficient * amplitude
    return jnp.asarray(matrix, dtype=COMPLEX_DTYPE)
