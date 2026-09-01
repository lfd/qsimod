"""Construction of dense JAX operators from symbolic models and Pauli sums.

Jordan-Wigner convention: a fermionic ladder operator at Jordan-Wigner position ``p`` is
realised as

``psi_l = (-1)**p * (prod_{q < p} P_q) * sigma^-_l``,   ``P = diag(+1, -1) = 1 - 2n``,

that is, the Jordan-Wigner transformation with alternating signs, under which
``psi_l psi_{l+1} -> + sigma^-_{2l} sigma^-_{2l+2}``.  The parity string runs over the fermionic
sites with a strictly smaller Jordan-Wigner position.  The anticommutation relations are carried
entirely by the explicit parity strings, and each term is assembled as a single tensor product.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from qsimod.artifact import ConstraintOperator, LocalSubspace
from qsimod.jax_setup import COMPLEX_DTYPE, require_x64
from qsimod.pauli import PauliAxis, PauliString, PauliSum
from qsimod.realise.hilbert import HilbertSpace, local_matrix
from qsimod.structure import Algebra
from qsimod.symbolic import OperatorSum, OpSymbol, SiteOperator, Term

__all__ = [
    "build_operator",
    "build_pauli_string",
    "build_pauli_sum",
    "local_subspace_projector",
    "restrict",
    "sandwich",
    "sector_projector",
    "subspace_indices",
]

_LADDER = frozenset({OpSymbol.CREATE, OpSymbol.ANNIHILATE})


def _parity_matrix() -> Array:
    """The fermionic parity operator ``diag(+1, -1) = 1 - 2n``."""
    return jnp.diag(jnp.asarray([1.0, -1.0], dtype=COMPLEX_DTYPE))


def _expand(operator: SiteOperator, space: HilbertSpace) -> tuple[list[tuple[int, Array]], float]:
    """One site operator as an ordered list of local matrices together with a scalar phase."""
    local = space.space_at(operator.site)
    matrix = local_matrix(operator.symbol, local.algebra, local.dimension)
    if local.algebra is not Algebra.FERMION or operator.symbol not in _LADDER:
        return [(operator.site, matrix)], 1.0
    string = space.jordan_wigner_string(operator.site)
    position = local.jordan_wigner_position or 0
    factors: list[tuple[int, Array]] = [(site, _parity_matrix()) for site in string]
    factors.append((operator.site, matrix))
    return factors, (-1.0) ** position


def _term_matrix(term: Term, space: HilbertSpace, environment: Mapping[str, float]) -> Array:
    """The full-space matrix of one term."""
    coefficient = complex(term.coefficient.evaluate(environment))
    per_site: dict[int, Array] = {}
    for operator in term.operators:
        factors, phase = _expand(operator, space)
        coefficient *= phase
        for site, matrix in factors:
            existing = per_site.get(site)
            per_site[site] = matrix if existing is None else existing @ matrix
    return coefficient * space.kron_of(per_site)


def build_operator(
    hamiltonian: OperatorSum,
    space: HilbertSpace,
    environment: Mapping[str, float],
) -> Array:
    """Realise a symbolic operator sum as a dense JAX array.

    Args:
        hamiltonian: the symbolic sum.
        space: the Hilbert space in which the operator is constructed.
        environment: values for every parameter occurring in the coefficients.

    Returns:
        A ``(dimension, dimension)`` ``complex128`` array.

    Raises:
        KeyError: if a coefficient refers to an unbound parameter.
        ValueError: if an operator is not defined on the algebra of its site.

    """
    require_x64()
    total = jnp.zeros((space.dimension, space.dimension), dtype=COMPLEX_DTYPE)
    for term in hamiltonian.terms:
        total = total + _term_matrix(term, space, environment)
    return total


#: The primitive operator realising each Pauli axis.
_AXIS_SYMBOL = {
    PauliAxis.X: OpSymbol.PAULI_X,
    PauliAxis.Y: OpSymbol.PAULI_Y,
    PauliAxis.Z: OpSymbol.PAULI_Z,
}


def build_pauli_string(string: PauliString, qubits: int) -> Array:
    """Realise one Pauli string on a register of ``qubits`` qubits.

    The tensor order is that of [`HilbertSpace`][qsimod.realise.hilbert.HilbertSpace]; qubit 0
    is the leftmost, most significant factor.
    """
    require_x64()
    axes = string.mapping
    result = jnp.asarray([[1.0 + 0.0j]], dtype=COMPLEX_DTYPE)
    for qubit in range(qubits):
        axis = axes.get(qubit)
        factor = (
            jnp.eye(2, dtype=COMPLEX_DTYPE)
            if axis is None
            else local_matrix(_AXIS_SYMBOL[axis], Algebra.QUBIT, 2)
        )
        result = jnp.kron(result, factor)
    return result


def build_pauli_sum(pauli: PauliSum, qubits: int) -> Array:
    """Realise a Pauli sum on a register of ``qubits`` qubits."""
    require_x64()
    dimension = 2**qubits
    total = jnp.zeros((dimension, dimension), dtype=COMPLEX_DTYPE)
    for string in pauli.strings:
        total = total + pauli.terms[string] * build_pauli_string(string, qubits)
    return total


def local_subspace_projector(subspace: LocalSubspace, space: HilbertSpace) -> Array:
    """The projector onto a declared per-site occupation subspace, as a diagonal basis mask.

    Raises:
        ValueError: if the cutoff of the space cannot represent a listed occupation.

    """
    require_x64()
    for site, occupations in subspace.allowed_occupations.items():
        local = space.space_at(site)
        if max(occupations) >= local.dimension:
            msg = (
                f"subspace {subspace.name!r} needs occupation {max(occupations)} at site "
                f"{site}, but the local space has only {local.dimension} level(s); "
                f"realise with boson_cutoff >= {subspace.required_cutoff()}"
            )
            raise ValueError(msg)
    keep = [
        all(
            configuration[space.position_of(site)] in occupations
            for site, occupations in subspace.allowed_occupations.items()
        )
        for configuration in space.configurations()
    ]
    return jnp.diag(jnp.asarray(keep, dtype=COMPLEX_DTYPE))


def sector_projector(
    constraints: Sequence[ConstraintOperator],
    space: HilbertSpace,
    environment: Mapping[str, float],
    tolerance: float = 1e-10,
) -> Array:
    """The projector onto a declared superselection sector.

    The projector is constructed from the declared target eigenvalue of every constraint
    operator, boundary included.  When every constraint operator is diagonal in the occupation
    basis the projector is a basis mask; otherwise it is the projector onto the kernel of
    ``sum_l (G_l - g_l)**2``, obtained from one ``eigh``.

    Args:
        constraints: the declared constraint operators, with their target eigenvalues.
        space: the Hilbert space.
        environment: parameter values for the coefficients of the constraint operators.
        tolerance: the largest admissible distance between an eigenvalue and its target.

    Returns:
        The projector, as a dense array.

    """
    require_x64()
    if not constraints:
        return space.identity()
    matrices = [build_operator(c.operator, space, environment) for c in constraints]
    if all(_is_diagonal(matrix, tolerance) for matrix in matrices):
        keep = jnp.ones((space.dimension,), dtype=bool)
        for constraint, matrix in zip(constraints, matrices, strict=True):
            diagonal = jnp.real(jnp.diagonal(matrix))
            keep = keep & (jnp.abs(diagonal - constraint.target_value) < tolerance)
        return jnp.diag(keep.astype(COMPLEX_DTYPE))

    penalty = jnp.zeros((space.dimension, space.dimension), dtype=COMPLEX_DTYPE)
    identity = space.identity()
    for constraint, matrix in zip(constraints, matrices, strict=True):
        shifted = matrix - constraint.target_value * identity
        penalty = penalty + shifted @ shifted
    eigenvalues, eigenvectors = jnp.linalg.eigh(penalty)
    selected = eigenvectors[:, jnp.abs(eigenvalues) < tolerance]
    projector: Array = selected @ selected.conj().T
    return projector


def _is_diagonal(matrix: Array, tolerance: float) -> bool:
    """Whether ``matrix`` is diagonal to within ``tolerance``."""
    off_diagonal = matrix - jnp.diag(jnp.diagonal(matrix))
    return bool(jnp.max(jnp.abs(off_diagonal)) <= tolerance)


def sandwich(operator: Array, projector: Array) -> Array:
    """The product ``P A P``, the operator as it acts inside a declared subspace."""
    return projector @ operator @ projector


def subspace_indices(projector: Array, tolerance: float = 0.5) -> Array:
    """The basis indices retained by a diagonal projector, ascending."""
    return jnp.where(jnp.real(jnp.diagonal(projector)) > tolerance)[0]


def restrict(operator: Array, indices: Array) -> Array:
    """The block of ``operator`` on the basis states listed in ``indices``."""
    return operator[jnp.ix_(indices, indices)]
