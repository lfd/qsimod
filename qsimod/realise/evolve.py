"""Exact diagonalisation, time evolution and error measures.

Time evolution is computed exactly: the propagator ``U(t) = V diag(exp(-i E t)) V^dagger`` is
obtained from one Hermitian eigendecomposition and carries no time-discretisation error.  Two
error measures are provided: [`spectral_norm`][qsimod.realise.evolve.spectral_norm], the
operator 2-norm in which the Trotter error bounds are stated, and
[`max_abs_deviation`][qsimod.realise.evolve.max_abs_deviation], the largest absolute entry,
which is not a submultiplicative norm.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from qsimod.jax_setup import COMPLEX_DTYPE, REAL_DTYPE, require_x64

__all__ = [
    "commutator_spectral_norm",
    "eigensystem",
    "evolve_state",
    "expectation",
    "gauss_violation",
    "hermiticity_defect",
    "low_lying_spectrum",
    "manifold_levels",
    "max_abs_deviation",
    "propagator",
    "spacing_deviation",
    "spectral_norm",
]


def hermiticity_defect(operator: Array) -> float:
    """The defect ``max |H - H^dagger|``, zero to machine precision for a Hermitian operator."""
    return float(jnp.max(jnp.abs(operator - operator.conj().T)))


def eigensystem(operator: Array) -> tuple[Array, Array]:
    """The eigenvalues and eigenvectors of a Hermitian operator.

    Args:
        operator: a Hermitian array; ``eigh`` uses only its Hermitian part.

    Returns:
        ``(eigenvalues, eigenvectors)`` with eigenvalues ascending and eigenvectors in
        the columns.

    """
    require_x64()
    eigenvalues, eigenvectors = jnp.linalg.eigh(operator)
    return jnp.real(eigenvalues).astype(REAL_DTYPE), eigenvectors


def low_lying_spectrum(operator: Array, count: int = 8) -> Array:
    """The ``count`` lowest eigenvalues of a Hermitian operator, ascending."""
    eigenvalues, _ = eigensystem(operator)
    return eigenvalues[:count]


def manifold_levels(operator: Array, manifold: Sequence[int]) -> tuple[Array, float]:
    """The levels continuing from a degenerate manifold, selected by overlap.

    Each eigenvector is ranked by the weight it retains on the basis states of the manifold,
    and the ``len(manifold)`` eigenvectors of largest weight are selected.

    Args:
        operator: the Hermitian operator to diagonalise.
        manifold: the basis indices of the configurations of the manifold.

    Returns:
        As many levels as the manifold has configurations, ascending, and the smallest weight
        that any selected eigenvector retains inside the manifold.

    """
    values, vectors = eigensystem(operator)
    weight = jnp.sum(jnp.abs(vectors[jnp.asarray(manifold), :]) ** 2, axis=0)
    chosen = jnp.argsort(weight)[-len(manifold) :]
    return jnp.sort(values[chosen]), float(jnp.sort(weight)[-len(manifold)])


def spacing_deviation(left: ArrayLike, right: ArrayLike, scale: float = 1.0) -> float:
    """The largest level-by-level gap between two mean-centred spectra, in units of ``scale``.

    Args:
        left: one spectrum, ascending.
        right: the other spectrum, of the same length.
        scale: the energy unit in which the gap is reported.

    Returns:
        The gap.

    """
    centred_left = jnp.asarray(left) - jnp.mean(jnp.asarray(left))
    centred_right = jnp.asarray(right) - jnp.mean(jnp.asarray(right))
    return float(jnp.max(jnp.abs(centred_left - centred_right)) / abs(scale))


def propagator(operator: Array, time: float) -> Array:
    """The propagator ``exp(-i H t)`` of a Hermitian ``H``, from one eigendecomposition."""
    eigenvalues, eigenvectors = eigensystem(operator)
    phases = jnp.exp(-1j * eigenvalues * time).astype(COMPLEX_DTYPE)
    return (eigenvectors * phases) @ eigenvectors.conj().T


def evolve_state(operator: Array, state: Array, times: Sequence[float] | ArrayLike) -> Array:
    """Evolve ``state`` under a Hermitian ``H`` to each of ``times``, from one eigendecomposition.

    Args:
        operator: the Hermitian generator.
        state: the initial state vector.
        times: the times at which the state is reported, as a sequence or a 1-d array.

    Returns:
        An array of shape ``(len(times), dimension)``, one state per time.

    """
    require_x64()
    eigenvalues, eigenvectors = eigensystem(operator)
    amplitudes = eigenvectors.conj().T @ state
    time_array = jnp.asarray(times, dtype=REAL_DTYPE)
    phases = jnp.exp(-1j * time_array[:, None] * eigenvalues[None, :])
    return (phases * amplitudes[None, :]) @ eigenvectors.T


def expectation(operator: Array, states: Array) -> Array:
    """The expectation value of ``operator`` in one state or a stack of states.

    Args:
        operator: a Hermitian observable.
        states: a state vector, or an array of shape ``(times, dimension)``.

    Returns:
        A scalar, or one real value per state.

    """
    require_x64()
    if states.ndim == 1:
        return jnp.real(jnp.vdot(states, operator @ states))
    return jnp.real(jnp.einsum("ti,ij,tj->t", states.conj(), operator, states))


def gauss_violation(constraint_operators: Sequence[Array], states: Array) -> Array:
    """The gauge violation ``sum_l <G_l**2>``, the diagnostic of gauge invariance.

    Args:
        constraint_operators: the realised constraint operators.
        states: a state vector, or a stack of states.

    Returns:
        The summed expectation of the squares, per state.

    """
    total = None
    for operator in constraint_operators:
        contribution = expectation(operator @ operator, states)
        total = contribution if total is None else total + contribution
    if total is None:
        return jnp.zeros(() if states.ndim == 1 else (states.shape[0],), dtype=REAL_DTYPE)
    return total


def spectral_norm(operator: Array) -> float:
    """The operator 2-norm, the largest singular value.

    The Trotter error bounds are stated in this norm.
    """
    require_x64()
    return float(jnp.linalg.norm(operator, ord=2))


def max_abs_deviation(left: Array, right: Array) -> float:
    """The largest absolute entry of ``left - right``.

    This measure is not a submultiplicative norm and not the norm in which the Trotter bounds
    are stated.
    """
    return float(jnp.max(jnp.abs(left - right)))


def commutator_spectral_norm(left: Array, right: Array) -> float:
    """The commutator norm ``||[A, B]||`` in the spectral norm, for Hermitian ``A`` and ``B``.

    ``i[A, B]`` is Hermitian, so its norm is its largest absolute eigenvalue.
    """
    require_x64()
    bracket = 1j * (left @ right - right @ left)
    return float(jnp.max(jnp.abs(jnp.linalg.eigvalsh(bracket))))
