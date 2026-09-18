"""A-priori error bounds of product formulas in the spectral norm.

[`error_bound`][qsimod.trotter.bounds.error_bound] returns the minimum of the applicable bounds:

* for order 1, the commutator bound
  ``|| exp(-iHt) - S_1(t/n)**n || <= (t**2 / 2n) sum_{g<g'} ||[H_g, H_g']||``;
* for order 2, the commutator bound of Childs et al. with ``tau = t/n`` and the tail
  ``R_g = sum_{g'>g} H_g'``, per step
  ``(tau**3/12) sum_g ||[R_g, [R_g, H_g]]|| + (tau**3/24) sum_g ||[H_g, [R_g, H_g]]||``;
* for every order, the stage-based Taylor-remainder bound
  ``t**(p+1) / (p+1)! / n**p * (||H|| + sum_s |c_s| ||H_g(s)||)**(p+1)``.

Norms are supplied by a [`NormEstimator`][qsimod.trotter.bounds.NormEstimator]; the default
[`AdaptiveNormEstimator`][qsimod.trotter.bounds.AdaptiveNormEstimator] is exact up to
`DEFAULT_SPECTRAL_QUBIT_LIMIT` qubits and uses the Pauli one-norm beyond.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

from qsimod.pauli import PauliSum, commutator
from qsimod.trotter.formulas import ProductFormulaSchedule
from qsimod.trotter.layers import LayerDecomposition

__all__ = [
    "DEFAULT_SPECTRAL_QUBIT_LIMIT",
    "AdaptiveNormEstimator",
    "ErrorBound",
    "NormEstimator",
    "OneNormEstimator",
    "SpectralNormEstimator",
    "commutator_bound",
    "error_bound",
    "taylor_remainder_bound",
]

#: The largest register size for which the adaptive estimator constructs a dense operator;
#: above this size it uses the one-norm bound.
DEFAULT_SPECTRAL_QUBIT_LIMIT = 14


class NormEstimator(ABC):
    """An estimator that provides an upper bound on the spectral norm of a Pauli sum."""

    name: str

    @abstractmethod
    def norm(self, operator: PauliSum) -> float:
        """An upper bound on ``||operator||`` in the spectral norm."""

    @abstractmethod
    def is_exact(self) -> bool:
        """Whether the estimate is the spectral norm itself rather than a bound."""


@dataclass(frozen=True)
class OneNormEstimator(NormEstimator):
    """The bound ``sum |c_P|`` obtained from the triangle inequality on the Pauli coefficients."""

    name: str = field(default="one-norm (triangle inequality)", init=False)

    def norm(self, operator: PauliSum) -> float:
        """The one-norm ``sum |c_P|`` of the coefficients."""
        return operator.one_norm

    def is_exact(self) -> bool:
        """Always ``False``; the one-norm is a bound, not the spectral norm itself."""
        return False


@dataclass(frozen=True)
class SpectralNormEstimator(NormEstimator):
    """The exact spectral norm, obtained by constructing the dense operator and diagonalising it.

    Attributes:
        qubits: the register size in which the operator is constructed; it must cover the
            support of the operator.

    """

    qubits: int
    name: str = field(default="spectral norm (exact)", init=False)

    def norm(self, operator: PauliSum) -> float:
        """``||operator||`` in the spectral norm."""
        # Deferred: keeps the structural layer free of JAX.
        from qsimod.realise.build import build_pauli_sum  # noqa: PLC0415
        from qsimod.realise.evolve import spectral_norm  # noqa: PLC0415

        if operator.is_zero:
            return 0.0
        return spectral_norm(build_pauli_sum(operator, self.qubits))

    def is_exact(self) -> bool:
        """Always ``True``."""
        return True


@dataclass(frozen=True)
class AdaptiveNormEstimator(NormEstimator):
    """An estimator that is exact on small registers and uses the one-norm bound on larger ones.

    Attributes:
        qubits: the register size.
        limit: the largest register size for which the exact spectral norm is computed.

    """

    qubits: int
    limit: int = DEFAULT_SPECTRAL_QUBIT_LIMIT
    name: str = field(default="adaptive", init=False)

    @property
    def _delegate(self) -> NormEstimator:
        if self.qubits <= self.limit:
            return SpectralNormEstimator(self.qubits)
        return OneNormEstimator()

    def norm(self, operator: PauliSum) -> float:
        """The estimate of the delegate, the exact or the one-norm estimator."""
        return self._delegate.norm(operator)

    def is_exact(self) -> bool:
        """Whether the exact estimator is the delegate."""
        return self._delegate.is_exact()


@dataclass(frozen=True)
class ErrorBound:
    """An a-priori bound on ``|| exp(-iHt) - S_p(t/n)**n ||`` in the spectral norm.

    Attributes:
        value: the bound.
        order: the order of the formula.
        time: the simulated time ``t``.
        steps: the step count ``n``.
        method: the family of bounds that produced ``value``.
        candidates: every candidate bound computed, by method name.
        components: the quantities from which the chosen bound is assembled, for instance
            individual commutator norms.
        norm_estimator: the name of the norm estimator.
        exact_norms: whether the underlying norms were exact spectral norms.

    """

    value: float
    order: int
    time: float
    steps: int
    method: str
    candidates: Mapping[str, float] = field(default_factory=dict)
    components: Mapping[str, float] = field(default_factory=dict)
    norm_estimator: str = ""
    exact_norms: bool = False

    def __str__(self) -> str:
        candidates = ", ".join(
            f"{name}={value:.4e}" for name, value in sorted(self.candidates.items())
        )
        return (
            f"bound {self.value:.6e} at order {self.order}, t={self.time:g}, n={self.steps} "
            f"[{self.method}; norms: {self.norm_estimator}"
            f"{'' if self.exact_norms else ', bounded'}]  candidates: {candidates}"
        )


def _tail(decomposition: LayerDecomposition, after: int) -> PauliSum:
    """The tail ``sum_{g > after} H_g`` of the layer ordering."""
    total = PauliSum.zero("tail")
    for layer in decomposition.layers[after + 1 :]:
        total = total + layer.operator
    return total


def commutator_bound(
    decomposition: LayerDecomposition,
    order: int,
    time: float,
    steps: int,
    estimator: NormEstimator,
) -> ErrorBound | None:
    """The commutator bound, implemented for orders 1 and 2.

    Returns:
        The bound, or ``None`` for an order above 2.

    Raises:
        ValueError: if ``steps`` is not positive.

    """
    if steps < 1:
        msg = f"step count must be positive, got {steps}"
        raise ValueError(msg)
    if order == 1:
        return _first_order_bound(decomposition, time, steps, estimator)
    if order == 2:
        return _second_order_bound(decomposition, time, steps, estimator)
    return None


def _first_order_bound(
    decomposition: LayerDecomposition,
    time: float,
    steps: int,
    estimator: NormEstimator,
) -> ErrorBound:
    """The first-order commutator bound ``(t**2 / 2n) sum_{g<g'} ||[H_g, H_g']||``."""
    components: dict[str, float] = {}
    total = 0.0
    for i, j in decomposition.non_commuting_pairs():
        bracket = decomposition.layer_commutator(i, j)
        norm = estimator.norm(bracket)
        components[f"||[{decomposition.names[i]},{decomposition.names[j]}]||"] = norm
        total += norm
    value = time * time / (2.0 * steps) * total
    return ErrorBound(
        value=value,
        order=1,
        time=time,
        steps=steps,
        method="commutator (first order)",
        candidates={"commutator (first order)": value},
        components=components,
        norm_estimator=estimator.name,
        exact_norms=estimator.is_exact(),
    )


def _second_order_bound(
    decomposition: LayerDecomposition,
    time: float,
    steps: int,
    estimator: NormEstimator,
) -> ErrorBound:
    """The second-order commutator bound of Childs et al., with constants ``1/12`` and ``1/24``."""
    tau = time / steps
    components: dict[str, float] = {}
    first_sum = 0.0
    second_sum = 0.0
    for index, layer in enumerate(decomposition.layers):
        tail = _tail(decomposition, index)
        if tail.is_zero:
            continue
        inner = commutator(tail, layer.operator)
        if inner.is_zero:
            continue
        outer_tail = estimator.norm(commutator(tail, inner))
        outer_self = estimator.norm(commutator(layer.operator, inner))
        components[f"||[R,[R,{layer.name}]]||"] = outer_tail
        components[f"||[{layer.name},[R,{layer.name}]]||"] = outer_self
        first_sum += outer_tail
        second_sum += outer_self
    per_step = tau**3 * (first_sum / 12.0 + second_sum / 24.0)
    value = steps * per_step
    return ErrorBound(
        value=value,
        order=2,
        time=time,
        steps=steps,
        method="commutator (second order, Childs-style)",
        candidates={"commutator (second order, Childs-style)": value},
        components=components,
        norm_estimator=estimator.name,
        exact_norms=estimator.is_exact(),
    )


def taylor_remainder_bound(
    decomposition: LayerDecomposition,
    schedule: ProductFormulaSchedule,
    time: float,
    steps: int,
    estimator: NormEstimator,
) -> ErrorBound:
    """The stage-based Taylor-remainder bound stated in the module docstring.

    Raises:
        ValueError: if ``steps`` is not positive.

    """
    if steps < 1:
        msg = f"step count must be positive, got {steps}"
        raise ValueError(msg)
    order = schedule.order
    layer_norms = [estimator.norm(layer.operator) for layer in decomposition.layers]
    hamiltonian_norm = estimator.norm(decomposition.total())
    stage_weight = sum(
        abs(stage.coefficient) * layer_norms[stage.layer] for stage in schedule.stages
    )
    scale = hamiltonian_norm + stage_weight
    value = (
        abs(time) ** (order + 1) / math.factorial(order + 1) / steps**order * scale ** (order + 1)
    )
    return ErrorBound(
        value=value,
        order=order,
        time=time,
        steps=steps,
        method="Taylor remainder (stage-based, rigorous)",
        candidates={"Taylor remainder (stage-based, rigorous)": value},
        components={
            "||H||": hamiltonian_norm,
            "sum_s |c_s| ||H_g(s)||": stage_weight,
        },
        norm_estimator=estimator.name,
        exact_norms=estimator.is_exact(),
    )


def error_bound(
    decomposition: LayerDecomposition,
    schedule: ProductFormulaSchedule,
    time: float,
    steps: int,
    estimator: NormEstimator | None = None,
) -> ErrorBound:
    """The tightest available a-priori error bound at the given settings.

    Args:
        decomposition: the layer decomposition to which the formula is applied.
        schedule: the stage schedule.
        time: the simulated time.
        steps: the number of Trotter steps.
        estimator: the norm estimator; defaults to
            [`AdaptiveNormEstimator`][qsimod.trotter.bounds.AdaptiveNormEstimator] on the
            register of the decomposition.

    Returns:
        The minimum of the applicable bounds; every candidate is recorded.

    """
    estimator = estimator or AdaptiveNormEstimator(decomposition.qubit_count)
    candidates: dict[str, float] = {}
    best = taylor_remainder_bound(decomposition, schedule, time, steps, estimator)
    candidates.update(best.candidates)
    commutator_candidate = commutator_bound(decomposition, schedule.order, time, steps, estimator)
    if commutator_candidate is not None:
        candidates.update(commutator_candidate.candidates)
        if commutator_candidate.value < best.value:
            best = commutator_candidate
    return ErrorBound(
        value=best.value,
        order=schedule.order,
        time=time,
        steps=steps,
        method=best.method,
        candidates=candidates,
        components=best.components,
        norm_estimator=estimator.name,
        exact_norms=estimator.is_exact(),
    )
