"""Suzuki-Trotter product formulas as stage schedules independent of the Hamiltonian.

A stage schedule is an ordered list of ``(layer index, coefficient)`` pairs whose coefficients
sum to one on every layer; [`qsimod.trotter.schedule`][qsimod.trotter.schedule] applies a
schedule to a layer decomposition.  The first-order Lie-Trotter formula, the second-order Strang
formula and the Suzuki recursion for any even order ``2k >= 4`` are provided, with

``S_2k(tau) = S_2k-2(p tau)**2 . S_2k-2((1 - 4p) tau) . S_2k-2(p tau)**2``,
``p = 1 / (4 - 4**(1/(2k-1)))``.

Adjacent stages on the same layer are merged by the identity
``exp(-i a H) exp(-i b H) = exp(-i (a+b) H)`` at every level of the recursion; stages are not
merged across step boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ProductFormulaSchedule",
    "Stage",
    "lie_trotter",
    "strang",
    "supported_orders",
    "suzuki",
    "suzuki_parameter",
]


@dataclass(frozen=True)
class Stage:
    """One exponential in a product formula.

    Attributes:
        layer: the index of the layer that the stage exponentiates.
        coefficient: the multiple of the step size ``tau`` in the exponent.

    """

    layer: int
    coefficient: float

    def __str__(self) -> str:
        return f"exp(-i {self.coefficient:+.6g} tau H_{self.layer})"


@dataclass(frozen=True)
class ProductFormulaSchedule:
    """An ordered list of stages: one step of a product formula, independent of the Hamiltonian.

    Attributes:
        order: the order ``p`` of the formula; the error per step is ``O(tau**(p+1))``.
        layer_count: the number of layers the schedule addresses.
        stages: the stages, from left to right as written; ``stages[0]`` is the leftmost
            factor of the product.
        merged: whether adjacent same-layer stages were merged.

    """

    order: int
    layer_count: int
    stages: tuple[Stage, ...]
    merged: bool = True

    def __post_init__(self) -> None:
        if self.order < 1:
            msg = f"product formula order must be at least 1, got {self.order}"
            raise ValueError(msg)
        for layer in range(self.layer_count):
            total = sum(stage.coefficient for stage in self.stages if stage.layer == layer)
            if abs(total - 1.0) > 1e-12:
                msg = (
                    f"the coefficients of layer {layer} sum to {total!r}, not 1; the "
                    "schedule would not reproduce the Hamiltonian at first order"
                )
                raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.stages)

    @property
    def stage_count(self) -> int:
        """The number of exponentials in one step."""
        return len(self.stages)

    @property
    def one_norm_of_coefficients(self) -> float:
        """``sum_s |c_s|``, as used by the Taylor-remainder error bound."""
        return sum(abs(stage.coefficient) for stage in self.stages)

    @property
    def is_symmetric(self) -> bool:
        """Whether the stage list is invariant under reversal."""
        return self.stages == tuple(reversed(self.stages))

    def __str__(self) -> str:
        body = " ".join(str(stage) for stage in self.stages)
        return f"S_{self.order} ({self.stage_count} stages): {body}"

    def __repr__(self) -> str:
        return f"ProductFormulaSchedule(order={self.order}, stages={self.stage_count})"


def _merge_adjacent(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    """Combine neighbouring stages that act on the same layer."""
    merged: list[Stage] = []
    for stage in stages:
        if merged and merged[-1].layer == stage.layer:
            merged[-1] = Stage(stage.layer, merged[-1].coefficient + stage.coefficient)
        else:
            merged.append(stage)
    return tuple(stage for stage in merged if stage.coefficient != 0.0)


def _require_layers(layer_count: int) -> None:
    """Reject a schedule over an empty set of layers.

    Raises:
        ValueError: if ``layer_count`` is less than one.

    """
    if layer_count < 1:
        msg = f"need at least one layer, got {layer_count}"
        raise ValueError(msg)


def lie_trotter(layer_count: int) -> ProductFormulaSchedule:
    """The first-order Lie-Trotter formula, one full-step exponential per layer in layer order.

    Raises:
        ValueError: if there are no layers.

    """
    _require_layers(layer_count)
    return ProductFormulaSchedule(
        order=1,
        layer_count=layer_count,
        stages=tuple(Stage(layer, 1.0) for layer in range(layer_count)),
    )


def strang(layer_count: int) -> ProductFormulaSchedule:
    """The second-order symmetric formula of Strang.

    For three layers this is the conventional ``S_2``,
    ``exp(-i H_0 tau/2) exp(-i H_1 tau/2) exp(-i H_2 tau) exp(-i H_1 tau/2) exp(-i H_0 tau/2)``.

    Raises:
        ValueError: if there are no layers.

    """
    _require_layers(layer_count)
    if layer_count == 1:
        return ProductFormulaSchedule(order=2, layer_count=1, stages=(Stage(0, 1.0),))
    forward = [Stage(layer, 0.5) for layer in range(layer_count - 1)]
    stages = [*forward, Stage(layer_count - 1, 1.0), *reversed(forward)]
    return ProductFormulaSchedule(order=2, layer_count=layer_count, stages=_merge_adjacent(stages))


def suzuki_parameter(order: int) -> float:
    """The Suzuki recursion parameter ``p = 1 / (4 - 4**(1/(2k-1)))`` for ``order = 2k``.

    Raises:
        ValueError: if ``order`` is not an even integer of at least four.

    """
    if order < 4 or order % 2 != 0:
        msg = f"the Suzuki recursion parameter is defined for even orders >= 4, got {order}"
        raise ValueError(msg)
    return 1.0 / (4.0 - float(4.0 ** (1.0 / (order - 1))))


def suzuki(order: int, layer_count: int) -> ProductFormulaSchedule:
    """The product formula of the given order over ``layer_count`` layers.

    Args:
        order: 1 for Lie-Trotter, 2 for Strang, or any even ``2k >= 4`` for the Suzuki
            recursion.
        layer_count: the number of layers.

    Returns:
        The stage schedule.

    Raises:
        ValueError: for an unsupported order.

    """
    if order == 1:
        return lie_trotter(layer_count)
    if order == 2:
        return strang(layer_count)
    if order < 4 or order % 2 != 0:
        msg = (
            f"unsupported product formula order {order}; supported are 1, 2 and any even "
            "order >= 4 via the Suzuki recursion"
        )
        raise ValueError(msg)
    inner = suzuki(order - 2, layer_count)
    parameter = suzuki_parameter(order)
    weights = (parameter, parameter, 1.0 - 4.0 * parameter, parameter, parameter)
    stages: list[Stage] = [
        Stage(stage.layer, stage.coefficient * weight)
        for weight in weights
        for stage in inner.stages
    ]
    return ProductFormulaSchedule(
        order=order, layer_count=layer_count, stages=_merge_adjacent(stages)
    )


def supported_orders(maximum: int = 8) -> tuple[int, ...]:
    """The orders [`suzuki`][qsimod.trotter.formulas.suzuki] accepts, up to ``maximum``."""
    return (1, *tuple(order for order in range(2, maximum + 1, 2)))
