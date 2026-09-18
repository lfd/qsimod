"""The Trotter step count as an instance of the general solve.

The Trotterisation is an approximate transformation whose error is controlled by a resource
parameter, the step count ``n`` of the product formula.  Given a simulated time ``t``, an order
``p`` and a target error ``epsilon``, the solve determines the smallest integer ``n`` with
``error_bound(t, n, p) <= epsilon``.  Every bound in
[`qsimod.trotter.bounds`][qsimod.trotter.bounds] has the form ``A / n**p`` at fixed ``t`` and
order; ``A`` is recovered by evaluating the bound at ``n = 1``, and the constraint becomes the
scalar expression ``epsilon - A * n**(-p) >= 0`` over the integral unknown `STEPS_SYMBOL`.  The
joint choice of ``(n, order)`` is one bisection per order, ranked by the total count of k-local
unitaries.
"""

from __future__ import annotations

from collections.abc import Sequence

from qsimod.parameters import Bound, InequalityConstraint
from qsimod.scalar import Symbol
from qsimod.solving.backends.base import SolverBackend
from qsimod.solving.backends.integer import (
    ResourceChoice,
)
from qsimod.solving.dispatch import solve
from qsimod.solving.objectives import minimise
from qsimod.solving.problem import (
    ConstraintSystem,
    RealisationProblem,
    Unknown,
    UnknownKind,
)
from qsimod.solving.result import SolveResult, SolveStatus
from qsimod.trotter.bounds import NormEstimator, error_bound
from qsimod.trotter.formulas import suzuki
from qsimod.trotter.layers import LayerDecomposition
from qsimod.trotter.schedule import factors_for, layer_depth_of

__all__ = [
    "STEPS_SYMBOL",
    "bound_coefficient",
    "minimal_steps",
    "resource_candidates",
    "step_count_problem",
]

#: The name of the step count inside the constraint system.
STEPS_SYMBOL = "trotter.n"

#: The largest step count the bisection considers.
DEFAULT_MAX_STEPS = 1 << 22


def bound_coefficient(
    decomposition: LayerDecomposition,
    order: int,
    time: float,
    estimator: NormEstimator | None = None,
) -> float:
    """The coefficient ``A`` in ``error_bound(t, n, order) == A / n**order``.

    It is obtained by evaluating the bound at ``n = 1``.
    """
    schedule = suzuki(order, len(decomposition))
    return error_bound(decomposition, schedule, time, 1, estimator).value


def step_count_problem(
    decomposition: LayerDecomposition,
    order: int,
    time: float,
    target_error: float,
    *,
    estimator: NormEstimator | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    name: str = "",
) -> RealisationProblem:
    """Pose the solve for the smallest ``n`` whose error bound is at most ``target_error``.

    Args:
        decomposition: the layer partition to which the product formula is applied.
        order: the order of the product formula.
        time: the simulated time.
        target_error: the accuracy target.
        estimator: the estimator that bounds the layer and commutator norms.
        max_steps: the largest step count considered.
        name: the name of the problem.

    Returns:
        The declarative problem, in the shape accepted by
        [`MonotoneBisectionBackend`][qsimod.solving.backends.integer.MonotoneBisectionBackend].

    Raises:
        ValueError: if ``target_error`` is not positive.

    """
    if target_error <= 0.0:
        msg = f"the target error must be positive, got {target_error!r}"
        raise ValueError(msg)
    coefficient = bound_coefficient(decomposition, order, time, estimator)
    steps = Symbol(STEPS_SYMBOL)
    constraint = InequalityConstraint(
        name=f"error bound at order {order} is at most {target_error:g}",
        expression=target_error - coefficient * steps ** (-float(order)),
        description=(
            f"the a-priori bound is {coefficient:.6e} / n**{order} at t = {time:g}; the "
            "constraint is monotone in n, which is what makes bisection exact"
        ),
    )
    system = ConstraintSystem(
        unknowns=(
            Unknown(
                name=STEPS_SYMBOL,
                kind=UnknownKind.INTEGRAL,
                bound=Bound(STEPS_SYMBOL, 1.0, float(max_steps)),
            ),
        ),
        equalities=(),
        inequalities=(constraint,),
        fixed={},
    )
    return RealisationProblem(
        name=name or f"least n at order {order} for error <= {target_error:g}",
        system=system,
        objective=minimise(
            steps,
            name="minimise the Trotter step count",
            description="the resource to be spent",
        ),
        max_iterations=max_steps,
    )


def minimal_steps(
    decomposition: LayerDecomposition,
    order: int,
    time: float,
    target_error: float,
    *,
    estimator: NormEstimator | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    backends: Sequence[SolverBackend] | None = None,
) -> SolveResult:
    """Solve for the least step count that meets an accuracy target.

    Returns:
        The solve result; ``point[STEPS_SYMBOL]`` is the step count, and
        ``notes["minimality"]`` records that ``n - 1`` violates the bound while ``n`` satisfies
        it.

    """
    problem = step_count_problem(
        decomposition,
        order,
        time,
        target_error,
        estimator=estimator,
        max_steps=max_steps,
    )
    return solve(problem, backends)


def resource_candidates(
    decomposition: LayerDecomposition,
    time: float,
    target_error: float,
    orders: Sequence[int] = (1, 2, 4),
    *,
    estimator: NormEstimator | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[ResourceChoice, ...]:
    """One [`ResourceChoice`][qsimod.solving.backends.integer.ResourceChoice] per feasible order.

    The candidates are ranked by the total count of k-local unitary factors; an order that
    cannot meet the target within ``max_steps`` is absent.
    """
    candidates: list[ResourceChoice] = []
    for order in orders:
        outcome = minimal_steps(
            decomposition,
            order,
            time,
            target_error,
            estimator=estimator,
            max_steps=max_steps,
        )
        if outcome.status is not SolveStatus.EXACT_SOLUTION:
            continue
        steps = round(outcome.point[STEPS_SYMBOL])
        schedule = suzuki(order, len(decomposition))
        factors = factors_for(decomposition, schedule, time / steps)
        candidates.append(
            ResourceChoice(
                order=order,
                steps=steps,
                cost=len(factors) * steps,
                error_bound=error_bound(decomposition, schedule, time, steps, estimator).value,
                depth_in_layers=layer_depth_of(factors) * steps,
            )
        )
    return tuple(candidates)
