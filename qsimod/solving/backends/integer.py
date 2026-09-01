"""A discrete backend: monotone bisection over one integral unknown.

The hard constraints are assumed monotone in the unknown; the a-priori error bounds of the
product formulas are proportional to ``n**-p``.  The feasible set is therefore an up-set, and
the least feasible integer is found by doubling to bracket it and bisecting within the bracket,
in ``O(log n)`` evaluations.  Minimality is witnessed by the bisection: the result records that
``n - 1`` violates the constraints and ``n`` satisfies them.
[`minimal_resource_setting`][qsimod.solving.backends.integer.minimal_resource_setting] selects
the ``(order, steps)`` candidate of least cost.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qsimod.relations import RelationKind
from qsimod.solving.backends.base import BackendCapability, SolverBackend
from qsimod.solving.problem import RealisationProblem, Unknown, UnknownKind
from qsimod.solving.result import SolveResult, SolveStatus

__all__ = [
    "DiscreteSolution",
    "MonotoneBisectionBackend",
    "ResourceChoice",
    "minimal_feasible_integer",
    "minimal_resource_setting",
]


@dataclass(frozen=True)
class DiscreteSolution:
    """The least integer that satisfies a monotone constraint.

    Attributes:
        value: the least feasible integer.
        evaluations: the number of predicate evaluations the search took.
        upper_bracket: the bracket found by doubling, from which the bisection started.
        predecessor_fails: whether ``value`` is known to be minimal: either ``value - 1``
            failed during the bisection, or ``value`` is the lower bound of the search.

    """

    value: int
    evaluations: int
    upper_bracket: int
    predecessor_fails: bool

    def __str__(self) -> str:
        return (
            f"least feasible integer {self.value} "
            f"({self.evaluations} evaluation(s), bracket {self.upper_bracket}, "
            f"minimality {'witnessed' if self.predecessor_fails else 'unverified'})"
        )


def minimal_feasible_integer(
    predicate: Callable[[int], bool],
    *,
    lower: int = 1,
    maximum: int = 1 << 30,
) -> DiscreteSolution | None:
    """The least integer at or above ``lower`` for which ``predicate`` holds.

    The predicate is assumed monotone: once it holds, it holds for every larger integer.  The
    search brackets the solution by doubling and then bisects.

    Args:
        predicate: the feasibility test.
        lower: the smallest integer considered.
        maximum: the largest integer considered; beyond it the search returns ``None``.

    Returns:
        The solution, or ``None`` if no integer at or below ``maximum`` is feasible.

    """
    counter = _Counter(predicate)
    lowest = max(lower, 1)
    if lowest > maximum:
        return None
    if counter.check(lowest):
        return DiscreteSolution(lowest, counter.calls, lowest, predecessor_fails=True)

    high = lowest
    while True:
        if high >= maximum:
            return None
        # Double, but never look past `maximum`: it is the last candidate tried.
        low, high = high, min(2 * high, maximum)
        if counter.check(high):
            break
    bracket = high
    while high - low > 1:
        middle = (low + high) // 2
        if counter.check(middle):
            high = middle
        else:
            low = middle
    # `low` is the largest infeasible value seen and `high = low + 1` the least feasible.
    return DiscreteSolution(
        value=high,
        evaluations=counter.calls,
        upper_bracket=bracket,
        predecessor_fails=True,
    )


class _Counter:
    """A predicate wrapper that counts its evaluations."""

    def __init__(self, predicate: Callable[[int], bool]) -> None:
        self._predicate = predicate
        self.calls = 0

    def check(self, candidate: int) -> bool:
        """Evaluate the predicate, counting the call."""
        self.calls += 1
        return self._predicate(candidate)


def _integral_unknowns(problem: RealisationProblem) -> tuple[Unknown, ...]:
    """The integer-valued unknowns of the problem, in declaration order."""
    return tuple(
        unknown for unknown in problem.system.unknowns if unknown.kind is UnknownKind.INTEGRAL
    )


class MonotoneBisectionBackend(SolverBackend):
    """The backend that solves for the least admissible value of a single integral unknown.

    The problem must have exactly one integral unknown and no equalities; the hard constraints
    are evaluated with that unknown set to each candidate.
    """

    def __init__(self, maximum: int = 1 << 24) -> None:
        self.name = "monotone bisection over an integer"
        self.maximum = maximum
        self.capability = BackendCapability(
            continuous=False,
            integral=True,
            equalities=False,
            inequalities=True,
            proves_infeasibility=False,
        )

    def supports(self, problem: RealisationProblem) -> bool:
        """Whether the problem has exactly one unknown, which is integral, and no equalities."""
        return (
            len(_integral_unknowns(problem)) == 1
            and len(problem.system.unknowns) == 1
            and not problem.system.equalities
        )

    def solve(self, problem: RealisationProblem) -> SolveResult:
        """Bisect for the least feasible value of the integral unknown.

        Raises:
            ValueError: if the problem does not have exactly one integral unknown.

        """
        integral = _integral_unknowns(problem)
        if len(integral) != 1:
            msg = (
                f"{self.name} needs exactly one integral unknown; {problem.name!r} has "
                f"{len(integral)}"
            )
            raise ValueError(msg)
        unknown = integral[0]
        constraints = problem.hard_constraints()
        lower = int(unknown.interval.lo) if unknown.interval.lo > float("-inf") else 1
        # The unknown's own box caps the search below the backend's ceiling.
        maximum = self.maximum
        if unknown.interval.hi < float("inf"):
            maximum = min(maximum, int(unknown.interval.hi))

        def feasible(candidate: int) -> bool:
            point = {**problem.system.fixed, unknown.name: float(candidate)}
            return all(expression.evaluate_real(point) >= 0.0 for _, expression in constraints)

        solution = minimal_feasible_integer(feasible, lower=lower, maximum=maximum)
        if solution is None:
            return SolveResult(
                status=SolveStatus.UNSOLVED,
                relation_kind=RelationKind.IMPLICIT,
                objective_name=problem.objective.name,
                backend=self.name,
                termination=(
                    f"no value of {unknown.name} up to {maximum} satisfies the "
                    "constraints; the search was exhausted, nothing was proved"
                ),
                seed=None,
            )
        point = {**problem.system.fixed, unknown.name: float(solution.value)}
        return SolveResult(
            status=SolveStatus.EXACT_SOLUTION,
            point=point,
            residuals={},
            relation_kind=RelationKind.IMPLICIT,
            validity=problem.validity.report(point),
            objective_name=problem.objective.name,
            objective_value=problem.objective.value(point),
            backend=self.name,
            iterations=solution.evaluations,
            termination=str(solution),
            seed=None,
            notes={
                "minimality": (
                    f"{unknown.name} = {solution.value} satisfies the constraints and "
                    f"{solution.value - 1} does not"
                    if solution.predecessor_fails
                    else f"{unknown.name} = {solution.value} is the lower bound of the search"
                )
            },
        )


@dataclass(frozen=True)
class ResourceChoice:
    """One candidate ``(order, steps)`` pair and its cost.

    Attributes:
        order: the order of the product formula.
        steps: the least step count that meets the accuracy target at that order.
        cost: the total count of k-local unitaries, by which candidates are ranked.
        error_bound: the a-priori error bound at that setting.
        depth_in_layers: the depth in layers of disjoint support.

    """

    order: int
    steps: int
    cost: int
    error_bound: float
    depth_in_layers: int

    def __str__(self) -> str:
        return (
            f"order {self.order}: n={self.steps}, {self.cost} k-local factors, "
            f"depth {self.depth_in_layers}, bound {self.error_bound:.3e}"
        )


def minimal_resource_setting(
    candidates: Sequence[ResourceChoice],
) -> ResourceChoice:
    """The candidate of least cost; ties are broken in favour of the lower order.

    Raises:
        ValueError: if there are no candidates.

    """
    if not candidates:
        msg = "no (order, steps) candidate met the accuracy target"
        raise ValueError(msg)
    return min(candidates, key=lambda choice: (choice.cost, choice.order))
