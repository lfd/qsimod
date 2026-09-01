"""The feasible set of a realisation problem, evaluated point-wise or region-wise.

[`FeasibleSet.check`][qsimod.solving.feasible.FeasibleSet.check] decides one assignment against
the admissible box, the coupled constraints, the equation residuals and the validity conditions.
[`FeasibleSet.grid`][qsimod.solving.feasible.FeasibleSet.grid] evaluates the feasible set on a
Cartesian grid, [`FeasibleSet.sample`][qsimod.solving.feasible.FeasibleSet.sample] on a seeded
uniform sample of the box (default seed `DEFAULT_SAMPLE_SEED`), and
[`FeasibleSet.constraint_system`][qsimod.solving.feasible.FeasibleSet.constraint_system] returns
the constraint system unevaluated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from qsimod.parameters import ConstraintViolation
from qsimod.solving.problem import ConstraintSystem, RealisationProblem
from qsimod.validity import ValidityReport

__all__ = ["DEFAULT_SAMPLE_SEED", "FeasiblePoint", "FeasibleSet"]

#: The random seed used by `FeasibleSet.sample` when none is supplied.
DEFAULT_SAMPLE_SEED = 20260831


@dataclass(frozen=True)
class FeasiblePoint:
    """The verdict on one point, with every reason for its failure.

    Attributes:
        values: the full parameter assignment tested.
        feasible: whether the point satisfies every constraint.
        violations: the admissibility constraints the point violates.
        validity: the validity report at the point.
        residuals: the equation residuals at the point.
        reasons: the failures in prose; empty when the point is feasible.

    """

    values: Mapping[str, float]
    feasible: bool
    violations: tuple[ConstraintViolation, ...] = ()
    validity: ValidityReport = field(default_factory=lambda: ValidityReport(()))
    residuals: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    @property
    def max_residual(self) -> float:
        """The largest absolute residual, or zero if there are no residuals."""
        return max((abs(value) for value in self.residuals.values()), default=0.0)

    @property
    def weakest_margin(self) -> float:
        """The least regime margin in decades, or ``nan`` if there is none."""
        return self.validity.weakest_margin

    def as_row(self) -> dict[str, float | bool | str]:
        """The point as a flat mapping, suitable as a row of a ``pandas`` frame."""
        row: dict[str, float | bool | str] = dict(self.values)
        row["feasible"] = self.feasible
        row["weakest_margin"] = self.weakest_margin
        row["max_residual"] = self.max_residual
        row["reasons"] = "; ".join(self.reasons)
        for outcome in self.validity.outcomes:
            row[f"margin[{outcome.name}]"] = outcome.margin
            row[f"value[{outcome.name}]"] = outcome.value
        return row

    def __str__(self) -> str:
        head = "feasible" if self.feasible else "INFEASIBLE"
        values = ", ".join(f"{k}={v:.6g}" for k, v in sorted(self.values.items()))
        detail = "" if self.feasible else "  because " + "; ".join(self.reasons)
        return f"{head}: {values}{detail}"


class FeasibleSet:
    """The feasible set of a realisation problem, evaluated point-wise or region-wise."""

    def __init__(self, problem: RealisationProblem, tolerance: float | None = None) -> None:
        """Construct the feasible set of a problem.

        Args:
            problem: the realisation problem whose feasible set is represented.
            tolerance: the largest residual tolerated before a point counts as infeasible;
                defaults to the tolerance of the problem.

        """
        self.problem = problem
        self.tolerance = problem.tolerance if tolerance is None else tolerance
        self.admissible_set = problem.system.admissible_set(
            description="from the problem's constraint system"
        )

    # -- the primitive -----------------------------------------------------------

    def check(self, assignment: Mapping[str, float]) -> FeasiblePoint:
        """Decide one assignment against every constraint of the problem."""
        values = dict(assignment)
        violations = tuple(self.admissible_set.violations(values))
        residuals = self.problem.system.residuals(values)
        validity = self.problem.validity.report(values)
        reasons: list[str] = [str(violation) for violation in violations]
        largest = max((abs(value) for value in residuals.values()), default=0.0)
        if largest > self.tolerance:
            worst = max(residuals, key=lambda name: abs(residuals[name]))
            reasons.append(
                f"equation {worst} has residual {residuals[worst]:+.3e}, above the "
                f"tolerance {self.tolerance:g}"
            )
        for outcome in validity.failures:
            reasons.append(
                f"{outcome.severity} failure: {outcome.name} ({outcome.quantity} = "
                f"{outcome.value:.6g}, requires {outcome.requirement})"
            )
        if not self.problem.enforce_regime_conditions:
            reasons = [reason for reason in reasons if "regime failure" not in reason]
        return FeasiblePoint(
            values=values,
            feasible=not reasons,
            violations=violations,
            validity=validity,
            residuals=residuals,
            reasons=tuple(reasons),
        )

    def _assignment(
        self,
        coordinates: Mapping[str, float],
        complete: Callable[[Mapping[str, float]], Mapping[str, float]] | None,
    ) -> dict[str, float]:
        """A grid node or sample, extended to a full parameter assignment.

        The fixed values of the problem are merged in before ``complete`` is applied.
        """
        node = {**self.problem.system.fixed, **coordinates}
        return node if complete is None else dict(complete(node))

    # -- region modes ------------------------------------------------------------

    def grid(
        self,
        axes: Mapping[str, Sequence[float]],
        complete: Callable[[Mapping[str, float]], Mapping[str, float]] | None = None,
    ) -> tuple[FeasiblePoint, ...]:
        """Evaluate the feasible set on a Cartesian grid over ``axes``.

        Args:
            axes: the gridded quantities and their values; they need not be unknowns of the
                problem.
            complete: a map from a grid node, with the fixed values of the problem merged in,
                to the full assignment.  Defaults to the identity.

        Returns:
            One verdict per node, in row-major order over the insertion order of ``axes``.

        """
        names = tuple(axes)
        nodes = product(*(tuple(axes[name]) for name in names))
        results: list[FeasiblePoint] = []
        for node in nodes:
            coordinates = dict(zip(names, node, strict=True))
            verdict = self.check(self._assignment(coordinates, complete))
            results.append(
                FeasiblePoint(
                    values={**coordinates, **verdict.values},
                    feasible=verdict.feasible,
                    violations=verdict.violations,
                    validity=verdict.validity,
                    residuals=verdict.residuals,
                    reasons=verdict.reasons,
                )
            )
        return tuple(results)

    def sample(
        self,
        count: int,
        complete: Callable[[Mapping[str, float]], Mapping[str, float]] | None = None,
        seed: int = DEFAULT_SAMPLE_SEED,
    ) -> tuple[FeasiblePoint, ...]:
        """Sample the admissible box uniformly and report the verdict on each point.

        Args:
            count: the number of points to draw.
            complete: as for [`grid`][qsimod.solving.feasible.FeasibleSet.grid].
            seed: the random seed.

        Returns:
            One verdict per sample.

        Raises:
            ValueError: if no unknown has a finite box, or if some unknown has no finite box
                and no ``complete`` is supplied to derive it.

        """
        unknowns = self.problem.system.unknowns
        bounded = [unknown for unknown in unknowns if unknown.interval.is_bounded]
        unbounded = [unknown.name for unknown in unknowns if not unknown.interval.is_bounded]
        if not bounded:
            msg = (
                "cannot sample: no unknown has a finite box.  Declare bounds on the "
                "hardware model, or use grid() over chosen axes instead"
            )
            raise ValueError(msg)
        if unbounded and complete is None:
            msg = (
                f"cannot sample: unknown(s) {unbounded} have no finite box.  These are a "
                "pipeline's intermediate parameters, which a device does not bound; pass a "
                "`complete` callback that derives them from the sampled knobs, as the "
                "window sweep does"
            )
            raise ValueError(msg)
        generator = np.random.default_rng(seed)
        results: list[FeasiblePoint] = []
        for _ in range(count):
            coordinates = {
                unknown.name: float(generator.uniform(unknown.interval.lo, unknown.interval.hi))
                for unknown in bounded
            }
            results.append(self.check(self._assignment(coordinates, complete)))
        return tuple(results)

    def constraint_system(self) -> ConstraintSystem:
        """The constraint system, unevaluated, for an external analysis."""
        return self.problem.system

    # -- summaries ---------------------------------------------------------------

    @staticmethod
    def feasible_fraction(points: Sequence[FeasiblePoint]) -> float:
        """The fraction of ``points`` that are feasible; ``nan`` for an empty sequence."""
        if not points:
            return math.nan
        return sum(1 for point in points if point.feasible) / len(points)

    @staticmethod
    def best(points: Sequence[FeasiblePoint]) -> FeasiblePoint | None:
        """The feasible point with the largest weakest margin, or ``None`` if there is none."""
        candidates = [
            point for point in points if point.feasible and not math.isnan(point.weakest_margin)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda point: point.weakest_margin)

    def __str__(self) -> str:
        return f"feasible set of {self.problem.name!r} ({self.admissible_set})"
