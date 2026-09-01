"""A backend that evaluates a closed-form elimination chain.

When the static classification identifies the relation as a closed form in the requested
direction, the elimination chain is evaluated in order, without iteration, tolerance, starting
point or optimiser.  The backend imports nothing beyond the expression evaluator of the package.
"""

from __future__ import annotations

from qsimod.relations import RelationKind
from qsimod.solving.backends.base import BackendCapability, SolverBackend
from qsimod.solving.problem import RealisationProblem
from qsimod.solving.result import SolveResult, SolveStatus

__all__ = ["ClosedFormBackend"]


class ClosedFormBackend(SolverBackend):
    """The backend that evaluates a closed-form elimination chain directly."""

    def __init__(self) -> None:
        self.name = "closed-form evaluation"
        self.capability = BackendCapability(
            continuous=True, integral=True, equalities=True, inequalities=True
        )

    def supports(self, problem: RealisationProblem) -> bool:
        """Whether the classification of the problem provides an elimination chain."""
        classification = problem.system.classification
        return (
            classification is not None
            and classification.kind is RelationKind.CLOSED_FORM
            and classification.elimination is not None
        )

    def solve(self, problem: RealisationProblem) -> SolveResult:
        """Evaluate the chain and report the resulting point.

        Raises:
            ValueError: if the problem has no elimination chain.

        """
        classification = problem.system.classification
        if classification is None or classification.elimination is None:
            msg = (
                f"{self.name} needs a closed-form elimination chain; the problem "
                f"{problem.name!r} has none"
            )
            raise ValueError(msg)
        relation = problem.system.relation
        if relation is None:  # pragma: no cover -- a classification implies a relation
            msg = f"{problem.name!r} has a classification but no relation"
            raise ValueError(msg)

        point = relation.evaluate_closed_form(classification.elimination, problem.system.fixed)
        residuals = problem.system.residuals(point)
        validity = problem.validity.report(point)
        largest = max((abs(value) for value in residuals.values()), default=0.0)
        rejections = problem.rejections(point)
        violations = tuple(problem.system.admissible_set().violations(point))
        if rejections:
            # The chain determines the point, and that point is not acceptable.  This backend
            # does not claim to prove infeasibility, so the outcome is UNSOLVED.
            status = SolveStatus.UNSOLVED
            termination = (
                "closed form evaluated in one pass; its point is not an acceptable solution: "
                + "; ".join(rejections)
            )
        else:
            status = (
                SolveStatus.EXACT_SOLUTION
                if largest <= problem.tolerance
                else SolveStatus.APPROXIMATE_SOLUTION
            )
            termination = "closed form evaluated in one pass; no iteration"
        return SolveResult(
            status=status,
            point=point,
            residuals=residuals,
            relation_kind=classification.kind,
            validity=validity,
            binding_constraints=tuple(violation.constraint for violation in violations),
            violations=violations,
            objective_name=problem.objective.name,
            objective_value=problem.objective.value(point),
            backend=self.name,
            iterations=0,
            termination=termination,
            seed=None,
            notes={
                "elimination": " -> ".join(
                    definition.parameter for definition in classification.elimination
                )
            },
        )
