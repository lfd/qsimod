"""The interface of a solver backend.

A backend accepts a [`RealisationProblem`][qsimod.solving.problem.RealisationProblem] and
returns a [`SolveResult`][qsimod.solving.result.SolveResult];
[`SolverBackend.supports`][qsimod.solving.backends.base.SolverBackend.supports] declares the
problem shapes a backend handles, by which the dispatcher selects one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from qsimod.solving.problem import RealisationProblem
from qsimod.solving.result import SolveResult

__all__ = ["BackendCapability", "SolverBackend"]


class BackendCapability:
    """The flags that describe the problem features a backend handles.

    Attributes:
        continuous: whether the backend handles continuous unknowns.
        integral: whether the backend handles integral unknowns.
        equalities: whether the backend handles equality constraints.
        inequalities: whether the backend handles inequality constraints.
        proves_infeasibility: whether a negative answer of the backend constitutes a proof.
            ``False`` for every numerical backend of the package, whose failures are reported
            as ``UNSOLVED``.

    """

    def __init__(
        self,
        *,
        continuous: bool = True,
        integral: bool = False,
        equalities: bool = True,
        inequalities: bool = True,
        proves_infeasibility: bool = False,
    ) -> None:
        self.continuous = continuous
        self.integral = integral
        self.equalities = equalities
        self.inequalities = inequalities
        self.proves_infeasibility = proves_infeasibility

    def __str__(self) -> str:
        flags = [
            name
            for name, value in (
                ("continuous", self.continuous),
                ("integral", self.integral),
                ("equalities", self.equalities),
                ("inequalities", self.inequalities),
                ("proves-infeasibility", self.proves_infeasibility),
            )
            if value
        ]
        return ", ".join(flags)


class SolverBackend(ABC):
    """A solver that accepts a declarative problem and returns a status and a point."""

    name: str
    capability: BackendCapability

    @abstractmethod
    def solve(self, problem: RealisationProblem) -> SolveResult:
        """Solve the problem.

        A backend returns [`INFEASIBLE`][qsimod.solving.result.SolveStatus.INFEASIBLE] only if
        its `BackendCapability.proves_infeasibility` is set; a failed search is reported as
        ``UNSOLVED``.
        """

    def supports(self, problem: RealisationProblem) -> bool:
        """Whether the backend handles the shape of the problem."""
        if problem.system.has_integral_unknown and not self.capability.integral:
            return False
        if problem.system.equalities and not self.capability.equalities:
            return False
        return not (problem.hard_constraints() and not self.capability.inequalities)

    def __str__(self) -> str:
        return f"{self.name} [{self.capability}]"
