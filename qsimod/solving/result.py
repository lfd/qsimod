"""The result of a parameter-realisation solve: a status and a point.

[`SolveStatus`][qsimod.solving.result.SolveStatus] distinguishes an exact solution, a best fit
with a residual per equation, a proved infeasibility that names the binding constraint, and an
unconverged search (``UNSOLVED``), which is not an infeasibility.  The backend, its termination
message, its iteration count and any random seed are recorded; every backend of the package is
deterministic, and the seed is therefore ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from qsimod.parameters import ConstraintViolation
from qsimod.relations import RelationKind
from qsimod.validity import ValidityReport

__all__ = ["SolveResult", "SolveStatus"]


class SolveStatus(Enum):
    """The outcome of a solve."""

    EXACT_SOLUTION = "EXACT_SOLUTION"
    """A point that satisfies every equation to the requested tolerance."""

    APPROXIMATE_SOLUTION = "APPROXIMATE_SOLUTION"
    """A best fit; the residual of each equation is reported and does not vanish."""

    INFEASIBLE = "INFEASIBLE"
    """A proof that no admissible point realises the request, naming the binding constraint."""

    UNSOLVED = "UNSOLVED"
    """The solver did not converge; nothing was proved, in contrast to ``INFEASIBLE``."""

    @property
    def is_success(self) -> bool:
        """Whether the solve produced a point that may be reported as a solution."""
        return self in {SolveStatus.EXACT_SOLUTION, SolveStatus.APPROXIMATE_SOLUTION}

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SolveResult:
    """The result of a solve.

    Attributes:
        status: the outcome of the solve.
        point: the full parameter assignment, including the intermediate parameters; empty
            for a request proved infeasible.
        residuals: the residual of each equation at ``point``.
        relation_kind: the static classification of the relation in the requested
            direction, determined before the solve.
        validity: the validity report at ``point``.
        binding_constraints: for an infeasible request, the constraints that make it
            infeasible; for an approximate solution, the constraints that are active.
        violations: the admissibility constraints ``point`` violates; empty for a successful
            status.
        objective_name: the name of the objective that was optimised.
        objective_value: the natural value of the objective at ``point``: the expression for
            a minimisation or maximisation, the least normalised margin for the default
            max-min objective.
        backend: the name of the backend used.
        iterations: the number of iterations the backend took.
        termination: the termination message of the backend.
        seed: the random seed used; ``None`` for every backend of the package.
        notes: further diagnostics for a report.

    """

    status: SolveStatus
    point: Mapping[str, float] = field(default_factory=dict)
    residuals: Mapping[str, float] = field(default_factory=dict)
    relation_kind: RelationKind = RelationKind.IMPLICIT
    validity: ValidityReport = field(default_factory=lambda: ValidityReport(()))
    binding_constraints: tuple[str, ...] = ()
    violations: tuple[ConstraintViolation, ...] = ()
    objective_name: str = ""
    objective_value: float = 0.0
    backend: str = ""
    iterations: int = 0
    termination: str = ""
    seed: int | None = None
    notes: Mapping[str, str] = field(default_factory=dict)

    @property
    def max_residual(self) -> float:
        """The largest absolute residual, or ``0.0`` if there are no residuals."""
        return max((abs(value) for value in self.residuals.values()), default=0.0)

    def value(self, parameter: str) -> float:
        """The solved value of one parameter.

        Raises:
            KeyError: if the parameter is not part of the solution.

        """
        try:
            return self.point[parameter]
        except KeyError:
            msg = f"{parameter!r} is not in the solution; have {sorted(self.point)}"
            raise KeyError(msg) from None

    def subset(self, parameters: Sequence[str]) -> dict[str, float]:
        """The solved values of a named subset of parameters, omitting any that are absent."""
        return {name: self.point[name] for name in parameters if name in self.point}

    def __str__(self) -> str:
        lines = [
            f"solve: {self.status}  [{self.backend}, {self.iterations} iteration(s), "
            f"seed {self.seed}]",
            f"  relation kind (static): {self.relation_kind}",
            f"  termination: {self.termination}",
        ]
        if self.point:
            values = ", ".join(f"{k}={v:.6g}" for k, v in sorted(self.point.items()))
            lines.append(f"  point: {values}")
        if self.residuals:
            residuals = ", ".join(f"{k}={v:+.3e}" for k, v in sorted(self.residuals.items()))
            lines.append(f"  residuals: {residuals}  (max {self.max_residual:.3e})")
        if self.objective_name:
            lines.append(f"  objective {self.objective_name} = {self.objective_value:.6g}")
        if self.binding_constraints:
            lines.append("  binding constraint(s):")
            lines += [f"    - {name}" for name in self.binding_constraints]
        if self.violations:
            lines.append("  admissibility violations:")
            lines += [f"    - {violation}" for violation in self.violations]
        if self.validity.outcomes:
            lines += [f"  {line}" for line in str(self.validity).splitlines()]
        for key, note in sorted(self.notes.items()):
            lines.append(f"  {key}: {note}")
        return "\n".join(lines)
