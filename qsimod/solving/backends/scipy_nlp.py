"""The constrained non-linear-programming backend on ``scipy.optimize``.

The default method is SLSQP (sequential least-squares programming) from SciPy.  The box bounds
are passed to the solver directly.  The parameter relations enter as equality constraints; the
hard inequalities, that is the coupled admissibility constraints and the domain and regime
conditions, and the epigraph objective enter as inequality constraints.  Each constraint is a
`NonlinearConstraint` object with an analytic Jacobian obtained by JAX automatic differentiation
of the expression layer.  An ``OVER_DETERMINED`` problem is solved as a least-squares fit of the
residuals.  The method is local and never returns ``INFEASIBLE``: a failed search is reported as
``UNSOLVED``, and infeasibility is proved separately by
[`qsimod.solving.reachability`][qsimod.solving.reachability].
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from qsimod.jax_setup import require_x64
from qsimod.parameters import ConstraintViolation
from qsimod.relations import RelationKind
from qsimod.scalar import DomainError, Scalar
from qsimod.solving.backends.base import BackendCapability, SolverBackend
from qsimod.solving.problem import RealisationProblem, UnknownKind
from qsimod.solving.result import SolveResult, SolveStatus

__all__ = ["NlpMethod", "ScipyNlpBackend"]

#: The ``scipy.optimize.minimize`` methods that support equality and inequality constraints
#: together with box bounds.
NlpMethod: TypeAlias = Literal["SLSQP", "trust-constr"]

#: A real optimisation vector, in the one-dimensional shape used by the ``scipy`` type stubs.
Vector: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]

#: A constraint Jacobian: one row per constraint component, one column per variable.
Matrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]

#: A value-and-gradient callable over `Vector`.
Graded: TypeAlias = Callable[[Vector], tuple[float, Vector]]

#: The upper bound that makes a `NonlinearConstraint` an inequality.
_UNBOUNDED_ABOVE = np.inf

#: A large finite box for the epigraph variable.
_SLACK_BOX = (-1e3, 1e3)


class ScipyNlpBackend(SolverBackend):
    """SLSQP over the admissible box, with equalities, inequalities and an epigraph objective.

    Attributes:
        method: the ``scipy.optimize.minimize`` method used.

    """

    def __init__(self, method: NlpMethod = "SLSQP") -> None:
        self.name = f"scipy.optimize.minimize({method})"
        self.method = method
        self.capability = BackendCapability(
            continuous=True,
            integral=False,
            equalities=True,
            inequalities=True,
            proves_infeasibility=False,
        )

    def solve(self, problem: RealisationProblem) -> SolveResult:
        """Run the constrained optimisation and classify the outcome."""
        require_x64()
        variables = _variable_names(problem)
        start: Vector = np.asarray(
            [_initial_value(problem, name) for name in variables], dtype=np.float64
        )
        bounds = [_box(problem, name) for name in variables]

        least_squares = (
            problem.system.classification is not None
            and problem.system.classification.kind is RelationKind.OVER_DETERMINED
        )
        log = _PoleLog()
        objective = _objective_function(problem, variables, least_squares=least_squares, log=log)
        constraints = _constraints(problem, variables, least_squares=least_squares, log=log)

        outcome = minimize(
            objective,
            start,
            method=self.method,
            jac=True,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": problem.max_iterations, "ftol": 1e-12},
        )
        point = _assignment(problem, variables, np.asarray(outcome.x, dtype=np.float64))
        residuals = problem.system.residuals(point)
        largest = max((abs(value) for value in residuals.values()), default=0.0)
        validity = problem.validity.report(point)
        violations = _violations(problem, point)
        rejections = problem.rejections(point)
        converged = bool(outcome.success)

        status = _classify(
            converged=converged,
            largest_residual=largest,
            tolerance=problem.tolerance,
            least_squares=least_squares,
            rejected=bool(rejections),
        )
        termination = str(outcome.message)
        if rejections:
            termination += "; the point is not an acceptable solution: " + "; ".join(rejections)
        notes = {
            "start": ", ".join(
                f"{name}={value:.6g}" for name, value in zip(variables, start, strict=True)
            ),
            "fit": "least squares (over-determined)" if least_squares else "equality constrained",
        }
        pole_note = log.note()
        if pole_note is not None:
            notes["poles"] = pole_note
        return SolveResult(
            status=status,
            point=point,
            residuals=residuals,
            relation_kind=(
                problem.system.classification.kind
                if problem.system.classification is not None
                else RelationKind.IMPLICIT
            ),
            validity=validity,
            binding_constraints=tuple(violation.constraint for violation in violations),
            violations=violations,
            objective_name=problem.objective.name,
            objective_value=problem.objective.value(point),
            backend=self.name,
            iterations=int(getattr(outcome, "nit", 0)),
            termination=termination,
            seed=None,
            notes=notes,
        )


def _classify(
    *,
    converged: bool,
    largest_residual: float,
    tolerance: float,
    least_squares: bool,
    rejected: bool,
) -> SolveStatus:
    """Map the outcome of the optimiser onto a [`SolveStatus`][qsimod.solving.result.SolveStatus].

    ``rejected`` states that the point violates an admissibility constraint or an enforced
    validity condition; such a point is never reported as a solution.  The function never
    returns ``INFEASIBLE``.
    """
    if rejected:
        return SolveStatus.UNSOLVED
    if largest_residual <= tolerance:
        # The point satisfies every equation regardless of the optimiser's convergence flag.
        return SolveStatus.EXACT_SOLUTION
    if least_squares and converged:
        return SolveStatus.APPROXIMATE_SOLUTION
    return SolveStatus.UNSOLVED


def _variable_names(problem: RealisationProblem) -> tuple[str, ...]:
    """The optimisation variables: the continuous unknowns and any epigraph variable."""
    names = [
        unknown.name
        for unknown in problem.system.unknowns
        if unknown.kind is UnknownKind.CONTINUOUS
    ]
    slack = problem.objective.slack_variable
    if slack is not None:
        names.append(slack)
    return tuple(names)


def _initial_value(problem: RealisationProblem, name: str) -> float:
    """A deterministic starting value: the declared one, otherwise the interior of the box."""
    if name == problem.objective.slack_variable:
        return 0.0
    unknown = problem.system.unknown(name)
    if unknown.initial is not None:
        return unknown.initial
    interval = unknown.interval
    if interval.is_bounded:
        return 0.5 * (interval.lo + interval.hi)
    if math.isfinite(interval.lo):
        return interval.lo + 1.0
    if math.isfinite(interval.hi):
        return interval.hi - 1.0
    return 1.0


def _box(problem: RealisationProblem, name: str) -> tuple[float | None, float | None]:
    """The box of one variable, in the ``(low, high)`` form of ``scipy``."""
    if name == problem.objective.slack_variable:
        return _SLACK_BOX
    interval = problem.system.unknown(name).interval
    low = interval.lo if math.isfinite(interval.lo) else None
    high = interval.hi if math.isfinite(interval.hi) else None
    return low, high


def _assignment(
    problem: RealisationProblem,
    variables: Sequence[str],
    values: Vector,
) -> dict[str, float]:
    """The full parameter assignment implied by an optimisation vector."""
    point = dict(problem.system.fixed)
    for name, value in zip(variables, values, strict=True):
        if name == problem.objective.slack_variable:
            continue
        point[name] = float(value)
    return point


def _environment(
    problem: RealisationProblem,
    variables: Sequence[str],
    vector: jax.Array,
) -> dict[str, jax.Array | float]:
    """A JAX-traceable environment for expression evaluation."""
    environment: dict[str, jax.Array | float] = dict(problem.system.fixed)
    for index, name in enumerate(variables):
        environment[name] = vector[index]
    return environment


def _traced(
    problem: RealisationProblem,
    variables: Sequence[str],
    expression: Scalar,
) -> Callable[[jax.Array], jax.Array]:
    """Wrap a scalar expression as a differentiable function of the variable vector."""

    def evaluate(vector: jax.Array) -> jax.Array:
        return jnp.real(expression.evaluate(_environment(problem, variables, vector)))

    return evaluate


#: The value reported for an expression whose evaluation raises at a pole.  An objective takes
#: ``+_AT_POLE``, a very high cost; an equality residual takes ``+_AT_POLE`` and an inequality
#: ``-_AT_POLE``, both violated, so that a pole is never mistaken for a satisfied constraint.
_AT_POLE = 1e18


class _PoleLog:
    """A count of the evaluations at which an expression was non-finite or raised."""

    def __init__(self) -> None:
        self.count = 0

    def note(self) -> str | None:
        if not self.count:
            return None
        return (
            f"the search evaluated an expression at a pole {self.count} time(s); a non-finite "
            "value was handed to the optimiser, which terminates on it"
        )


def _value_and_grad(
    function: Callable[[jax.Array], jax.Array], *, at_pole: float, log: _PoleLog
) -> Graded:
    """Convert a traceable function into the ``(value, gradient)`` callable of ``scipy``.

    A [`DomainError`][qsimod.scalar.DomainError], raised only on a concrete zero denominator,
    becomes ``at_pole`` with a zero gradient.  Under tracing a vanishing denominator yields
    ``inf`` or ``nan`` instead; the value is passed through, which terminates the optimiser,
    and counted in ``log`` so that the result reports it.
    """
    # Compiled once per callable: SLSQP evaluates every constraint at every iteration.
    graded = jax.jit(jax.value_and_grad(function))

    def wrapped(vector: Vector) -> tuple[float, Vector]:
        try:
            value, gradient = graded(jnp.asarray(vector, dtype=float))
        except DomainError:
            log.count += 1
            return at_pole, np.zeros_like(vector)
        scalar = float(value)
        row = np.asarray(gradient, dtype=np.float64)
        if not math.isfinite(scalar) or not np.all(np.isfinite(row)):
            log.count += 1
        return scalar, row

    return wrapped


def _objective_function(
    problem: RealisationProblem,
    variables: Sequence[str],
    *,
    least_squares: bool,
    log: _PoleLog,
) -> Graded:
    """The cost to minimise, with its gradient.

    A least-squares fit minimises the sum of the squared residuals; otherwise the
    [`cost_expression`][qsimod.solving.objectives.Objective.cost_expression] of the objective
    is minimised, or a constant when the objective declares no preference.
    """
    if least_squares:
        residuals = [equation.residual for equation in problem.system.equalities]

        def cost(vector: jax.Array) -> jax.Array:
            environment = _environment(problem, variables, vector)
            total = jnp.asarray(0.0)
            for residual in residuals:
                total = total + jnp.real(residual.evaluate(environment)) ** 2
            return total

        return _value_and_grad(cost, at_pole=_AT_POLE, log=log)

    expression = problem.objective.cost_expression()
    if expression is not None:
        return _value_and_grad(_traced(problem, variables, expression), at_pole=_AT_POLE, log=log)

    def constant(vector: jax.Array) -> jax.Array:
        return jnp.sum(vector) * 0.0

    return _value_and_grad(constant, at_pole=_AT_POLE, log=log)


def _constraints(
    problem: RealisationProblem,
    variables: Sequence[str],
    *,
    least_squares: bool,
    log: _PoleLog,
) -> list[NonlinearConstraint]:
    """The equality and inequality constraints, as ``NonlinearConstraint`` objects.

    An equality is posed as ``lb = ub = 0``, an inequality as ``lb = 0, ub = inf``; each
    carries its analytic Jacobian.
    """
    constraints: list[NonlinearConstraint] = []
    if not least_squares:
        for equation in problem.system.equalities:
            wrapped = _value_and_grad(
                _traced(problem, variables, equation.residual), at_pole=_AT_POLE, log=log
            )
            constraints.append(
                NonlinearConstraint(_only_value(wrapped), 0.0, 0.0, jac=_only_gradient(wrapped))
            )
    for _, expression in problem.hard_constraints():
        wrapped = _value_and_grad(
            _traced(problem, variables, expression), at_pole=-_AT_POLE, log=log
        )
        constraints.append(
            NonlinearConstraint(
                _only_value(wrapped),
                0.0,
                _UNBOUNDED_ABOVE,
                jac=_only_gradient(wrapped),
            )
        )
    return constraints


def _only_value(wrapped: Graded) -> Callable[[Vector], Vector]:
    """The value component of a ``(value, gradient)`` callable, as a vector of length one."""

    def value(vector: Vector) -> Vector:
        return np.asarray([wrapped(vector)[0]], dtype=np.float64)

    return value


def _only_gradient(wrapped: Graded) -> Callable[[Vector], Matrix]:
    """The gradient component, in the shape ``(1, n)`` of the Jacobian of a scalar constraint."""

    def gradient(vector: Vector) -> Matrix:
        row: Matrix = np.asarray(wrapped(vector)[1], dtype=np.float64).reshape(1, -1)
        return row

    return gradient


def _violations(
    problem: RealisationProblem,
    point: Mapping[str, float],
) -> tuple[ConstraintViolation, ...]:
    """The admissibility violations at ``point``."""
    return tuple(problem.system.admissible_set().violations(point))
