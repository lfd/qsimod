"""Selection and execution of a solver backend.

[`solve`][qsimod.solving.dispatch.solve] first attempts to prove infeasibility by interval
arithmetic ([`qsimod.solving.reachability`][qsimod.solving.reachability]); if nothing is
proved, it runs the first backend whose `supports` accepts the problem.  The default order is
[`ClosedFormBackend`][qsimod.solving.backends.closed_form.ClosedFormBackend],
[`MonotoneBisectionBackend`][qsimod.solving.backends.integer.MonotoneBisectionBackend],
[`ScipyNlpBackend`][qsimod.solving.backends.scipy_nlp.ScipyNlpBackend].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qsimod.parameters import AdmissibleSet
from qsimod.pipeline import Pipeline
from qsimod.relations import ParameterRelation, RelationClassification, RelationKind
from qsimod.solving.backends.base import SolverBackend
from qsimod.solving.backends.closed_form import ClosedFormBackend
from qsimod.solving.backends.integer import MonotoneBisectionBackend
from qsimod.solving.backends.scipy_nlp import ScipyNlpBackend
from qsimod.solving.objectives import Objective, default_objective
from qsimod.solving.problem import ConstraintSystem, RealisationProblem, build_system
from qsimod.solving.reachability import ReachabilityVerdict, check_reachability
from qsimod.solving.result import SolveResult, SolveStatus
from qsimod.validity import Conjunction, ValidityReport

__all__ = [
    "classify_request",
    "default_backends",
    "problem_for",
    "realise_parameters",
    "solve",
]


def default_backends() -> tuple[SolverBackend, ...]:
    """The backends tried, in order, when none are supplied."""
    return (ClosedFormBackend(), MonotoneBisectionBackend(), ScipyNlpBackend())


def solve(
    problem: RealisationProblem,
    backends: Sequence[SolverBackend] | None = None,
) -> SolveResult:
    """Solve a realisation problem, after an attempt to prove it infeasible.

    Args:
        problem: the declarative problem.
        backends: the backends to try, in order.  Defaults to
            [`default_backends`][qsimod.solving.dispatch.default_backends].

    Returns:
        The solve result.

    Raises:
        ValueError: if no supplied backend supports the shape of the problem.

    """
    verdict = _prove_infeasible(problem)
    if verdict is not None:
        return _infeasible_result(problem, verdict)

    candidates = tuple(backends) if backends is not None else default_backends()
    for backend in candidates:
        if backend.supports(problem):
            return backend.solve(problem)
    shapes = ", ".join(str(backend) for backend in candidates)
    msg = (
        f"no backend supports {problem.name!r}: it has "
        f"{len(problem.system.unknowns)} unknown(s), "
        f"{len(problem.system.equalities)} equation(s) and "
        f"{'an' if problem.system.has_integral_unknown else 'no'} integral unknown. "
        f"Backends offered: {shapes}"
    )
    raise ValueError(msg)


def _prove_infeasible(problem: RealisationProblem) -> ReachabilityVerdict | None:
    """Attempt to prove the request unreachable.

    A ``None`` result establishes only that infeasibility is not proved, not that the request
    is reachable.  The attempt is omitted for an ``OVER_DETERMINED`` request, for which no
    exact solution is expected.
    """
    classification = problem.system.classification
    if classification is not None and classification.kind is RelationKind.OVER_DETERMINED:
        return None
    return check_reachability(problem.system, problem.system.admissible_set())


def _infeasible_result(
    problem: RealisationProblem,
    verdict: ReachabilityVerdict,
) -> SolveResult:
    """The result of a proved infeasibility.

    The point is empty, there are no residuals, and the binding constraints of the verdict are
    reported.
    """
    return SolveResult(
        status=SolveStatus.INFEASIBLE,
        point={},
        residuals={},
        relation_kind=_kind_of(problem.system.classification),
        validity=ValidityReport(()),
        binding_constraints=verdict.binding,
        objective_name=problem.objective.name,
        backend="interval arithmetic over the admissible knob set",
        iterations=0,
        termination=verdict.note,
        seed=None,
        notes={
            "proof": str(verdict),
            "eliminated": "; ".join(verdict.eliminated) or "nothing",
        },
    )


def _kind_of(classification: RelationClassification | None) -> RelationKind:
    """The kind of the classification, or ``IMPLICIT`` when there is no classification."""
    return RelationKind.IMPLICIT if classification is None else classification.kind


def classify_request(
    relation: ParameterRelation,
    targets: Mapping[str, float],
) -> RelationClassification:
    """Classify a relation in the inverse direction.

    Every parameter that ``targets`` does not fix is treated as an unknown.
    """
    return relation.classify(known=frozenset(targets))


def realise_parameters(
    pipeline: Pipeline,
    targets: Mapping[str, float],
    unknowns: Sequence[str],
    admissible_set: AdmissibleSet,
    *,
    objective: Objective | None = None,
    validity: Conjunction | None = None,
    initial: Mapping[str, float] | None = None,
    integral: Sequence[str] = (),
    enforce_regime_conditions: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 400,
    backends: Sequence[SolverBackend] | None = None,
    name: str = "",
) -> SolveResult:
    """Solve a pipeline for the knob settings of the hardware model that realise a target.

    The composite parameter relation is solved once, over the full pipeline: the intermediate
    parameters are unknowns alongside the hardware knobs.

    Args:
        pipeline: the pipeline whose composite relation is inverted.
        targets: the parameters the request fixes.
        unknowns: the hardware knobs to solve for.  The intermediate parameters are added as
            unknowns automatically.
        admissible_set: the declared knob limits of the hardware model.
        objective: the objective that resolves under-determination; defaults to
            [`MaximiseWeakestMargin`][qsimod.solving.objectives.MaximiseWeakestMargin]
            over the validity conditions of the pipeline.
        validity: the conditions to enforce and report; defaults to those of the pipeline.
        initial: deterministic starting values.
        integral: the unknowns that range over the integers.
        enforce_regime_conditions: whether the regime conditions are imposed as constraints
            in addition to being reported.
        tolerance: the largest residual that counts as an exact solution.
        max_iterations: the iteration budget of the backend.
        backends: the backends to try; defaults to
            [`default_backends`][qsimod.solving.dispatch.default_backends].
        name: the name of the request.

    Returns:
        The solve result.

    """
    conditions = pipeline.validity if validity is None else validity
    system = build_system(
        relation=pipeline.relation,
        targets=targets,
        unknowns=unknowns,
        admissible_set=admissible_set,
        initial=initial,
        integral=integral,
    )
    problem = RealisationProblem(
        name=name or f"realise {pipeline.name}",
        system=system,
        objective=objective or default_objective(tuple(conditions)),
        validity=conditions,
        enforce_regime_conditions=enforce_regime_conditions,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return solve(problem, backends)


def problem_for(
    relation: ParameterRelation,
    targets: Mapping[str, float],
    unknowns: Sequence[str],
    admissible_set: AdmissibleSet,
    *,
    objective: Objective | None = None,
    validity: Conjunction | None = None,
    initial: Mapping[str, float] | None = None,
    integral: Sequence[str] = (),
    tolerance: float = 1e-10,
    max_iterations: int = 400,
    name: str = "realisation",
) -> RealisationProblem:
    """Assemble a realisation problem without solving it.

    Returns a [`RealisationProblem`][qsimod.solving.problem.RealisationProblem], for instance
    for the inspection of the static classification or the construction of a
    [`FeasibleSet`][qsimod.solving.feasible.FeasibleSet].
    """
    conditions = validity if validity is not None else Conjunction()
    system: ConstraintSystem = build_system(
        relation=relation,
        targets=targets,
        unknowns=unknowns,
        admissible_set=admissible_set,
        initial=initial,
        integral=integral,
    )
    return RealisationProblem(
        name=name,
        system=system,
        objective=objective or default_objective(tuple(conditions)),
        validity=conditions,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
