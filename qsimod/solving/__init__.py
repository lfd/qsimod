"""The solve for the knob settings: parameter realisation as a constrained solve.

The parameter relation of a pipeline is inverted in a single constrained solve for the knob
settings of the hardware model.  [`qsimod.solving.problem`][qsimod.solving.problem] collects the
equations, admissible sets and validity conditions of a relation into a constraint system;
[`qsimod.solving.backends`][qsimod.solving.backends] contains the only modules that import an
optimiser; [`qsimod.solving.dispatch`][qsimod.solving.dispatch] selects a backend after an
attempt to prove infeasibility with
[`qsimod.solving.reachability`][qsimod.solving.reachability]; and
[`qsimod.solving.feasible`][qsimod.solving.feasible] evaluates the same declarations point-wise
and region-wise.  [`qsimod.solving.stepcount`][qsimod.solving.stepcount] poses the step count of
a product formula as an integer solve against the a-priori error bound.
"""

from qsimod.solving.backends import (
    BackendCapability,
    ClosedFormBackend,
    DiscreteSolution,
    MonotoneBisectionBackend,
    ResourceChoice,
    ScipyNlpBackend,
    SolverBackend,
    minimal_feasible_integer,
    minimal_resource_setting,
)
from qsimod.solving.dispatch import (
    classify_request,
    default_backends,
    problem_for,
    realise_parameters,
    solve,
)
from qsimod.solving.feasible import FeasiblePoint, FeasibleSet
from qsimod.solving.objectives import (
    FeasibilityOnly,
    MaximiseExpression,
    MaximiseWeakestMargin,
    MinimiseExpression,
    Objective,
    default_objective,
    maximise,
    minimise,
)
from qsimod.solving.problem import (
    ConstraintSystem,
    RealisationProblem,
    Unknown,
    UnknownKind,
    build_system,
)
from qsimod.solving.reachability import (
    ReachabilityReport,
    ReachabilityVerdict,
    check_reachability,
    reachability_report,
)
from qsimod.solving.result import SolveResult, SolveStatus
from qsimod.solving.stepcount import (
    STEPS_SYMBOL,
    bound_coefficient,
    minimal_steps,
    resource_candidates,
    step_count_problem,
)

__all__ = [
    "STEPS_SYMBOL",
    "BackendCapability",
    "ClosedFormBackend",
    "ConstraintSystem",
    "DiscreteSolution",
    "FeasibilityOnly",
    "FeasiblePoint",
    "FeasibleSet",
    "MaximiseExpression",
    "MaximiseWeakestMargin",
    "MinimiseExpression",
    "MonotoneBisectionBackend",
    "Objective",
    "ReachabilityReport",
    "ReachabilityVerdict",
    "RealisationProblem",
    "ResourceChoice",
    "ScipyNlpBackend",
    "SolveResult",
    "SolveStatus",
    "SolverBackend",
    "Unknown",
    "UnknownKind",
    "bound_coefficient",
    "build_system",
    "check_reachability",
    "classify_request",
    "default_backends",
    "default_objective",
    "maximise",
    "minimal_feasible_integer",
    "minimal_resource_setting",
    "minimal_steps",
    "minimise",
    "problem_for",
    "reachability_report",
    "realise_parameters",
    "resource_candidates",
    "solve",
    "step_count_problem",
]
