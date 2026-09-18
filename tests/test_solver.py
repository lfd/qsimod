"""Parameter realisation as a constrained solve."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from examples.infeasible_request import tightened_device

from qsimod.parameters import AdmissibleSet, Bound, Parameter, ParameterSet
from qsimod.pipeline import Pipeline
from qsimod.relations import Definition, ParameterRelation, RelationKind, equation
from qsimod.scalar import Symbol
from qsimod.solving import (
    ClosedFormBackend,
    FeasibleSet,
    MaximiseWeakestMargin,
    Objective,
    SolveResult,
    SolveStatus,
    check_reachability,
    minimise,
    problem_for,
    realise_parameters,
    solve,
)
from qsimod.solving.backends.base import BackendCapability, SolverBackend
from qsimod.solving.backends.scipy_nlp import _PoleLog, _value_and_grad
from qsimod.solving.problem import RealisationProblem
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import (
    SchwingerGraph,
    build_graph,
    device_limits,
)
from qsimod.validity import MuchLessThan
from tests.conftest import (
    ELECTRIC_GAP,
    WINDOW_INTERACTION,
    WINDOW_MASS,
    forward_map,
    window_point,
)

KNOBS = (P.TUNNELLING, P.INTERACTION, P.SUPERLATTICE, P.TILT)
TARGET_COUPLING = 0.004525
TARGETS = {
    P.MASS_LATTICE_QED: WINDOW_MASS,
    P.COUPLING_QUANTUM_LINK_STAGGERED: TARGET_COUPLING,
    P.ELECTRIC_GAP: ELECTRIC_GAP,
}


def _solve(
    graph: SchwingerGraph,
    *,
    admissible_set: AdmissibleSet | None = None,
    objective: Objective | None = None,
    max_iterations: int = 400,
) -> SolveResult:
    """Solve the analogue pipeline for the hardware knobs, with optional overrides."""
    return realise_parameters(
        graph.analogue,
        targets=TARGETS,
        unknowns=list(KNOBS),
        admissible_set=device_limits() if admissible_set is None else admissible_set,
        objective=objective,
        max_iterations=max_iterations,
    )


def test_a_target_is_solved_for_with_margins_reported(graph3: SchwingerGraph) -> None:
    """A reachable target gives an exact point with margins reported at that point."""
    result = _solve(graph3)
    assert result.status is SolveStatus.EXACT_SOLUTION
    assert result.max_residual < 1e-10
    assert result.relation_kind is RelationKind.UNDER_DETERMINED
    assert set(KNOBS) <= set(result.point)
    assert not result.violations
    assert result.validity.is_valid, str(result.validity)
    assert len(result.validity.outcomes) == 11
    assert result.backend.startswith("scipy.optimize")
    assert result.iterations > 0


def test_infeasibility_is_proved_and_the_binding_constraint_is_named(
    graph3: SchwingerGraph,
) -> None:
    """An unreachable coupling gives ``INFEASIBLE``, names the cap and returns no point."""
    result = _solve(graph3, admissible_set=tightened_device())
    assert result.status is SolveStatus.INFEASIBLE
    assert result.binding_constraints
    assert any(P.TUNNELLING in constraint for constraint in result.binding_constraints)
    assert result.point == {}
    assert "interval arithmetic" in result.backend
    assert result.iterations == 0


def test_unsolved_and_infeasible_are_distinguishable(graph3: SchwingerGraph) -> None:
    """A budget too small to converge gives ``UNSOLVED``, not ``INFEASIBLE``."""
    result = _solve(graph3, max_iterations=1)
    assert result.status is SolveStatus.UNSOLVED
    assert result.max_residual > 1e-10
    assert _solve(graph3).status is SolveStatus.EXACT_SOLUTION


def test_the_objective_is_honoured(graph3: SchwingerGraph) -> None:
    """Two objectives give two different points, both satisfying the relation."""
    default = _solve(graph3)
    minimal_interaction = _solve(
        graph3,
        objective=minimise(Symbol(P.INTERACTION), name="minimise U"),
    )
    assert default.status.is_success
    assert minimal_interaction.status.is_success
    assert default.max_residual < 1e-10
    assert minimal_interaction.max_residual < 1e-10
    assert default.point[P.INTERACTION] != pytest.approx(minimal_interaction.point[P.INTERACTION])
    assert minimal_interaction.point[P.INTERACTION] < default.point[P.INTERACTION]
    assert isinstance(
        realise_parameters(
            graph3.analogue,
            targets=TARGETS,
            unknowns=list(KNOBS),
            admissible_set=device_limits(),
        ).validity.outcomes,
        tuple,
    )


def test_the_default_objective_equalises_the_binding_margins(graph3: SchwingerGraph) -> None:
    """The default objective is ``MaximiseWeakestMargin`` and equalises the binding margins."""
    result = _solve(graph3)
    assert isinstance(
        problem_for(
            graph3.analogue.relation,
            targets=TARGETS,
            unknowns=list(KNOBS),
            admissible_set=device_limits(),
            validity=graph3.analogue.validity,
        ).objective,
        MaximiseWeakestMargin,
    )
    margins = sorted(
        outcome.margin for outcome in result.validity.outcomes if outcome.severity.value == "regime"
    )
    # At least two regime conditions bind at the optimum.
    assert margins[0] == pytest.approx(margins[1], abs=1e-6)


def test_the_same_request_solved_twice_gives_the_same_point(graph3: SchwingerGraph) -> None:
    """Solving the same request twice gives the same point, with no seed involved."""
    first = _solve(graph3)
    second = _solve(graph3)
    assert first.point == second.point
    assert first.seed is None
    assert second.seed is None
    assert first.iterations == second.iterations


# ---------------------------------------------------------------------------
# Over-determination
# ---------------------------------------------------------------------------

TOY_KNOB = "toy.g"
TOY_TARGET_A = "toy.a"
TOY_TARGET_B = "toy.b"


def _over_determined_relation() -> ParameterRelation:
    """A toy relation with one knob and two target parameters."""
    knob = Symbol(TOY_KNOB)
    return ParameterRelation(
        name="toy over-determined relation",
        equations=(
            equation("a from g", Symbol(TOY_TARGET_A), knob, "a = g"),
            equation("b from g", Symbol(TOY_TARGET_B), 2 * knob, "b = 2g"),
        ),
        definitions=(
            Definition(TOY_TARGET_A, knob, "a from g"),
            Definition(TOY_TARGET_B, 2 * knob, "b from g"),
            Definition(TOY_KNOB, Symbol(TOY_TARGET_A), "a from g"),
        ),
    )


def _over_determined_problem() -> RealisationProblem:
    """The toy problem: one knob, two incompatible targets."""
    return problem_for(
        _over_determined_relation(),
        targets={TOY_TARGET_A: 1.0, TOY_TARGET_B: 5.0},
        unknowns=[TOY_KNOB],
        admissible_set=AdmissibleSet(
            bounds=(Bound(TOY_KNOB, -10.0, 10.0),),
            description="a toy device with one knob",
        ),
        name="toy over-determined realisation",
    )


def test_over_determination_is_detected_statically(graph3: SchwingerGraph) -> None:
    """``OVER_DETERMINED`` is classified from the relation before any solve."""
    problem = _over_determined_problem()
    classification = problem.system.classification
    assert classification is not None
    assert classification.kind is RelationKind.OVER_DETERMINED
    assert classification.degrees_of_freedom < 0

    # On the real pipeline, with knobs pinned as targets.
    pinned = {**TARGETS, P.TUNNELLING: 0.02, P.TILT: 0.05, P.INTERACTION: 1.0, P.SUPERLATTICE: 0.52}
    pinned_classification = graph3.analogue.classify_relation(frozenset(pinned))
    assert pinned_classification.kind is RelationKind.OVER_DETERMINED
    assert not pinned_classification.kind.can_be_exact


def test_over_determination_carries_a_separate_residual() -> None:
    """An over-determined request gives ``APPROXIMATE_SOLUTION`` with per-target residuals."""
    result = solve(_over_determined_problem())
    assert result.status is SolveStatus.APPROXIMATE_SOLUTION
    assert result.relation_kind is RelationKind.OVER_DETERMINED
    assert set(result.residuals) == {"a from g", "b from g"}
    assert result.max_residual > 1e-6
    assert result.point[TOY_KNOB] == pytest.approx(2.2, abs=1e-6)


def test_the_residual_is_reported_apart_from_the_other_error_axes(
    graph3: SchwingerGraph,
) -> None:
    """Structural, solver-residual and regime errors are reported in separate sections."""
    solved = _solve(graph3)
    report = graph3.analogue.error_report(
        solved.point, solver_residuals={"a from g": 0.4, "b from g": -0.8}
    )
    assert report.max_solver_residual == pytest.approx(0.8)
    assert len(report.structural) == 2
    assert report.regime.is_valid
    rendered = str(report)
    assert "structural approximation:" in rendered
    assert "solver residual:" in rendered
    assert "regime margins:" in rendered
    assert report.solver_residuals["b from g"] == pytest.approx(-0.8)


# ---------------------------------------------------------------------------
# The cheap path stays cheap
# ---------------------------------------------------------------------------


class ExplodingBackend(SolverBackend):
    """A backend that raises when asked to solve."""

    def __init__(self) -> None:
        self.name = "exploding backend"
        self.capability = BackendCapability(
            continuous=True, integral=True, equalities=True, inequalities=True
        )

    def solve(self, problem: RealisationProblem) -> SolveResult:
        """Raise unconditionally.

        Raises:
            AssertionError: always.

        """
        message = f"an iterative solver was called for {problem.name!r}, but should not have been"
        raise AssertionError(message)


def test_a_closed_form_direction_never_reaches_an_iterative_solver() -> None:
    """A closed-form direction is solved without invoking an iterative backend."""
    relation = build_graph(3).analogue.relation
    knobs = {
        P.TUNNELLING: 0.02,
        P.INTERACTION: WINDOW_INTERACTION,
        P.SUPERLATTICE: WINDOW_MASS + WINDOW_INTERACTION / 2,
        P.TILT: 0.048,
    }
    problem = problem_for(
        relation,
        targets=knobs,
        unknowns=[P.MASS_LATTICE_QED],
        admissible_set=device_limits(),
        name="derived direction",
    )
    assert problem.system.classification is not None
    assert problem.system.classification.kind is RelationKind.CLOSED_FORM

    result = solve(problem, backends=[ClosedFormBackend(), ExplodingBackend()])
    assert result.status is SolveStatus.EXACT_SOLUTION
    assert result.backend == "closed-form evaluation"
    assert result.iterations == 0
    assert result.point[P.MASS_LATTICE_QED] == pytest.approx(WINDOW_MASS)


def test_an_under_determined_direction_does_need_a_solver(graph3: SchwingerGraph) -> None:
    """An under-determined direction is not accepted by the closed-form backend."""
    problem = problem_for(
        graph3.analogue.relation,
        targets=TARGETS,
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        validity=graph3.analogue.validity,
        name="user-facing direction",
    )
    assert not ClosedFormBackend().supports(problem)
    with pytest.raises(AssertionError, match="an iterative solver was called"):
        solve(problem, backends=[ClosedFormBackend(), ExplodingBackend()])


# ---------------------------------------------------------------------------
# The feasible set and the single-point check are one facility
# ---------------------------------------------------------------------------


def test_the_grid_and_the_single_point_checker_agree(graph3: SchwingerGraph) -> None:
    """A feasible grid node is accepted by ``check`` and an infeasible one is rejected."""
    problem = problem_for(
        graph3.analogue.relation,
        targets={P.MASS_LATTICE_QED: WINDOW_MASS, P.ELECTRIC_GAP: ELECTRIC_GAP},
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        validity=graph3.analogue.validity,
        name="the validity window of the analogue branch",
    )
    checker = FeasibleSet(problem)

    def complete(node: Mapping[str, float]) -> dict[str, float]:
        superlattice = WINDOW_MASS + WINDOW_INTERACTION / 2
        knobs = window_point(
            tunnelling_ratio=node["J_over_U"], tilt=node["Delta_over_delta"] * superlattice
        )
        return {**node, **knobs, **forward_map(knobs)}

    verdicts = checker.grid(
        {"J_over_U": (1 / 10, 1 / 80), "Delta_over_delta": (1 / 5, 1 / 20, 1 / 50)}, complete
    )
    assert len(verdicts) == 6
    assert any(point.feasible for point in verdicts), "the grid holds no feasible node at all"
    assert not all(point.feasible for point in verdicts), "the grid holds no infeasible node"

    for expected in (True, False):
        point = next(point for point in verdicts if point.feasible is expected)
        assignment = {name: point.values[name] for name in problem.parameters()}
        verdict = checker.check(assignment)
        assert verdict.feasible is expected, verdict.reasons


def test_the_feasible_set_can_also_be_sampled_and_exported(graph3: SchwingerGraph) -> None:
    """The feasible set exposes its constraint system and samples deterministically."""
    problem = problem_for(
        graph3.analogue.relation,
        targets={P.MASS_LATTICE_QED: WINDOW_MASS, P.ELECTRIC_GAP: ELECTRIC_GAP},
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        validity=graph3.analogue.validity,
    )
    feasible = FeasibleSet(problem)
    system = feasible.constraint_system()
    assert system.equalities
    assert system.unknown_names

    # Sampling needs either a box on every unknown, or a completer for the intermediates.
    with pytest.raises(ValueError, match="have no finite box"):
        feasible.sample(4)
    with pytest.raises(ValueError, match="no unknown has a finite box"):
        FeasibleSet(
            problem_for(
                graph3.analogue.relation,
                targets={P.MASS_LATTICE_QED: WINDOW_MASS},
                unknowns=list(KNOBS),
                admissible_set=AdmissibleSet(description="no bounds declared"),
            )
        ).sample(4)

    forward = graph3.analogue.relation.classify(known=frozenset(KNOBS))
    elimination = forward.elimination
    assert elimination is not None

    def complete(node: Mapping[str, float]) -> Mapping[str, float]:
        knobs = {name: node[name] for name in KNOBS}
        derived = graph3.analogue.relation.evaluate_closed_form(elimination, knobs)
        return {**node, **derived}

    samples = feasible.sample(12, complete=complete, seed=7)
    assert len(samples) == 12
    assert all(isinstance(sample.feasible, bool) for sample in samples)
    assert all(sample.max_residual < 1e-9 for sample in samples)
    # Deterministic for a fixed seed.
    assert [s.values for s in samples] == [
        s.values for s in feasible.sample(12, complete=complete, seed=7)
    ]


def test_reachability_makes_no_claim_when_the_box_admits_a_pole(graph3: SchwingerGraph) -> None:
    """Reachability returns no verdict when the box admits a pole."""
    problem = problem_for(
        graph3.analogue.relation,
        targets=TARGETS,
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        validity=graph3.analogue.validity,
    )
    verdict = check_reachability(problem.system, device_limits())
    assert verdict is None


# ---------------------------------------------------------------------------
# The solver seam
# ---------------------------------------------------------------------------

#: Modules that declare physics and must not import a solver backend.
PHYSICS_MODULES = (
    "qsimod/models",
    "qsimod/transformations",
    "qsimod/usecases",
    "qsimod/artifact.py",
    "qsimod/symbolic.py",
    "qsimod/structure.py",
    "qsimod/relations.py",
    "qsimod/validity.py",
    "qsimod/parameters.py",
    "qsimod/transform.py",
    "qsimod/pipeline.py",
    "qsimod/scalar.py",
    "qsimod/units.py",
    "qsimod/levels.py",
)

#: Imports the physics modules may not make directly.
FORBIDDEN_IMPORTS = (
    "qsimod.solving",
    "scipy",
    "z3",
    "cvxpy",
    "nlopt",
)


def _python_files(target: str) -> list[Path]:
    """Every ``.py`` file under a path relative to the repository root."""
    root = Path(__file__).resolve().parent.parent
    path = root / target
    return sorted(path.rglob("*.py")) if path.is_dir() else [path]


def _imported_modules(source: Path) -> set[str]:
    """The module names a file imports directly, from its AST."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_no_physics_module_imports_a_solver_backend() -> None:
    """No physics module imports a solver backend."""
    offenders: list[str] = []
    for target in PHYSICS_MODULES:
        for source in _python_files(target):
            for imported in _imported_modules(source):
                if any(
                    imported == forbidden or imported.startswith(f"{forbidden}.")
                    for forbidden in FORBIDDEN_IMPORTS
                ):
                    offenders.append(f"{source.name} imports {imported}")
    assert not offenders, "physics modules must not import a solver: " + "; ".join(offenders)


def test_the_solver_layer_is_the_only_place_scipy_appears() -> None:
    """Only ``solving/backends/scipy_nlp.py`` imports scipy."""
    root = Path(__file__).resolve().parent.parent / "qsimod"
    importers = {
        source.relative_to(root).as_posix()
        for source in root.rglob("*.py")
        if "scipy" in _imported_modules(source)
        or any(name.startswith("scipy.") for name in _imported_modules(source))
    }
    assert importers == {"solving/backends/scipy_nlp.py"}, importers


def test_a_second_backend_needs_only_the_backend_interface() -> None:
    """A backend implementing only ``SolverBackend`` is usable by ``solve``."""

    class ConstantBackend(SolverBackend):
        """A stand-in backend returning box midpoints."""

        def __init__(self) -> None:
            self.name = "constant backend (test)"
            self.capability = BackendCapability()

        def solve(self, problem: RealisationProblem) -> SolveResult:
            """Return the midpoint of every box, unsolved."""
            point = {
                unknown.name: 0.5 * (unknown.interval.lo + unknown.interval.hi)
                for unknown in problem.system.unknowns
                if unknown.interval.is_bounded
            }
            return SolveResult(
                status=SolveStatus.UNSOLVED,
                point={**problem.system.fixed, **point},
                backend=self.name,
                termination="returned the box midpoint without solving anything",
            )

    problem = problem_for(
        _over_determined_relation(),
        targets={TOY_TARGET_A: 1.0, TOY_TARGET_B: 5.0},
        unknowns=[TOY_KNOB],
        admissible_set=AdmissibleSet(bounds=(Bound(TOY_KNOB, -4.0, 8.0),)),
    )
    result = solve(problem, backends=[ConstantBackend()])
    assert result.backend == "constant backend (test)"
    assert result.point[TOY_KNOB] == pytest.approx(2.0)


def test_the_composite_is_solved_once_over_the_whole_pipeline(graph3: SchwingerGraph) -> None:
    """Intermediate parameters are returned solved alongside the knobs."""
    result = _solve(graph3)
    for name in (
        P.MASS_QUANTUM_LINK_STAGGERED,
        P.MASS_QUANTUM_LINK_HOMOGENEOUS,
        P.MASS_EFFECTIVE_BOSONIC,
        P.COUPLING_QUANTUM_LINK_HOMOGENEOUS,
        P.COUPLING_EFFECTIVE_BOSONIC,
    ):
        assert name in result.point
    assert result.point[P.COUPLING_EFFECTIVE_BOSONIC] == pytest.approx(TARGET_COUPLING, rel=1e-9)
    assert isinstance(Pipeline.of(list(graph3.analogue.steps)), Pipeline)


def test_a_parameter_set_is_a_parameter_set() -> None:
    """Duplicate parameter names are refused."""
    with pytest.raises(ValueError, match="duplicate parameter names"):
        ParameterSet((Parameter("x"), Parameter("x")))


# ---------------------------------------------------------------------------
# A reported solution is one the problem accepts
# ---------------------------------------------------------------------------

OUT_OF_REGIME_TARGETS = {**TARGETS, P.MASS_LATTICE_QED: 0.5}


def test_an_out_of_regime_target_is_not_reported_as_a_solution(graph3: SchwingerGraph) -> None:
    """With regime conditions enforced, a point outside the window is ``UNSOLVED``."""
    result = realise_parameters(
        graph3.analogue,
        targets=OUT_OF_REGIME_TARGETS,
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
    )
    assert result.status is SolveStatus.UNSOLVED
    assert not result.validity.is_valid
    assert "not an acceptable solution" in result.termination
    assert result.validity.regime_violations

    reported_only = realise_parameters(
        graph3.analogue,
        targets=OUT_OF_REGIME_TARGETS,
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        enforce_regime_conditions=False,
    )
    assert reported_only.status.is_success
    assert not reported_only.validity.is_valid


def test_the_closed_form_backend_rejects_a_point_outside_the_admissible_set() -> None:
    """A closed-form point that violates a bound is ``UNSOLVED`` with the violation named."""
    knobs = {
        P.TUNNELLING: 0.02,
        P.INTERACTION: WINDOW_INTERACTION,
        P.SUPERLATTICE: WINDOW_MASS + WINDOW_INTERACTION / 2,
        P.TILT: 0.048,
    }
    problem = problem_for(
        build_graph(3).analogue.relation,
        targets=knobs,
        unknowns=[P.MASS_LATTICE_QED],
        admissible_set=device_limits().union(
            AdmissibleSet(
                bounds=(Bound(P.MASS_LATTICE_QED, 0.1, 1.0),), description="a heavy theory"
            )
        ),
        name="derived direction, mass bounded away from the answer",
    )
    # The dispatcher proves the request out of reach before any backend runs.
    assert solve(problem, backends=[ClosedFormBackend()]).status is SolveStatus.INFEASIBLE
    # The backend on its own evaluates the chain, then refuses to call the point a solution.
    result = ClosedFormBackend().solve(problem)
    assert result.status is SolveStatus.UNSOLVED
    assert result.point[P.MASS_LATTICE_QED] == pytest.approx(WINDOW_MASS)
    assert result.violations
    assert P.MASS_LATTICE_QED in result.violations[0].constraint
    assert "not an acceptable solution" in result.termination


def test_the_reported_objective_value_is_the_quantity_the_objective_names(
    graph3: SchwingerGraph,
) -> None:
    """``objective_value`` is the maximised margin, or the expression, not a negated cost."""
    default = _solve(graph3)
    assert isinstance(default.objective_value, float)
    assert default.objective_value > 0.0
    normalised = [
        condition.threshold
        - abs(condition.numerator.evaluate_real(default.point))
        / abs(condition.denominator.evaluate_real(default.point))
        for condition in graph3.analogue.validity
        if isinstance(condition, MuchLessThan)
    ]
    assert default.objective_value == pytest.approx(min(normalised))
    assert default.validity.weakest_margin > 0.0

    minimal_interaction = _solve(
        graph3, objective=minimise(Symbol(P.INTERACTION), name="minimise U")
    )
    assert minimal_interaction.objective_value == pytest.approx(
        minimal_interaction.point[P.INTERACTION]
    )


def test_an_objective_must_say_what_the_backend_minimises() -> None:
    """A subclass without ``cost_expression`` cannot be instantiated: no silent constant."""

    class Incomplete(Objective):
        name = "incomplete"
        description = ""

        def value(self, assignment: Mapping[str, float]) -> float:
            return assignment.get(TOY_KNOB, 0.0)

        def parameters(self) -> frozenset[str]:
            return frozenset({TOY_KNOB})

    with pytest.raises(TypeError, match="cost_expression"):
        Incomplete()  # ty: ignore[call-non-callable]


def test_a_pole_is_a_violated_constraint_not_a_satisfied_one() -> None:
    """A traced evaluation at a pole yields the caller's sentinel and a zero gradient."""

    def reciprocal(vector: jnp.ndarray) -> jnp.ndarray:
        return 1.0 / vector[0]

    log = _PoleLog()
    at_zero = np.asarray([0.0], dtype=np.float64)
    value, gradient = _value_and_grad(reciprocal, at_pole=-1e18, log=log)(at_zero)
    assert not math.isfinite(value) or value == -1e18
    assert log.count == 1
    assert "pole" in (log.note() or "")
    value, gradient = _value_and_grad(reciprocal, at_pole=1e18, log=log)(np.asarray([2.0]))
    assert value == pytest.approx(0.5)
    assert gradient.tolist() == pytest.approx([-0.25])
    assert log.count == 1
