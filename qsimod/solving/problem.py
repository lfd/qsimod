"""The declarative statement of a parameter-realisation problem.

[`RealisationProblem`][qsimod.solving.problem.RealisationProblem] collects the equations,
admissible sets and validity conditions of a parameter relation into a
[`ConstraintSystem`][qsimod.solving.problem.ConstraintSystem]; the backend that solves it is
selected in [`qsimod.solving.dispatch`][qsimod.solving.dispatch], and this module imports no
solver.  [`Unknown`][qsimod.solving.problem.Unknown] records whether an unknown ranges over the
reals or the integers, and a backend declares the problem shapes it handles through
[`supports`][qsimod.solving.backends.base.SolverBackend.supports].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from qsimod.parameters import AdmissibleSet, Bound, InequalityConstraint
from qsimod.relations import (
    Definition,
    Equation,
    ParameterRelation,
    RelationClassification,
    residuals_of,
)
from qsimod.scalar import Interval, Scalar
from qsimod.solving.objectives import FeasibilityOnly, Objective
from qsimod.validity import Conjunction, Severity, ValidityCondition

__all__ = [
    "MARGIN_SLACK_DECADES",
    "ConstraintSystem",
    "RealisationProblem",
    "Unknown",
    "UnknownKind",
    "build_system",
]


#: The distance below zero, in decades, by which an enforced validity margin may fall and still
#: count as satisfied; it absorbs the rounding of a constraint the optimiser holds active at its
#: boundary.
MARGIN_SLACK_DECADES = 1e-8


class UnknownKind(Enum):
    """The domain of an unknown: the reals or the integers."""

    CONTINUOUS = "continuous"
    INTEGRAL = "integral"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Unknown:
    """One quantity determined by a solve.

    Attributes:
        name: the parameter name.
        kind: whether the unknown is continuous or integral.
        bound: the box bound, if the admissible set declares one.
        initial: a deterministic starting value, if one is declared.

    """

    name: str
    kind: UnknownKind = UnknownKind.CONTINUOUS
    bound: Bound | None = None
    initial: float | None = None

    @property
    def interval(self) -> Interval:
        """The box of the unknown; unbounded if the admissible set declares none."""
        return self.bound.interval if self.bound else Interval(float("-inf"), float("inf"))

    def __str__(self) -> str:
        box = f" in {self.bound}" if self.bound else " unbounded"
        return f"{self.name} ({self.kind}){box}"


@dataclass(frozen=True)
class ConstraintSystem:
    """The constraint system: equations, inequalities and boxes over named unknowns.

    Attributes:
        unknowns: the quantities the solve determines.
        equalities: the equations whose residual must vanish.
        inequalities: the expressions that must be non-negative.
        fixed: the parameters fixed to values: the requested targets and any prescribed knobs.
        relation: the parameter relation the equalities are taken from, if any.
        classification: the static classification of the relation in the requested direction.

    """

    unknowns: tuple[Unknown, ...]
    equalities: tuple[Equation, ...] = ()
    inequalities: tuple[InequalityConstraint, ...] = ()
    fixed: Mapping[str, float] = field(default_factory=dict)
    relation: ParameterRelation | None = None
    classification: RelationClassification | None = None

    @property
    def unknown_names(self) -> tuple[str, ...]:
        """The names of the unknowns, in declaration order."""
        return tuple(unknown.name for unknown in self.unknowns)

    @property
    def has_integral_unknown(self) -> bool:
        """Whether any unknown ranges over the integers."""
        return any(unknown.kind is UnknownKind.INTEGRAL for unknown in self.unknowns)

    def unknown(self, name: str) -> Unknown:
        """The unknown with the given name.

        Raises:
            KeyError: if no unknown has that name.

        """
        for candidate in self.unknowns:
            if candidate.name == name:
                return candidate
        msg = f"{name!r} is not an unknown of this system; have {self.unknown_names}"
        raise KeyError(msg)

    def bounds(self) -> tuple[Bound, ...]:
        """The declared box bounds of the unknowns, in declaration order."""
        return tuple(unknown.bound for unknown in self.unknowns if unknown.bound is not None)

    def admissible_set(self, description: str = "") -> AdmissibleSet:
        """The boxes of the unknowns and the coupled constraints, as an admissible set."""
        return AdmissibleSet(
            bounds=self.bounds(),
            constraints=self.inequalities,
            description=description,
        )

    def interval_environment(self) -> dict[str, Interval]:
        """The boxes of the unknowns and the fixed values, as an interval environment."""
        environment = {name: Interval(value, value) for name, value in self.fixed.items()}
        for unknown in self.unknowns:
            environment[unknown.name] = unknown.interval
        return environment

    def residuals(self, assignment: Mapping[str, float]) -> dict[str, float]:
        """The residual of every equality whose parameters are all assigned."""
        return residuals_of(self.equalities, assignment)

    def definitions(self) -> tuple[Definition, ...]:
        """The solved forms of the relation; empty if the system has no relation."""
        return () if self.relation is None else self.relation.definitions

    def __str__(self) -> str:
        lines = [
            f"constraint system: {len(self.unknowns)} unknown(s), "
            f"{len(self.equalities)} equation(s), {len(self.inequalities)} inequality(ies)"
        ]
        lines += [f"  unknown  {unknown}" for unknown in self.unknowns]
        lines += [f"  equation {eq}" for eq in self.equalities]
        lines += [f"  requires {constraint}" for constraint in self.inequalities]
        if self.fixed:
            pinned = ", ".join(f"{k}={v:.6g}" for k, v in sorted(self.fixed.items()))
            lines.append(f"  fixed: {pinned}")
        if self.classification is not None:
            lines.append(f"  classification: {self.classification}")
        return "\n".join(lines)


@dataclass(frozen=True)
class RealisationProblem:
    """A complete parameter-realisation request, independent of the backend that solves it.

    Attributes:
        name: the name of the request, used in reports.
        system: the constraint system.
        objective: the objective of the solve; defaults to feasibility only.
        validity: the validity conditions; the domain conditions are always hard constraints.
        enforce_regime_conditions: whether the regime conditions are imposed as constraints in
            addition to being reported (default ``True``).
        tolerance: the largest residual that counts as an exact solution.
        max_iterations: the iteration budget of the backend; an exhausted budget yields
            ``UNSOLVED``, never ``INFEASIBLE``.

    """

    name: str
    system: ConstraintSystem
    objective: Objective = field(default_factory=FeasibilityOnly)
    validity: Conjunction = field(default_factory=Conjunction)
    enforce_regime_conditions: bool = True
    tolerance: float = 1e-10
    max_iterations: int = 400

    def hard_constraints(self) -> tuple[tuple[str, Scalar], ...]:
        """Every named expression that a solution must keep non-negative.

        The tuple comprises the coupled admissibility constraints, the margin surrogates of the
        domain conditions, the margin surrogates of the regime conditions when these are
        enforced, and the epigraph constraints of the objective.
        """
        constraints: list[tuple[str, Scalar]] = [
            (constraint.name, constraint.expression) for constraint in self.system.inequalities
        ]
        for condition in self.enforced_conditions():
            surrogate = condition.margin_expression()
            if surrogate is not None:
                constraints.append((f"validity: {condition.name}", surrogate))
        constraints.extend(self.objective.epigraph_constraints())
        return tuple(constraints)

    def enforced_conditions(self) -> tuple[ValidityCondition, ...]:
        """The validity conditions a solution must satisfy.

        The domain conditions are always included; the regime conditions are included when
        ``enforce_regime_conditions`` is set.
        """
        return tuple(
            condition
            for condition in self.validity
            if condition.severity is not Severity.REGIME or self.enforce_regime_conditions
        )

    def rejections(self, point: Mapping[str, float]) -> tuple[str, ...]:
        """The reasons for which ``point`` is not an acceptable solution.

        The tuple lists every admissibility constraint the point violates and every enforced
        validity condition that fails at the point by more than `MARGIN_SLACK_DECADES`, the
        rounding permitted to a constraint active at its boundary.  It is empty exactly when
        the point may be reported as a solution.
        """
        reasons = [str(violation) for violation in self.system.admissible_set().violations(point)]
        for condition in self.enforced_conditions():
            outcome = condition.evaluate(point)
            if outcome.status.is_failure and not outcome.margin >= -MARGIN_SLACK_DECADES:
                reasons.append(f"validity: {outcome}")
        return tuple(reasons)

    def parameters(self) -> frozenset[str]:
        """Every parameter the problem refers to."""
        names = set(self.system.unknown_names) | set(self.system.fixed)
        for eq in self.system.equalities:
            names |= eq.parameters()
        for constraint in self.system.inequalities:
            names |= constraint.expression.symbols()
        names |= self.validity.parameters()
        names |= self.objective.parameters()
        return frozenset(names)

    def __str__(self) -> str:
        return "\n".join(
            [
                f"realisation problem {self.name!r}",
                *(f"  {line}" for line in str(self.system).splitlines()),
                f"  objective: {self.objective}",
                f"  validity: {self.validity}",
                f"  regime conditions enforced: {self.enforce_regime_conditions}",
                f"  tolerance {self.tolerance:g}, iteration budget {self.max_iterations}",
            ]
        )


def build_system(
    relation: ParameterRelation,
    targets: Mapping[str, float],
    unknowns: Sequence[str],
    admissible_set: AdmissibleSet,
    *,
    initial: Mapping[str, float] | None = None,
    integral: Sequence[str] = (),
) -> ConstraintSystem:
    """Assemble a constraint system from a relation, a target assignment and an admissible set.

    Args:
        relation: the parameter relation, possibly the composite relation of a pipeline.
        targets: the parameters fixed by the request.
        unknowns: the parameters to solve for.  Every parameter the relation refers to that is
            neither fixed nor listed becomes an unknown as well.
        admissible_set: the declared knob limits of the hardware model.
        initial: deterministic starting values, per unknown.
        integral: the unknowns that range over the integers.

    Returns:
        The constraint system, with its static classification.

    """
    mentioned = set(relation.parameters())
    listed = list(dict.fromkeys(unknowns))
    implied = sorted(mentioned - set(targets) - set(listed))
    names = tuple(listed + implied)
    initial = initial or {}
    system_unknowns = tuple(
        Unknown(
            name=name,
            kind=UnknownKind.INTEGRAL if name in set(integral) else UnknownKind.CONTINUOUS,
            bound=admissible_set.bound_for(name),
            initial=initial.get(name),
        )
        for name in names
    )
    classification = relation.classify(known=frozenset(targets), unknowns=set(names))
    return ConstraintSystem(
        unknowns=system_unknowns,
        equalities=tuple(
            eq for eq in relation.equations if eq.parameters() & (set(names) | set(targets))
        ),
        inequalities=admissible_set.constraints,
        fixed=dict(targets),
        relation=relation,
        classification=classification,
    )
