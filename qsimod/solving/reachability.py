"""Proving a request unreachable for a hardware model by interval arithmetic.

The proof involves no numerical solve.  The procedure is as follows: (1) every solved form whose
inputs are fixed values or bounded knobs is substituted, repeatedly, until no further parameter
can be eliminated; (2) for each remaining equation with one side thereby constant, the other
side is bounded over the admissible box by the intersection of the interval-arithmetic bound and
the [`qsimod.affine`][qsimod.affine] bound; (3) if the constant lies outside the bound, the
request is proved unreachable; (4) the binding constraints are named by relaxing the knob boxes
one at a time.  A bound that contains the request establishes only that infeasibility is not
proved, not that the request is reachable:
[`check_reachability`][qsimod.solving.reachability.check_reachability] then returns ``None`` and
the caller proceeds to a backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from qsimod import affine
from qsimod.parameters import AdmissibleSet
from qsimod.scalar import Const, Interval, Scalar
from qsimod.solving.problem import ConstraintSystem

__all__ = [
    "ReachabilityReport",
    "ReachabilityVerdict",
    "achievable_range",
    "bound_over",
    "check_reachability",
    "reachability_report",
]


@dataclass(frozen=True)
class ReachabilityVerdict:
    """A proof that a request cannot be met, naming the constraint responsible.

    Attributes:
        equation: the equation that cannot be satisfied.
        requested: the value the request imposes on one side of the equation.
        achievable: the interval that side attains over the admissible box.
        binding: the constraints whose relaxation would make the request reachable.
        note: an explanation in prose.
        eliminated: the solved forms substituted in the course of the proof, for the report.

    """

    equation: str
    requested: float
    achievable: Interval
    binding: tuple[str, ...]
    note: str
    eliminated: tuple[str, ...] = ()

    def __str__(self) -> str:
        return (
            f"{self.equation}: request needs {self.requested:.6g}, but the admissible knob "
            f"set can only reach {self.achievable}; "
            f"binding: {', '.join(self.binding) or 'the box itself'}"
        )


def _substituted_system(
    system: ConstraintSystem,
    exclude_equation: str | None = None,
) -> tuple[dict[str, Scalar], tuple[str, ...]]:
    """Eliminate solved forms, leaving ``exclude_equation`` unused so that it can be tested.

    Each equation is used at most once, and the equation under test is never used.  The
    elimination of a bounded knob replaces its box by an exact relation, which is a relaxation;
    the resulting bound therefore remains sound.
    """
    known = set(system.fixed)
    bounded = {unknown.name for unknown in system.unknowns if unknown.interval.is_bounded}
    substitutions: dict[str, Scalar] = {name: Const(value) for name, value in system.fixed.items()}
    used: set[str] = set() if exclude_equation is None else {exclude_equation}
    eliminated: list[str] = []
    progress = True
    while progress:
        progress = False
        for definition in system.definitions():
            if definition.parameter in substitutions or definition.parameter in system.fixed:
                continue
            if definition.equation in used:
                continue
            if not definition.inputs() <= known | bounded | set(substitutions):
                continue
            substitutions[definition.parameter] = definition.expression.substitute(substitutions)
            used.add(definition.equation)
            eliminated.append(f"{definition.parameter} := {definition.expression}")
            progress = True
    return substitutions, tuple(eliminated)


def achievable_range(
    expression: Scalar,
    system: ConstraintSystem,
    substitutions: Mapping[str, Scalar] | None = None,
) -> Interval:
    """Bound ``expression`` over the admissible box, after optional substitutions.

    The result is the intersection of the interval-arithmetic bound and the affine bound; it is
    sound and never wider than either.
    """
    if substitutions:
        expression = expression.substitute(substitutions)
    return bound_over(expression, system.interval_environment())


def bound_over(expression: Scalar, env: Mapping[str, Interval]) -> Interval:
    """The tightest sound bound the package forms for ``expression``.

    Args:
        expression: the expression to bound.
        env: an interval per symbol.

    Returns:
        The interval-arithmetic bound, narrowed by the affine bound where the latter exists
        and is narrower.

    """
    interval = expression.interval(env)
    refined = affine.bound(expression, env)
    if refined is None:
        return interval
    low, high = max(interval.lo, refined.lo), min(interval.hi, refined.hi)
    if low > high:  # pragma: no cover - two sound bounds intersect, but floats are floats
        return interval
    return Interval(low, high)


def check_reachability(
    system: ConstraintSystem,
    admissible_set: AdmissibleSet,
) -> ReachabilityVerdict | None:
    """Attempt to prove the request unreachable over the declared admissible set.

    Args:
        system: the constraint system.
        admissible_set: the knob limits of the hardware model, used to name the binding
            constraint.

    Returns:
        A verdict if unreachability is proved, otherwise ``None``.  A ``None`` result
        establishes only that infeasibility is not proved, not that the request is reachable.

    """
    for equation in system.equalities:
        substitutions, eliminated = _substituted_system(system, exclude_equation=equation.name)
        sides = ((equation.left, equation.right), (equation.right, equation.left))
        for known_side, other_side in sides:
            requested = _as_constant(known_side, substitutions)
            if requested is None:
                continue
            achievable = achievable_range(other_side, system, substitutions)
            if not achievable.is_bounded:
                continue
            if achievable.contains(requested, tolerance=1e-12 * max(1.0, abs(requested))):
                continue
            binding = _binding_constraints(
                other_side, system, substitutions, requested, admissible_set
            )
            direction = "above" if requested > achievable.hi else "below"
            return ReachabilityVerdict(
                equation=equation.name,
                requested=requested,
                achievable=achievable,
                binding=binding,
                note=(
                    f"the requested value lies {direction} everything the admissible knob "
                    f"set can produce, proved by interval arithmetic after eliminating "
                    f"{len(eliminated)} solved form(s)"
                ),
                eliminated=eliminated,
            )
    return None


def _as_constant(side: Scalar, substitutions: Mapping[str, Scalar]) -> float | None:
    """The value of ``side`` if the substitutions make it constant, otherwise ``None``."""
    substituted = side.substitute(substitutions)
    if substituted.symbols():
        return None
    try:
        return substituted.evaluate_real({})
    except (ZeroDivisionError, ValueError):
        return None


def _binding_constraints(
    expression: Scalar,
    system: ConstraintSystem,
    substitutions: Mapping[str, Scalar],
    requested: float,
    admissible_set: AdmissibleSet,
) -> tuple[str, ...]:
    """The boxes whose relaxation, one at a time, brings the request within reach.

    Each box is widened one-sidedly by `_RELAXATION_FACTOR` spans.  If any box exists whose
    relaxation leaves the achievable range bounded and containing the request, only such boxes
    are reported.
    """
    substituted = expression.substitute(substitutions)
    relevant = sorted(substituted.symbols())
    base = system.interval_environment()
    decisive: list[str] = []
    contributing: list[str] = []
    for name in relevant:
        box = base.get(name)
        if box is None or not box.is_bounded:
            continue
        for widened_box in _one_sided_widenings(box):
            relaxed = dict(base)
            relaxed[name] = widened_box
            widened = bound_over(substituted, relaxed)
            if not widened.contains(requested):
                continue
            bucket = decisive if widened.is_bounded else contributing
            bucket.append(_render_bound(name, admissible_set))
            break
    if decisive:
        return tuple(dict.fromkeys(decisive))
    if contributing:
        return tuple(dict.fromkeys(contributing))
    return tuple(
        _render_bound(name, admissible_set)
        for name in relevant
        if admissible_set.bound_for(name) is not None
    )


#: The distance by which a one-sided relaxation moves a box endpoint, as a multiple of its span.
_RELAXATION_FACTOR = 1000.0


def _one_sided_widenings(box: Interval) -> tuple[Interval, ...]:
    """The box widened upwards, then downwards, by `_RELAXATION_FACTOR` spans."""
    span = max(box.hi - box.lo, abs(box.hi), abs(box.lo), 1.0)
    reach = _RELAXATION_FACTOR * span
    return (Interval(box.lo, box.hi + reach), Interval(box.lo - reach, box.hi))


def _render_bound(name: str, admissible_set: AdmissibleSet) -> str:
    """The declared bound on ``name``, or a placeholder that names the knob."""
    bound = admissible_set.bound_for(name)
    return str(bound) if bound is not None else f"the box on {name}"


@dataclass(frozen=True)
class ReachabilityReport:
    """The achievable range of every equation, for a report or a feasibility study.

    Attributes:
        ranges: the interval the unknown side of each equation attains over the admissible box.
        eliminated: the solved forms that were substituted.

    """

    ranges: Mapping[str, Interval] = field(default_factory=dict)
    eliminated: tuple[str, ...] = ()

    def __str__(self) -> str:
        lines = ["achievable ranges over the admissible knob set:"]
        lines += [f"  {name}: {interval}" for name, interval in sorted(self.ranges.items())]
        if self.eliminated:
            lines.append("  after eliminating: " + "; ".join(self.eliminated))
        return "\n".join(lines)


def reachability_report(system: ConstraintSystem) -> ReachabilityReport:
    """Bound the unknown side of every equation over the admissible box."""
    ranges: dict[str, Interval] = {}
    all_eliminated: list[str] = []
    for equation in system.equalities:
        substitutions, eliminated = _substituted_system(system, exclude_equation=equation.name)
        for name in eliminated:
            if name not in all_eliminated:
                all_eliminated.append(name)
        for side in (equation.left, equation.right):
            if _as_constant(side, substitutions) is not None:
                continue
            ranges[equation.name] = achievable_range(side, system, substitutions)
            break
    return ReachabilityReport(ranges, tuple(all_eliminated))
