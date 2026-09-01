"""Validity-window checks of the perturbative step: regime violations and domain errors."""

from __future__ import annotations

import math
from contextlib import suppress

import pytest

from qsimod.scalar import DomainError
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import coupling_map, validity
from qsimod.validity import ConditionStatus, Severity, ValidityReport
from tests.conftest import ELECTRIC_GAP, forward_map, window_point


def _report(knobs: dict[str, float]) -> ValidityReport:
    """The validity report of the perturbative step at a knob setting."""
    point = dict(knobs)
    # At a pole the effective parameters are undefined; evaluate on the knobs alone.
    with suppress(DomainError):
        point.update(forward_map(knobs))
    point[P.ELECTRIC_GAP] = ELECTRIC_GAP
    return validity().report(point)


def test_a_point_inside_the_window_passes_with_margins_listed() -> None:
    """A point inside the window is valid and every condition is reported with a margin."""
    report = _report(window_point())
    assert report.is_valid, str(report)
    assert len(report.outcomes) == 10
    for outcome in report.outcomes:
        assert outcome.status is ConditionStatus.VALID
        assert outcome.quantity
        assert outcome.requirement
        assert math.isfinite(outcome.value)
        assert outcome.margin >= 0.0
    names = {outcome.name for outcome in report.outcomes}
    assert {"J << delta", "J << U", "Delta << delta", "Delta << U", "kappa << Delta"} <= names
    weakest = report.weakest()
    assert weakest is not None
    assert weakest.severity is Severity.REGIME


def test_tunnelling_equal_to_the_interaction_fails_and_names_the_condition() -> None:
    """J = U violates ``J << U`` and ``J << delta`` with a margin of -1 decade."""
    knobs = window_point()
    knobs[P.TUNNELLING] = knobs[P.INTERACTION]
    report = _report(knobs)
    assert not report.is_valid
    violated = {outcome.name for outcome in report.regime_violations}
    assert "J << U" in violated
    assert "J << delta" in violated
    outcome = next(o for o in report.outcomes if o.name == "J << U")
    assert outcome.value == pytest.approx(1.0)
    # Margin is in decades relative to the threshold of 0.1.
    assert outcome.margin == pytest.approx(-1.0)
    assert not report.has_domain_error


def test_tilt_equal_to_the_superlattice_depth_is_a_domain_error() -> None:
    """Delta = delta is reported as a domain error, not a regime violation."""
    knobs = window_point()
    knobs[P.TILT] = knobs[P.SUPERLATTICE]
    report = _report(knobs)
    assert report.has_domain_error
    domain = {outcome.name for outcome in report.domain_errors}
    assert "pole: delta != +Delta" in domain
    for outcome in report.domain_errors:
        assert outcome.severity is Severity.DOMAIN
        assert outcome.status is ConditionStatus.DOMAIN_ERROR


def test_the_coupling_formula_refuses_to_produce_a_finite_looking_value_at_a_pole() -> None:
    """At Delta = delta the coupling map raises ``DomainError``."""
    knobs = window_point()
    knobs[P.TILT] = knobs[P.SUPERLATTICE]
    with pytest.raises(DomainError) as caught:
        coupling_map().evaluate_real(knobs)
    assert "division by zero" in str(caught.value)


def test_a_vanishing_tilt_fails_the_coupling_condition() -> None:
    """Delta = 0 violates ``kappa << Delta`` with an infinite value and margin."""
    knobs = window_point()
    knobs[P.TILT] = 0.0
    report = _report(knobs)
    assert not report.is_valid
    violated = {outcome.name for outcome in report.regime_violations}
    assert "kappa << Delta" in violated
    outcome = next(o for o in report.outcomes if o.name == "kappa << Delta")
    assert math.isinf(outcome.value)
    assert outcome.margin == -math.inf


def test_thresholds_are_overridable() -> None:
    """The much-less-than threshold can be overridden."""
    knobs = window_point()
    knobs[P.TUNNELLING] = 0.15 * knobs[P.INTERACTION]
    point = {**knobs, **forward_map(knobs), P.ELECTRIC_GAP: ELECTRIC_GAP}
    strict = validity().report(point)
    lenient = validity(much_less_threshold=0.3).report(point)
    assert not strict.is_valid
    assert "J << U" in {outcome.name for outcome in strict.regime_violations}
    assert lenient.is_valid or "J << U" not in {
        outcome.name for outcome in lenient.regime_violations
    }
