"""Reference data of Zhou et al. (2022) and the numerical checks of the analogue branch.

The request of the experiment, its prescribed knobs, the shared numerical realisation in which
a knob setting is evaluated, and the reproductions of two of its figures: the oscillation fit
of Fig. S3a and the gauge-violation scan of Fig. 2D.  ``examples/analogue_end_to_end.py``
uses the request and the realisation; the test suite and the scripts under ``scripts/`` use
the rest.  Z.-Y. Zhou et al., *Thermalization dynamics of a gauge theory on a quantum
simulator*, Science **377**, 311-314 (2022).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.optimize import brentq, curve_fit

from qsimod.models.gauge import ElectricField, gauss_operators
from qsimod.parameters import AdmissibleSet
from qsimod.realise import (
    HilbertSpace,
    RealisationRequest,
    SectorBasis,
    build_operator,
    build_operator_in,
    evolve_state,
    expectation,
    local_subspace_projector,
    sandwich,
)
from qsimod.symbolic import OperatorSum
from qsimod.units import from_hertz, to_hertz
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import (
    canonical_state_configuration,
    coupling_map,
    device_limits,
    effective_bosonic,
    gauge_violation_observable,
    local_occupation_subspace,
    mass_map,
    matter_occupation_observable,
    superlattice,
    validity,
)

# ---------------------------------------------------------------------------
# The request, and the experiment that already answered it
# ---------------------------------------------------------------------------

#: The request at the application layer: a massless fermion at the resonance ``U = 2 delta``
#: and the coupling of the quench data, "kappa = 14.5 Hz and m = 0" (main text).
TARGET_MASS_HZ = 0.0
TARGET_COUPLING_HZ = 14.5

#: The electric-flux gap of the Kogut-Susskind theory, which the reference does not quote.
ELECTRIC_GAP = 0.5

#: The settings of the experiment: ``Delta = 57 Hz`` and ``delta/J = 16`` (case 1 of Fig. 2A).
EXPERIMENT_TILT_HZ = 57.0
EXPERIMENT_STAGGERING = 16.0

#: The staggering ``delta/J`` of the three quench cases of Fig. 2A, all on the resonance line.
PAPER_CASES: tuple[tuple[str, float], ...] = (("case 1", 16.0), ("case 2", 6.5), ("case 3", 1.0))

#: The ``U/J`` grid of the Fig. 2D scan; on resonance ``U/J = 2 (delta/J)``.
PAPER_INTERACTION_RATIOS: tuple[float, ...] = (
    1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 16.0, 20.0, 24.0, 28.0, 32.0, 35.0
)  # fmt: skip

#: The oscillation the reference fits at ``delta/J = 16``: ``f_exp = 21 Hz`` (Fig. S3a) and
#: ``1/gamma = 63 +- 9 ms`` (main text).
PAPER_FREQUENCY_HZ = 21.0
PAPER_DAMPING_MS = 63.0
PAPER_DAMPING_UNCERTAINTY_MS = 9.0

#: The chain length at which the gauge theory is realised in its gauge-invariant sector.
GAUGE_THEORY_SITES = 13

#: The admissible ranges of the hardware knobs, in Hz.
DEVICE_TUNNELLING_MAX_HZ = 400.0
DEVICE_INTERACTION_RANGE_HZ = (100.0, 4000.0)
DEVICE_SUPERLATTICE_MAX_HZ = 2000.0
DEVICE_TILT_MAX_HZ = 200.0

#: The experimental time window (Fig. 1C) and the window of the parameter scan (supplement).
WINDOW_MS = 150.0
SCAN_WINDOW_MS = 120.0

#: The number of sample times over a time window.
SAMPLES = 301

#: The number of largest ``U/J`` points over which the tail slope of the scan is fitted.
TAIL_POINTS = 4


def _knobs_at(tunnelling: float, tilt: float) -> dict[str, float]:
    """The knobs of the experiment at a given tunnelling: ``delta = 16 J`` and ``U = 2 delta``."""
    superlattice_depth = EXPERIMENT_STAGGERING * tunnelling
    return {
        P.TUNNELLING: tunnelling,
        P.INTERACTION: 2 * superlattice_depth,
        P.SUPERLATTICE: superlattice_depth,
        P.TILT: tilt,
    }


def experiment_knobs() -> dict[str, float]:
    """The four knobs of the experiment, recovered from the quoted ratios.

    The ratios fix three knobs relative to ``J`` and ``Delta`` is quoted directly; the scale
    follows from inverting [`coupling_map`][qsimod.usecases.schwinger.coupling_map] for
    ``kappa = 14.5 Hz``.

    Returns:
        ``J``, ``U``, ``delta`` and ``Delta``, in rad/ms.

    """
    tilt = from_hertz(EXPERIMENT_TILT_HZ)
    target = from_hertz(TARGET_COUPLING_HZ)
    forward = coupling_map()

    def residual(tunnelling: float) -> float:
        return forward.evaluate_real(_knobs_at(tunnelling, tilt)) - target

    lowest = 2 * tilt / EXPERIMENT_STAGGERING
    tunnelling = float(brentq(residual, lowest, from_hertz(DEVICE_TUNNELLING_MAX_HZ), xtol=1e-15))
    return _knobs_at(tunnelling, tilt)


def case_knobs(experiment: dict[str, float], staggering: float) -> dict[str, float]:
    """One quench case of Fig. 2A: ``U`` and ``Delta`` of the experiment, ``J`` set by the case.

    Args:
        experiment: the four knobs of the experiment, in rad/ms.
        staggering: the ``delta / J`` of the case.

    Returns:
        The four knobs of the case, in rad/ms.

    """
    return {
        P.TUNNELLING: experiment[P.SUPERLATTICE] / staggering,
        P.INTERACTION: experiment[P.INTERACTION],
        P.SUPERLATTICE: experiment[P.SUPERLATTICE],
        P.TILT: experiment[P.TILT],
    }


def device_box() -> AdmissibleSet:
    """The admissible set of the hardware knobs, at the scale of an optical superlattice."""
    low, high = DEVICE_INTERACTION_RANGE_HZ
    return device_limits(
        tunnelling_max=from_hertz(DEVICE_TUNNELLING_MAX_HZ),
        interaction_range=(from_hertz(low), from_hertz(high)),
        superlattice_max=from_hertz(DEVICE_SUPERLATTICE_MAX_HZ),
        tilt_max=from_hertz(DEVICE_TILT_MAX_HZ),
    )


# ---------------------------------------------------------------------------
# What one knob setting is worth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """One knob setting and its measured errors.

    Attributes:
        label: the origin of the setting.
        knobs: the four hardware knobs, in rad/ms.
        mass: the effective mass the setting realises, in rad/ms.
        coupling: the effective coupling the setting realises, in rad/ms.
        margin: the weakest regime margin of the perturbative transformation, in decades.
        deviation: the largest difference between the trajectories of ``<n_matter>`` under the
            hardware model and under the effective theory.
        leakage: the largest weight outside the declared occupation subspace.
        violation: the time-averaged gauge violation ``eta``.
        window_ms: the time window, in ms.

    """

    label: str
    knobs: dict[str, float]
    mass: float
    coupling: float
    margin: float
    deviation: float
    leakage: float
    violation: float
    window_ms: float

    @property
    def staggering(self) -> float:
        """``delta / J``, the ratio by which the experiment reports its cases."""
        return self.knobs[P.SUPERLATTICE] / self.knobs[P.TUNNELLING]

    @property
    def interaction_ratio(self) -> float:
        """``U / J``, the abscissa of Fig. 2D."""
        return self.knobs[P.INTERACTION] / self.knobs[P.TUNNELLING]

    @property
    def dimensionless_time(self) -> float:
        """``t * kappa`` over the time window, the time axis of Fig. 2A."""
        return self.window_ms * 1e-3 * to_hertz(self.coupling)

    @property
    def coupling_per_tunnelling(self) -> float:
        """``kappa / J``, the second-order coupling relative to the tunnelling."""
        return self.coupling / self.knobs[P.TUNNELLING]

    def knob_line(self) -> str:
        """The knobs and the effective parameters they realise, in Hz, as one table row."""
        return (
            f"{to_hertz(self.knobs[P.TUNNELLING]):9.2f}"
            f"{to_hertz(self.knobs[P.INTERACTION]):10.2f}"
            f"{to_hertz(self.knobs[P.SUPERLATTICE]):10.2f}"
            f"{to_hertz(self.knobs[P.TILT]):9.2f}"
            f"{to_hertz(self.mass):9.3f}"
            f"{to_hertz(self.coupling):9.3f}"
        )

    def measurement_line(self) -> str:
        """The declared margin and the three measured errors, as one table row."""
        return (
            f"{self.staggering:9.1f}{self.dimensionless_time:9.2f}{self.margin:+9.2f}"
            f"{self.deviation:11.4f}{self.leakage:9.4f}{self.violation:9.4f}"
        )


@dataclass(frozen=True)
class Bench:
    """The shared numerical realisation in which knob settings are evaluated.

    Attributes:
        matter_sites: the chain length ``N``.
        space: the Hilbert space in which both models are realised.
        projector: the projector onto the declared local occupation subspace.
        initial: the ``|1 0 1 0 1 ...>`` initial state.
        occupation: the realised ``<n_matter>`` observable.
        violation: the realised gauge-violation observable ``eta``.
        device: the Hamiltonian of ``bose_hubbard``, with unbound parameters.
        theory: the Hamiltonian of ``effective_bosonic``, with unbound parameters.

    """

    matter_sites: int
    space: HilbertSpace
    projector: Array
    initial: Array
    occupation: Array
    violation: Array
    device: OperatorSum
    theory: OperatorSum

    @classmethod
    def of(cls, matter_sites: int) -> Bench:
        """Realise both models of the analogue branch at a given chain length."""
        device_model = superlattice(matter_sites)
        theory_model = effective_bosonic(matter_sites)
        subspace = local_occupation_subspace(matter_sites)
        space = HilbertSpace.of(
            device_model.structure, RealisationRequest(boson_cutoff=subspace.required_cutoff())
        )
        return cls(
            matter_sites=matter_sites,
            space=space,
            projector=local_subspace_projector(subspace, space),
            initial=space.basis_state(canonical_state_configuration(matter_sites)),
            occupation=build_operator(matter_occupation_observable(matter_sites), space, {}),
            violation=build_operator(gauge_violation_observable(matter_sites), space, {}),
            device=device_model.hamiltonian,
            theory=theory_model.hamiltonian,
        )

    def times(self, window_ms: float = WINDOW_MS) -> NDArray[np.float64]:
        """The sample times of a time window, in ms."""
        return np.linspace(0.0, window_ms, SAMPLES)

    def assess(
        self, label: str, knobs: dict[str, float], window_ms: float = WINDOW_MS
    ) -> tuple[Outcome, NDArray[np.float64]]:
        """Evaluate one knob setting: its regime margin and its measured dynamical errors.

        Args:
            label: the origin of the setting.
            knobs: the four hardware knobs, in rad/ms.
            window_ms: the time window, in ms.

        Returns:
            The outcome and the trajectory of ``<n_matter>`` under the hardware model.

        """
        mass = mass_map().evaluate_real(knobs)
        coupling = coupling_map().evaluate_real(knobs)
        effective = {P.MASS_EFFECTIVE_BOSONIC: mass, P.COUPLING_EFFECTIVE_BOSONIC: coupling}

        device = build_operator(self.device, self.space, knobs)
        theory = sandwich(build_operator(self.theory, self.space, effective), self.projector)

        times = self.times(window_ms)
        device_states = evolve_state(device, self.initial, times)
        theory_states = evolve_state(theory, self.initial, times)
        device_occupation = expectation(self.occupation, device_states)
        theory_occupation = expectation(self.occupation, theory_states)

        outcome = Outcome(
            label=label,
            knobs=dict(knobs),
            mass=mass,
            coupling=coupling,
            margin=validity().report({**knobs, **effective}).weakest_margin,
            deviation=float(jnp.max(jnp.abs(device_occupation - theory_occupation))),
            leakage=float(jnp.max(1.0 - expectation(self.projector, device_states))),
            violation=float(jnp.mean(expectation(self.violation, device_states))),
            window_ms=window_ms,
        )
        return outcome, np.asarray(device_occupation)


# ---------------------------------------------------------------------------
# Two figures of the reference
# ---------------------------------------------------------------------------


def damped_sine(
    times_ms: NDArray[np.float64],
    offset: float,
    amplitude: float,
    rate: float,
    frequency_hz: float,
    phase: float,
) -> NDArray[np.float64]:
    """``n0 + A exp(-gamma t) sin(2 pi f t + b)``, Eq. (S10) of the reference with one frequency."""
    decay: NDArray[np.float64] = np.exp(-rate * times_ms)
    return offset + amplitude * decay * np.sin(2 * np.pi * frequency_hz * times_ms * 1e-3 + phase)


@dataclass(frozen=True)
class Oscillation:
    """A many-body oscillation, characterised by the fit of Eq. (S10).

    Attributes:
        frequency_hz: the fitted frequency, in Hz.
        damping_ms: the fitted ``1/gamma``, in ms; infinite if the fit finds no decay.

    """

    frequency_hz: float
    damping_ms: float

    @classmethod
    def fitted(cls, times_ms: NDArray[np.float64], signal: NDArray[np.float64]) -> Oscillation:
        """Fit Eq. (S10) to one trajectory."""
        guess = (float(signal.mean()), 0.5, 1e-2, PAPER_FREQUENCY_HZ, math.pi / 2)
        parameters, _ = curve_fit(damped_sine, times_ms, signal, p0=guess, maxfev=200_000)
        _, _, rate, frequency_hz, _ = (float(value) for value in parameters)
        return cls(
            frequency_hz=abs(frequency_hz),
            damping_ms=math.inf if abs(rate) < 1e-9 else 1.0 / abs(rate),
        )


def gauge_theory_oscillation(
    matter_sites: int,
    coupling: float,
    window_ms: float = WINDOW_MS,
) -> tuple[Oscillation, int]:
    """Fit Eq. (S10) to the effective theory realised in its gauge-invariant sector.

    Args:
        matter_sites: the chain length ``N``.
        coupling: ``kappa``, in rad/ms; the mass is zero.
        window_ms: the time window, in ms.

    Returns:
        The fitted oscillation and the dimension of the sector.

    """
    model = effective_bosonic(matter_sites)
    subspace = local_occupation_subspace(matter_sites)
    basis = SectorBasis.of(
        model.structure,
        subspace,
        gauss_operators(matter_sites, field=ElectricField.BOSON),
        RealisationRequest(boson_cutoff=subspace.required_cutoff()),
    )
    generator = build_operator_in(
        model.hamiltonian,
        basis,
        {P.MASS_EFFECTIVE_BOSONIC: 0.0, P.COUPLING_EFFECTIVE_BOSONIC: coupling},
    )
    occupation = build_operator_in(matter_occupation_observable(matter_sites), basis, {})
    initial = basis.state(canonical_state_configuration(matter_sites))
    times = np.linspace(0.0, window_ms, SAMPLES)
    states = evolve_state(generator, initial, times)
    trajectory = np.asarray(expectation(occupation, states))
    return Oscillation.fitted(times, trajectory), basis.dimension


def scan_the_resonance_line(
    bench: Bench,
    experiment: dict[str, float],
    interaction_ratios: tuple[float, ...],
) -> tuple[Outcome, ...]:
    """The curve of Fig. 2D: the steady-state gauge violation along the resonance line.

    Args:
        bench: the shared realisation.
        experiment: the four knobs of the experiment, in rad/ms.
        interaction_ratios: the ``U/J`` grid.

    Returns:
        One outcome per grid point, in the order given; points that coincide with a case of
        Fig. 2A carry its name.

    """
    named = {2 * staggering: name for name, staggering in PAPER_CASES}
    return tuple(
        bench.assess(named.get(ratio, ""), case_knobs(experiment, ratio / 2), SCAN_WINDOW_MS)[0]
        for ratio in interaction_ratios
    )


def tail_slope(scan: tuple[Outcome, ...], points: int = TAIL_POINTS) -> float:
    """``d ln(eta) / d ln(U/J)`` over the largest ``U/J`` points; second order predicts ``-2``."""
    tail = scan[-points:]
    if len(tail) < 2 or any(point.violation <= 0.0 for point in tail):
        return math.nan
    ratios = np.log([point.interaction_ratio for point in tail])
    violations = np.log([point.violation for point in tail])
    return float(np.polyfit(ratios, violations, 1)[0])


@dataclass(frozen=True)
class Figures:
    """The reproductions of Fig. S3a and Fig. 2D.

    Attributes:
        oscillation: Eq. (S10) fitted to the hardware model at the prescribed knobs.
        converged: the same fit on the effective theory in its gauge-invariant sector at
            [`GAUGE_THEORY_SITES`][examples.zhou2022.GAUGE_THEORY_SITES].
        sector: the dimension of that sector.
        scan: the curve of Fig. 2D, one point per ``U/J``, ascending.

    """

    oscillation: Oscillation
    converged: Oscillation
    sector: int
    scan: tuple[Outcome, ...]

    @property
    def labelled_cases(self) -> tuple[Outcome, ...]:
        """The scan points that are named cases of Fig. 2A, in the order of the reference."""
        names = [name for name, _ in PAPER_CASES]
        found = [point for point in self.scan if point.label in names]
        return tuple(sorted(found, key=lambda point: names.index(point.label)))


def reproduce_figures(
    bench: Bench,
    experiment: Outcome,
    trajectory: NDArray[np.float64],
    interaction_ratios: tuple[float, ...] = PAPER_INTERACTION_RATIOS,
) -> Figures:
    """Reproduce Fig. S3a and Fig. 2D from the prescribed knobs.

    Args:
        bench: the shared realisation.
        experiment: the assessed prescribed knobs.
        trajectory: ``<n_matter>(t)`` under the hardware model at those knobs.
        interaction_ratios: the ``U/J`` grid of the scan.

    Returns:
        The two fits and the scan.

    """
    converged, sector = gauge_theory_oscillation(GAUGE_THEORY_SITES, experiment.coupling)
    return Figures(
        oscillation=Oscillation.fitted(bench.times(), trajectory),
        converged=converged,
        sector=sector,
        scan=scan_the_resonance_line(bench, experiment.knobs, interaction_ratios),
    )
