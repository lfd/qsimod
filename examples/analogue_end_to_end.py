"""The analogue pipeline of the Schwinger case study, end to end, against a reference experiment.

The parameters of the application model are entered, the pipeline is type-checked, the
hardware knobs are solved for, and both layers are realised numerically and compared.  The
request is that of Zhou et al., *Thermalization dynamics of a gauge theory on a quantum
simulator*, Science **377**, 311-314 (2022), whose Eqs. (S5) and (S6) are the artifact
``effective_bosonic`` (the boson encoding ``H_IR3`` of the article) and the perturbative
transformation of the framework.  The knobs of the experiment itself are evaluated as a second
setting, the prescribed knobs.  Section 6a fits Eq. (S10) to the oscillation of Fig. S3a; section
6b scans the steady-state gauge violation along the resonance line as in Fig. 2D.

Run it with::

    uv run python examples/analogue_end_to_end.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.optimize import brentq, curve_fit

from examples import Report
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
from qsimod.solving import realise_parameters
from qsimod.symbolic import OperatorSum
from qsimod.units import from_hertz, to_hertz
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import (
    build_graph,
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

#: The request at the application layer, in the units of the reference: a massless fermion at
#: the resonance ``U = 2 delta`` and the coupling of its quench data ("kappa = 14.5 Hz and m = 0",
#: main text).
TARGET_MASS_HZ = 0.0
TARGET_COUPLING_HZ = 14.5

#: The electric-flux gap of the Kogut-Susskind theory, against which the validity condition of
#: the quantum-link truncation is measured (see `qsimod.transformations.truncations`).  The
#: reference does not quote this number.
ELECTRIC_GAP = 0.5

#: The settings of the experiment: ``Delta = 57 Hz`` (the gravitational gradient) and
#: ``delta/J = 16`` at case 1 of Fig. 2A, on the resonance line.
EXPERIMENT_TILT_HZ = 57.0
EXPERIMENT_STAGGERING = 16.0

#: The staggering of the three quench cases of Fig. 2A: "tuning the Bose-Hubbard parameters
#: from delta/J = 1 (case 3) to delta/J = 16 (case 1)" (main text), with case 2 at
#: ``delta/J = 6.5`` from Fig. S3b.  All three lie on the resonance line (Fig. 2 caption).  The
#: reference quotes the cases by ratios only; ``U`` and ``Delta`` are held fixed and ``J`` varies.
PAPER_CASES: tuple[tuple[str, float], ...] = (
    ("case 1", 16.0),
    ("case 2", 6.5),
    ("case 3", 1.0),
)

#: The ``U/J`` grid of the Fig. 2D scan.  On the resonance line ``U/J = 2 (delta/J)``, so the
#: three cases above are the points ``32, 13, 2``.
PAPER_INTERACTION_RATIOS: tuple[float, ...] = (
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.5,
    8.0,
    10.0,
    13.0,
    16.0,
    20.0,
    24.0,
    28.0,
    32.0,
    35.0,
)

#: The oscillation frequency the reference fits at ``delta/J = 16``: "we extract the
#: oscillation frequency (f_exp = 21 Hz) at large staggering delta/J = 16" (Fig. S3a).
PAPER_FREQUENCY_HZ = 21.0

#: The damping the reference fits at the same point: "``1/gamma = 63 +- 9 ms`` (experiment) and
#: ``64.4 +- 0.4 ms`` (t-DMRG)" (main text).
PAPER_DAMPING_MS = 63.0
PAPER_DAMPING_UNCERTAINTY_MS = 9.0

#: The chain length at which the gauge theory is realised for section 6a, in its
#: gauge-invariant sector (see [`qsimod.realise.sector`][qsimod.realise.sector]).
GAUGE_THEORY_SITES = 13

#: The admissible ranges of the hardware knobs, in Hz.
DEVICE_TUNNELLING_MAX_HZ = 400.0
DEVICE_INTERACTION_RANGE_HZ = (100.0, 4000.0)
DEVICE_SUPERLATTICE_MAX_HZ = 2000.0
DEVICE_TILT_MAX_HZ = 200.0

#: The experimental time window, in ms: "evolution times t <= 150 ms" (Fig. 1C).
WINDOW_MS = 150.0

#: The time window of the scan, in ms: "each data point in the parameter scan is obtained with
#: 120 ms of quench evolution during which the system roughly relaxes to a steady state"
#: (supp.).
SCAN_WINDOW_MS = 120.0

#: The number of sample times over a time window.
SAMPLES = 301


def _knobs_at(tunnelling: float, tilt: float) -> dict[str, float]:
    """The knobs of the experiment at a given tunnelling: ``delta = 16 J`` and ``U = 2 delta``."""
    superlattice = EXPERIMENT_STAGGERING * tunnelling
    return {
        P.TUNNELLING: tunnelling,
        P.INTERACTION: 2 * superlattice,
        P.SUPERLATTICE: superlattice,
        P.TILT: tilt,
    }


def experiment_knobs() -> dict[str, float]:
    """The four knobs of the experiment, the prescribed knobs, recovered from the quoted values.

    The quoted ratios fix three knobs relative to ``J``, and ``Delta`` is quoted directly; the
    remaining scale is found by inverting [`coupling_map`][qsimod.usecases.schwinger.coupling_map]
    (Eq. (S6)) for ``kappa = 14.5 Hz`` with a root-find bracketed by the admissible set of the
    hardware model.

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
    """One of the quench cases of Fig. 2A, at the scale of the experiment.

    ``U`` and ``Delta`` are held fixed, ``delta = U/2`` keeps the resonance, and the staggering
    of the case is set by ``J`` alone.

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
        margin: the weakest regime margin of the perturbative transformation, in decades (positive
            inside the declared regime).
        deviation: the largest absolute difference between the trajectories of ``<n_matter>``
            under ``bose_hubbard`` and under ``effective_bosonic``.
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
        """``kappa / J``, the second-order coupling relative to the tunnelling it derives from."""
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


def damped_sine(
    times_ms: NDArray[np.float64],
    offset: float,
    amplitude: float,
    rate: float,
    frequency_hz: float,
    phase: float,
) -> NDArray[np.float64]:
    """``n0 + A exp(-gamma t) sin(2 pi f t + b)``, Eq. (S10) of the reference with one frequency.

    Eq. (S10) has two damped sines; at ``delta/J = 16`` the reference finds "identical f1 and
    f2, whence we take the average".
    """
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
        """Fit Eq. (S10) to one trajectory.

        Args:
            times_ms: the sample times, in ms.
            signal: the observable at those times.

        Returns:
            The fitted oscillation.

        """
        guess = (float(signal.mean()), 0.5, 1e-2, PAPER_FREQUENCY_HZ, math.pi / 2)
        parameters, _ = curve_fit(damped_sine, times_ms, signal, p0=guess, maxfev=200_000)
        _, _, rate, frequency_hz, _ = (float(value) for value in parameters)
        return cls(
            frequency_hz=abs(frequency_hz),
            damping_ms=math.inf if abs(rate) < 1e-9 else 1.0 / abs(rate),
        )


@dataclass(frozen=True)
class Comparison:
    """The results of one run.

    Attributes:
        solved: the free-knob setting, found by the solve of the framework.
        experiment: the prescribed-knob setting of the experiment.
        oscillation: Eq. (S10) fitted to ``<n_matter>(t)`` under ``bose_hubbard`` at the prescribed
            knobs, in the full space.
        converged: the same fit on ``effective_bosonic``, realised in its gauge-invariant sector at
            [`GAUGE_THEORY_SITES`][examples.analogue_end_to_end.GAUGE_THEORY_SITES].
        scan: the curve of Fig. 2D, one point per ``U/J``, ascending.

    """

    solved: Outcome
    experiment: Outcome
    oscillation: Oscillation
    converged: Oscillation
    scan: tuple[Outcome, ...]

    @property
    def labelled_cases(self) -> tuple[Outcome, ...]:
        """The scan points that are named cases of Fig. 2A, in the order of the reference."""
        names = [name for name, _ in PAPER_CASES]
        found = [point for point in self.scan if point.label in names]
        return tuple(sorted(found, key=lambda point: names.index(point.label)))


# ---------------------------------------------------------------------------
# The numerical check, run identically on either setting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bench:
    """The shared numerical realisation in which both knob settings are evaluated.

    Attributes:
        matter_sites: the chain length ``N``.
        space: the Hilbert space in which both models are realised.
        projector: the projector onto the declared local occupation subspace.
        initial: the ``|1 0 1 0 1 ...>`` initial state.
        occupation: the realised ``<n_matter>`` observable.
        violation: the realised gauge-violation observable ``eta``.
        device: the Hamiltonian of the analogue simulator model ``bose_hubbard``, with unbound
            parameters.
        theory: the Hamiltonian of the effective theory ``effective_bosonic``, with unbound
            parameters.

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

        The effective parameters follow from the knobs through the forward maps of
        the perturbative transformation.

        Args:
            label: the origin of the setting.
            knobs: the four hardware knobs, in rad/ms.
            window_ms: the time window, in ms.

        Returns:
            The outcome and the trajectory of ``<n_matter>`` under ``bose_hubbard``.

        """
        mass = mass_map().evaluate_real(knobs)
        coupling = coupling_map().evaluate_real(knobs)
        effective = {P.MASS_EFFECTIVE_BOSONIC: mass, P.COUPLING_EFFECTIVE_BOSONIC: coupling}

        device = build_operator(self.device, self.space, knobs)
        # The effective generator on the physical subspace is the projected operator P B P.
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


def gauge_theory_oscillation(
    matter_sites: int,
    coupling: float,
    window_ms: float = WINDOW_MS,
) -> tuple[Oscillation, int]:
    """Fit Eq. (S10) to the effective theory ``effective_bosonic`` in its gauge-invariant sector.

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


#: The number of largest ``U/J`` points of the scan over which the tail slope is fitted.
TAIL_POINTS = 4


def scan_the_resonance_line(
    bench: Bench,
    experiment: dict[str, float],
    interaction_ratios: tuple[float, ...],
) -> tuple[Outcome, ...]:
    """The curve of Fig. 2D: the steady-state gauge violation along the resonance line.

    ``U`` and ``Delta`` stay at the values of the experiment and ``J`` alone varies (see
    [`case_knobs`][examples.analogue_end_to_end.case_knobs]); on resonance
    ``delta/J = (U/J) / 2``.  Points coinciding with a case of Fig. 2A carry its name.

    Args:
        bench: the shared realisation.
        experiment: the four knobs of the experiment, in rad/ms.
        interaction_ratios: the ``U/J`` grid.

    Returns:
        One outcome per grid point, in the order given.

    """
    named = {2 * staggering: name for name, staggering in PAPER_CASES}
    return tuple(
        bench.assess(named.get(ratio, ""), case_knobs(experiment, ratio / 2), SCAN_WINDOW_MS)[0]
        for ratio in interaction_ratios
    )


def tail_slope(scan: tuple[Outcome, ...], points: int = TAIL_POINTS) -> float:
    """``d ln(eta) / d ln(U/J)`` over the scan's largest ``U/J`` points.

    Second-order leakage ``eta ~ (J/delta)**2`` predicts a slope of ``-2``.

    Args:
        scan: the scan, ascending in ``U/J``.
        points: the number of largest ``U/J`` points fitted over.

    Returns:
        The fitted slope, or ``nan`` if there are too few points or any violation vanishes.

    """
    tail = scan[-points:]
    if len(tail) < 2 or any(point.violation <= 0.0 for point in tail):
        return math.nan
    ratios = np.log([point.interaction_ratio for point in tail])
    violations = np.log([point.violation for point in tail])
    return float(np.polyfit(ratios, violations, 1)[0])


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def main(
    matter_sites: int = 3,
    *,
    interaction_ratios: tuple[float, ...] = PAPER_INTERACTION_RATIOS,
    verbose: bool = True,
) -> Comparison:
    """Run the analogue pipeline and compare its knob setting with that of the experiment.

    Args:
        matter_sites: the chain length ``N``.
        interaction_ratios: the ``U/J`` grid of the scan of Fig. 2D.
        verbose: whether the report is printed.

    Returns:
        The two knob settings, the fitted oscillations and the scan of Fig. 2D.

    Raises:
        RuntimeError: if the solve does not return a usable point.

    """
    graph = build_graph(matter_sites)
    pipeline = graph.analogue
    report = Report(verbose=verbose)
    say = report.say

    report.section("1. pipeline")
    say(pipeline)

    targets = {
        P.MASS_LATTICE_QED: from_hertz(TARGET_MASS_HZ),
        P.COUPLING_QUANTUM_LINK_STAGGERED: from_hertz(TARGET_COUPLING_HZ),
        P.ELECTRIC_GAP: ELECTRIC_GAP,
    }
    say()
    report.section("2. composite relation")
    say(pipeline.classify_relation(frozenset(targets)))

    knobs = [P.TUNNELLING, P.INTERACTION, P.SUPERLATTICE, P.TILT]
    result = realise_parameters(
        pipeline, targets=targets, unknowns=knobs, admissible_set=device_box()
    )
    say()
    report.section(f"3. solve for m = {TARGET_MASS_HZ:g} Hz, kappa = {TARGET_COUPLING_HZ:g} Hz")
    say(result)
    if not result.status.is_success:
        msg = f"the solve did not produce a usable point: {result.status}"
        raise RuntimeError(msg)

    say()
    report.section("4. error axes")
    say(pipeline.error_report(result.point, solver_residuals=result.residuals))

    say()
    report.section(f"5. knobs and errors, N = {matter_sites}, window {WINDOW_MS:g} ms")
    bench = Bench.of(matter_sites)
    solved, _ = bench.assess("framework", result.subset(knobs))
    experiment, trajectory = bench.assess("experiment", experiment_knobs())
    say(f"   {bench.space}")
    say()
    say(f"   {'setting':<11}{'J':>9}{'U':>10}{'delta':>10}{'Delta':>9}{'m':>9}{'kappa':>9}   (Hz)")
    for outcome in (solved, experiment):
        say(f"   {outcome.label:<11}{outcome.knob_line()}")
    say()
    say(
        f"   {'setting':<11}{'delta/J':>9}{'t*kappa':>9}{'margin':>9}"
        f"{'deviation':>11}{'leakage':>9}{'eta':>9}"
    )
    for outcome in (solved, experiment):
        say(f"   {outcome.label:<11}{outcome.measurement_line()}")

    say()
    report.section("6a. oscillation frequency and damping, against Fig. S3a and Fig. 3")
    oscillation = Oscillation.fitted(bench.times(), trajectory)
    converged, sector = gauge_theory_oscillation(GAUGE_THEORY_SITES, experiment.coupling)
    gap = 100 * (converged.frequency_hz - PAPER_FREQUENCY_HZ) / PAPER_FREQUENCY_HZ
    say(f"   {'':<34}{'f / Hz':>9}{'1/gamma / ms':>14}")
    say(
        f"   {f'bose_hubbard, full space, N = {matter_sites}':<34}"
        f"{oscillation.frequency_hz:9.2f}{oscillation.damping_ms:14.1f}"
    )
    say(
        f"   {f'effective_bosonic, sector, N = {GAUGE_THEORY_SITES} (dim {sector})':<34}"
        f"{converged.frequency_hz:9.2f}{converged.damping_ms:14.1f}"
    )
    say(
        f"   {'Zhou et al. (2022) measured':<34}{PAPER_FREQUENCY_HZ:9.2f}"
        f"{PAPER_DAMPING_MS:9.1f} +-{PAPER_DAMPING_UNCERTAINTY_MS:2.0f}"
    )
    say(f"   {'relative gap, effective_bosonic on the frequency':<34}{gap:+9.2f} %")

    say()
    report.section(
        f"6b. steady-state violation vs U/J, against Fig. 2D, window {SCAN_WINDOW_MS:g} ms"
    )
    scan = scan_the_resonance_line(bench, experiment.knobs, interaction_ratios)
    say(
        f"   {'U/J':>7}{'delta/J':>9}{'J/Hz':>9}{'kappa/J':>9}"
        f"{'t*kappa':>10}{'margin':>9}{'eta':>9}   case"
    )
    for outcome in scan:
        say(
            f"   {outcome.interaction_ratio:7.1f}{outcome.staggering:9.1f}"
            f"{to_hertz(outcome.knobs[P.TUNNELLING]):9.1f}"
            f"{outcome.coupling_per_tunnelling:9.2f}{outcome.dimensionless_time:10.2f}"
            f"{outcome.margin:+9.2f}{outcome.violation:9.4f}   {outcome.label}"
        )
    peak = max(scan, key=lambda point: point.violation)
    say()
    say(f"   {'peak at U/J':<30} {peak.interaction_ratio:8.1f}")
    say(f"   {'tail slope, dln(eta)/dln(U/J)':<30} {tail_slope(scan):+8.2f}")
    say(f"   {'expected from (J/delta)**2':<30} {-2.0:+8.2f}")
    say()
    say(
        "   not reproduced: the thermalisation of Figs. 3 and 4, which needs a thermal\n"
        "   ensemble over the whole spectrum rather than one quench trajectory."
    )
    return Comparison(
        solved=solved,
        experiment=experiment,
        oscillation=oscillation,
        converged=converged,
        scan=scan,
    )


if __name__ == "__main__":
    main()
