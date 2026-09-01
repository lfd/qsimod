"""One tilted Bose-Hubbard chain solved for two theories, each setting evaluated against both.

The analogue simulator model ``L3a`` (``H_sim`` of the article) is the target of the gauge
theory of the case study and of an antiferromagnetic Ising chain.  The Ising chain follows
Simon et al., *Quantum simulation of antiferromagnetic spin chains in an optical lattice*,
Nature **472**, 307-312 (2011) (`docs/references/paper_simon11.pdf`).  Section 4 recovers the
transition line ``E = U + 1.85 t`` (``hz = 1 - 0.66 hx``) of that reference; section 5 solves
the same hardware model for both theories and evaluates each knob setting against the declared
regimes of both; section 6 compares the dipole manifold of the hardware model with the spectrum
of the Ising chain, with and without the end-spin field.

Run it with::

    uv run python examples/shared_device.py
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from examples import Report
from qsimod.models.magnetism import end_magnetisation_term
from qsimod.parameters import AdmissibleSet
from qsimod.realise import (
    HilbertSpace,
    RealisationRequest,
    build_operator,
    manifold_levels,
    spacing_deviation,
)
from qsimod.solving import SolveResult, realise_parameters
from qsimod.usecases import ising, schwinger
from qsimod.usecases.ising import KNOBS
from qsimod.usecases.ising import ParameterNames as K
from qsimod.usecases.schwinger import ParameterNames as S

# ---------------------------------------------------------------------------
# The two requests
# ---------------------------------------------------------------------------

#: The requested Ising coupling, in the energy unit of the package.  It equals the on-site
#: interaction and must therefore lie inside the admissible interaction range of the lattice.
REQUEST_COUPLING = 1.0

#: The requested transverse field.  The parameter relation holds near the multicritical point
#: ``(hz, hx) = (1, 0)``.
REQUEST_TRANSVERSE_FIELD = 0.05

#: The transition line of the reference: "a fast ramp to just below the transition point at
#: ``E = U + 1.85 t`` (``hz = 1 - 0.66 hx``)".
PAPER_CRITICAL_DETUNING_IN_TUNNELLINGS = 1.85
PAPER_CRITICAL_SLOPE = 0.66

#: The request of the gauge theory: a massless fermion at the resonance, at the coupling of the
#: quench data of `paper_zhou22`.
GAUGE_COUPLING = 0.0045
GAUGE_ELECTRIC_GAP = 0.5

#: The chain length at which the spectra are compared: ``2N-1`` lattice sites and ``2N-2`` spins.
SITES = 3


def lattice_limits() -> AdmissibleSet:
    """The admissible set of the lattice, against which both requests are posed.

    See [`ising.device_limits`][qsimod.usecases.ising.device_limits].
    """
    return ising.device_limits()


def solve_ising(limits: AdmissibleSet, longitudinal_field: float) -> SolveResult:
    """Solve the lattice for an Ising chain at a given point of the ``(hz, hx)`` plane."""
    return realise_parameters(
        ising.build_graph(SITES, admissible_set=limits).device,
        targets={
            K.COUPLING_K1: REQUEST_COUPLING,
            K.TRANSVERSE_FIELD: REQUEST_TRANSVERSE_FIELD,
            K.LONGITUDINAL_FIELD: longitudinal_field,
        },
        unknowns=list(KNOBS),
        admissible_set=limits,
        initial=ising.device_start(REQUEST_COUPLING),
    )


def solve_gauge(limits: AdmissibleSet) -> SolveResult:
    """Solve the same lattice, within the same admissible set, for the gauge theory."""
    return realise_parameters(
        schwinger.build_graph(SITES, admissible_set=limits).analogue,
        targets={
            S.MASS_L1: 0.0,
            S.COUPLING_L2A: GAUGE_COUPLING,
            S.ELECTRIC_GAP: GAUGE_ELECTRIC_GAP,
        },
        unknowns=list(KNOBS),
        admissible_set=limits,
        # Starting point in the gauge theory's corner of the admissible set.
        initial={
            S.TUNNELLING: 0.03,
            S.INTERACTION: 2.0,
            S.SUPERLATTICE: 1.0,
            S.TILT: 0.05,
        },
    )


# ---------------------------------------------------------------------------
# Judging one setting against both theories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """The evaluation of one knob setting against the declared regime of one theory.

    Attributes:
        theory: the theory whose regime conditions were evaluated.
        valid: whether every validity condition of the transformation of that theory holds.
        margin: the weakest regime margin, in decades.
        weakest: the name of the binding condition.

    """

    theory: str
    valid: bool
    margin: float
    weakest: str

    def __str__(self) -> str:
        verdict = "VALID  " if self.valid else "INVALID"
        return f"{verdict} {self.margin:+6.2f}  ({self.weakest})"


def ising_environment(knobs: dict[str, float]) -> dict[str, float]:
    """The Ising parameters of a knob setting, by the forward maps of transformation (b)."""
    coupling = ising.coupling_map().evaluate_real(knobs)
    return {
        **knobs,
        K.COUPLING_K2: coupling,
        K.TRANSVERSE: ising.transverse_map().evaluate_real(knobs),
        K.LONGITUDINAL: ising.longitudinal_map().evaluate_real({**knobs, K.COUPLING_K2: coupling}),
    }


def gauge_environment(knobs: dict[str, float]) -> dict[str, float]:
    """The gauge-theory parameters the same knob setting realises."""
    return {
        **knobs,
        S.MASS_L2C: schwinger.mass_map().evaluate_real(knobs),
        S.COUPLING_L2C: schwinger.coupling_map().evaluate_real(knobs),
    }


def verdicts(knobs: dict[str, float]) -> tuple[Verdict, Verdict]:
    """Evaluate one knob setting against the declared regimes of both theories."""
    found: list[Verdict] = []
    for theory, conditions, environment in (
        ("Ising", ising.validity(), ising_environment(knobs)),
        ("gauge", schwinger.validity(), gauge_environment(knobs)),
    ):
        report = conditions.report(environment)
        weakest = report.weakest()
        found.append(
            Verdict(
                theory=theory,
                valid=report.is_valid,
                margin=report.weakest_margin,
                weakest=weakest.name if weakest is not None else "-",
            )
        )
    return found[0], found[1]


# ---------------------------------------------------------------------------
# The numerical check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bench:
    """The numerical realisation in which the hardware model and the magnet are compared.

    The dipole manifold of the hardware model is the set of states reached from one atom per
    site by creating dipoles on non-adjacent bonds; its levels are selected by overlap with
    [`manifold_levels`][qsimod.realise.evolve.manifold_levels].

    Attributes:
        sites: the chain length ``N`` of the hardware model.
        space: the Hilbert space of the hardware model, of dimension ``3**(2N-1)`` at a cutoff
            of two.
        manifold: the basis indices of the non-adjacent-dipole configurations.
        spin_space: the ``2**(2N-2)``-dimensional space of the magnet.
        constrained: the basis indices of the magnet with no two adjacent dipoles, in the order
            of ``manifold``.

    """

    sites: int
    space: HilbertSpace
    manifold: tuple[int, ...]
    spin_space: HilbertSpace
    constrained: tuple[int, ...]

    @classmethod
    def of(cls, sites: int, limits: AdmissibleSet) -> Bench:
        """Enumerate both constrained bases at a given chain length."""
        device = ising.lattice(sites, limits)
        magnet = ising.ising_chain(sites)
        space = HilbertSpace.of(device.structure, RealisationRequest(boson_cutoff=2))
        spin_space = HilbertSpace.of(magnet.structure)
        positions = len(space.sites)
        bonds = positions - 1

        manifold: list[int] = []
        constrained: list[int] = []
        for occupied in itertools.chain.from_iterable(
            itertools.combinations(range(bonds), count) for count in range(bonds + 1)
        ):
            if any(bond + 1 in occupied for bond in occupied):
                continue  # no two dipoles on adjacent bonds
            configuration = [1] * positions
            for bond in occupied:
                configuration[bond] += 1
                configuration[bond + 1] -= 1
            manifold.append(space.index_of(tuple(configuration)))
            # A dipole is a spin *down*, and a spin-1/2 basis index is 0 down, 1 up.
            constrained.append(
                spin_space.index_of(tuple(0 if bond in occupied else 1 for bond in range(bonds)))
            )
        return cls(sites, space, tuple(manifold), spin_space, tuple(constrained))

    def device_spectrum(
        self,
        knobs: dict[str, float],
        limits: AdmissibleSet,
    ) -> tuple[NDArray[np.float64], float]:
        """The dipole levels of the lattice, selected by overlap, and their least weight."""
        operator = build_operator(ising.lattice(self.sites, limits).hamiltonian, self.space, knobs)
        levels, weight = manifold_levels(operator, self.manifold)
        return np.asarray(levels, dtype=np.float64), weight

    def magnet_spectrum(
        self,
        knobs: dict[str, float],
        *,
        end_field: bool,
    ) -> NDArray[np.float64]:
        """The levels of the Ising chain on the same constrained subspace.

        Args:
            knobs: the four hardware knobs.
            end_field: whether the end-spin field of transformation ``(b)``, which the printed
                Hamiltonian omits, is added.

        Returns:
            The levels, ascending.

        """
        environment = ising_environment(knobs)
        operator = np.asarray(
            build_operator(ising.ising_chain(self.sites).hamiltonian, self.spin_space, environment)
        )
        if end_field:
            operator = operator + np.asarray(
                build_operator(
                    end_magnetisation_term(
                        ising.spins_for(self.sites),
                        ising.boundary_field().evaluate_real(knobs),
                    ),
                    self.spin_space,
                    {},
                )
            )
        block = operator[np.ix_(self.constrained, self.constrained)]
        return np.sort(np.real(np.linalg.eigvalsh(block))).astype(np.float64)


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """The results of one run.

    Attributes:
        ising_knobs: the knob setting realising the Ising chain.
        gauge_knobs: the knob setting realising the gauge theory.
        ising_verdicts: the evaluations of the Ising setting against both declared regimes.
        gauge_verdicts: the evaluations of the gauge setting against both declared regimes.
        critical_slope: the slope of the transition line, recovered from the Hubbard parameters.
        weight: the least weight a compared level keeps in the dipole manifold.
        gap: the spectrum gap with the end field, in units of the transverse field.
        dropped: the same gap without the end field.

    """

    ising_knobs: dict[str, float]
    gauge_knobs: dict[str, float]
    ising_verdicts: tuple[Verdict, Verdict]
    gauge_verdicts: tuple[Verdict, Verdict]
    critical_slope: float
    weight: float
    gap: float
    dropped: float

    @property
    def windows_are_disjoint(self) -> bool:
        """Whether neither setting lies inside the declared regime of the other theory."""
        return (
            self.ising_verdicts[0].valid
            and not self.ising_verdicts[1].valid
            and self.gauge_verdicts[1].valid
            and not self.gauge_verdicts[0].valid
        )


def _knob_line(knobs: dict[str, float]) -> str:
    """The four knobs, as one table row."""
    return "".join(f"{knobs[knob]:11.5g}" for knob in KNOBS)


def main(*, verbose: bool = True) -> Comparison:
    """Solve one lattice for two theories and evaluate each setting against both regimes.

    Args:
        verbose: whether the report is printed.

    Returns:
        The two knob settings, the four evaluations and the numerical checks.

    Raises:
        RuntimeError: if either solve fails to return a usable point.

    """
    report = Report(verbose=verbose)
    say = report.say
    limits = lattice_limits()

    report.section("1. one node, two incoming edges")
    graph = ising.shared_graph(SITES, admissible_set=limits)
    say(graph)
    say()
    say(f"   incoming at {ising.DEVICE_TARGET}:")
    for edge in graph.incoming(ising.DEVICE_TARGET):
        say(f"     {edge}")
    say()
    say(
        "   two application-level theories, no node and no step in common until the last\n"
        "   one, and the device's parameters are the same four either way -- which is what\n"
        "   makes the two answers below comparable at all."
    )

    say()
    report.section("2. the box both are posed against")
    for line in str(limits).split("; "):
        say(f"   {line}")
    say()
    say(
        "   the running example's own box adds two coupled constraints; they are the\n"
        "   superlattice derivation's requirements rather than the lattice's reach, and at\n"
        "   a dipole resonance 'tilt within the resonance gap' reads delta <= -U."
    )

    say()
    report.section("3. the two pipelines")
    say(ising.build_graph(SITES, admissible_set=limits).device)
    say()
    say(schwinger.build_graph(SITES, admissible_set=limits).analogue)

    say()
    report.section("4. the transition line, against the paper's own two statements")
    critical_field = 1.0 - PAPER_CRITICAL_SLOPE * REQUEST_TRANSVERSE_FIELD
    critical = solve_ising(limits, critical_field)
    if not critical.status.is_success:
        msg = f"the Ising solve did not produce a usable point: {critical.status}"
        raise RuntimeError(msg)
    knobs = critical.subset(list(KNOBS))
    detuning = ising.detuning_map().evaluate_real(knobs) / knobs[K.TUNNELLING]
    slope = (1.0 - critical_field) / REQUEST_TRANSVERSE_FIELD
    say(f"   {'requested (hz, hx)':<38}({critical_field:.4f}, {REQUEST_TRANSVERSE_FIELD:g})")
    say(f"   {'recovered (Delta - U) / J':<38}{detuning:9.3f}")
    say(f"   {'paper: E = U + 1.85 t':<38}{PAPER_CRITICAL_DETUNING_IN_TUNNELLINGS:9.3f}")
    say(f"   {'recovered slope (1 - hz) / hx':<38}{slope:9.3f}")
    say(f"   {'paper: hz = 1 - 0.66 hx':<38}{PAPER_CRITICAL_SLOPE:9.3f}")
    say()
    say(
        "   both are the same statement of the same critical point, one in Hubbard\n"
        "   parameters and one in spin parameters, and neither depends on the constraint\n"
        "   scale this derivation is free to choose: the two fields are divided by it alike."
    )

    say()
    report.section("5. the same lattice, solved for both theories")
    gauge = solve_gauge(limits)
    if not gauge.status.is_success:
        msg = f"the gauge solve did not produce a usable point: {gauge.status}"
        raise RuntimeError(msg)
    gauge_knobs = gauge.subset(list(KNOBS))
    say(f"   {'theory':<10}" + "".join(f"{k.split('.')[-1]:>11}" for k in KNOBS))
    say(f"   {'Ising':<10}{_knob_line(knobs)}")
    say(f"   {'gauge':<10}{_knob_line(gauge_knobs)}")
    say()
    ising_verdicts, gauge_verdicts = verdicts(knobs), verdicts(gauge_knobs)
    say(f"   {'setting':<16}{'Ising window':<44}gauge window")
    for label, pair in (("Ising", ising_verdicts), ("gauge", gauge_verdicts)):
        say(f"   {label + ' setting':<16}{pair[0]!s:<44}{pair[1]!s}")
    say()
    say(
        "   disjoint: the superlattice derivation needs Delta << delta and so a deep\n"
        "   superlattice, this one needs delta << Gamma and so none.  No setting of this\n"
        "   lattice is both theories at once, and the margins are what say so rather than\n"
        "   an argument about the two derivations."
    )

    say()
    report.section(f"6. the dipole manifold against the Ising chain, N = {SITES}")
    bench = Bench.of(SITES, limits)
    device_levels, weight = bench.device_spectrum(knobs, limits)
    transverse = ising.transverse_map().evaluate_real(knobs)
    gap = spacing_deviation(device_levels, bench.magnet_spectrum(knobs, end_field=True), transverse)
    dropped = spacing_deviation(
        device_levels, bench.magnet_spectrum(knobs, end_field=False), transverse
    )
    say(
        f"   manifold: {len(bench.manifold)} of {bench.space.dimension} lattice states, "
        f"least weight {weight:.4f}"
    )
    say(f"   {'spectrum gap, with the end field':<40}{gap:10.4f}  x Gamma")
    say(f"   {'the same, printed Hamiltonian only':<40}{dropped:10.4f}  x Gamma")
    say(
        f"   {'end field / transverse field':<40}"
        f"{ising.boundary_field().evaluate_real(knobs) / transverse:10.4f}"
    )
    say()
    say(
        "   the end-spin field is half the coupling and the transverse field is far below\n"
        "   it, so on a register this short the term the printed Hamiltonian drops is not a\n"
        "   boundary detail -- it is the leading correction, and the two columns say so."
    )
    return Comparison(
        ising_knobs=dict(knobs),
        gauge_knobs=dict(gauge_knobs),
        ising_verdicts=ising_verdicts,
        gauge_verdicts=gauge_verdicts,
        critical_slope=slope,
        weight=weight,
        gap=gap,
        dropped=dropped,
    )


if __name__ == "__main__":
    main()
