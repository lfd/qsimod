"""Reference data of Simon et al. (2011) and the numerical checks of the shared hardware model.

The two requests posed to the tilted Bose-Hubbard chain, the evaluation of a knob setting
against the declared regimes of both theories, the transition line of the reference, and the
comparison of the dipole manifold with the spectrum of the Ising chain.
``examples/shared_device.py`` uses the requests and the verdicts; the test suite uses the rest.
J. Simon et al., *Quantum simulation of antiferromagnetic spin chains in an optical lattice*,
Nature **472**, 307-312 (2011).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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

#: The requested Ising coupling; it equals the on-site interaction of the lattice.
REQUEST_COUPLING = 1.0

#: The requested transverse field; the relation holds near ``(hz, hx) = (1, 0)``.
REQUEST_TRANSVERSE_FIELD = 0.05

#: The transition line of the reference, ``E = U + 1.85 t`` and ``hz = 1 - 0.66 hx``.
PAPER_CRITICAL_DETUNING_IN_TUNNELLINGS = 1.85
PAPER_CRITICAL_SLOPE = 0.66

#: The request of the gauge theory: a massless fermion at the coupling of Zhou et al. (2022).
GAUGE_COUPLING = 0.0045
GAUGE_ELECTRIC_GAP = 0.5

#: The chain length: ``2N-1`` lattice sites and ``2N-2`` spins.
SITES = 3


def lattice_limits() -> AdmissibleSet:
    """The admissible set of the lattice, against which both requests are posed."""
    return ising.device_limits()


def solve_ising(limits: AdmissibleSet, longitudinal_field: float) -> SolveResult:
    """Solve the lattice for an Ising chain at a given point of the ``(hz, hx)`` plane."""
    return realise_parameters(
        ising.build_graph(SITES, admissible_set=limits).device,
        targets={
            K.COUPLING_ISING_MAGNET: REQUEST_COUPLING,
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
            S.MASS_LATTICE_QED: 0.0,
            S.COUPLING_QUANTUM_LINK_STAGGERED: GAUGE_COUPLING,
            S.ELECTRIC_GAP: GAUGE_ELECTRIC_GAP,
        },
        unknowns=list(KNOBS),
        admissible_set=limits,
        initial={S.TUNNELLING: 0.03, S.INTERACTION: 2.0, S.SUPERLATTICE: 1.0, S.TILT: 0.05},
    )


# ---------------------------------------------------------------------------
# Judging one setting against both theories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """The evaluation of one knob setting against the declared regime of one theory.

    Attributes:
        theory: the theory whose regime conditions were evaluated.
        valid: whether every validity condition holds.
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
    """The Ising parameters of a knob setting, by the forward maps of the dipole reduction."""
    coupling = ising.coupling_map().evaluate_real(knobs)
    return {
        **knobs,
        K.COUPLING_ISING_CHAIN: coupling,
        K.TRANSVERSE: ising.transverse_map().evaluate_real(knobs),
        K.LONGITUDINAL: ising.longitudinal_map().evaluate_real(
            {**knobs, K.COUPLING_ISING_CHAIN: coupling}
        ),
    }


def gauge_environment(knobs: dict[str, float]) -> dict[str, float]:
    """The gauge-theory parameters the same knob setting realises."""
    return {
        **knobs,
        S.MASS_EFFECTIVE_BOSONIC: schwinger.mass_map().evaluate_real(knobs),
        S.COUPLING_EFFECTIVE_BOSONIC: schwinger.coupling_map().evaluate_real(knobs),
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
# Two checks against the reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionLine:
    """The transition point, requested in spin parameters and read back in Hubbard parameters.

    Attributes:
        knobs: the knob setting at the requested point.
        slope: ``(1 - hz) / hx`` of the request; the reference quotes ``0.66``.
        detuning: ``(Delta - U) / J`` read back from the knobs; the reference quotes ``1.85``.

    """

    knobs: dict[str, float]
    slope: float
    detuning: float


def transition_line(limits: AdmissibleSet) -> TransitionLine:
    """Solve for the point just below the transition, ``hz = 1 - 0.66 hx``.

    Raises:
        RuntimeError: if the solve does not return a usable point.

    """
    critical_field = 1.0 - PAPER_CRITICAL_SLOPE * REQUEST_TRANSVERSE_FIELD
    result = solve_ising(limits, critical_field)
    if not result.status.is_success:
        msg = f"the Ising solve did not produce a usable point: {result.status}"
        raise RuntimeError(msg)
    knobs = result.subset(list(KNOBS))
    return TransitionLine(
        knobs=dict(knobs),
        slope=(1.0 - critical_field) / REQUEST_TRANSVERSE_FIELD,
        detuning=ising.detuning_map().evaluate_real(knobs) / knobs[K.TUNNELLING],
    )


@dataclass(frozen=True)
class Bench:
    """The numerical realisation in which the hardware model and the magnet are compared.

    The dipole manifold is the set of states reached from one atom per site by creating
    dipoles on non-adjacent bonds; its levels are selected by overlap with
    [`manifold_levels`][qsimod.realise.evolve.manifold_levels].

    Attributes:
        sites: the chain length ``N`` of the hardware model.
        space: the Hilbert space of the hardware model at a cutoff of two.
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
            # A dipole is a spin down; a spin-1/2 basis index is 0 down, 1 up.
            constrained.append(
                spin_space.index_of(tuple(0 if bond in occupied else 1 for bond in range(bonds)))
            )
        return cls(sites, space, tuple(manifold), spin_space, tuple(constrained))

    def device_spectrum(
        self, knobs: dict[str, float], limits: AdmissibleSet
    ) -> tuple[NDArray[np.float64], float]:
        """The dipole levels of the lattice, selected by overlap, and their least weight."""
        operator = build_operator(ising.lattice(self.sites, limits).hamiltonian, self.space, knobs)
        levels, weight = manifold_levels(operator, self.manifold)
        return np.asarray(levels, dtype=np.float64), weight

    def magnet_spectrum(self, knobs: dict[str, float], *, end_field: bool) -> NDArray[np.float64]:
        """The levels of the Ising chain on the constrained subspace.

        Args:
            knobs: the four hardware knobs.
            end_field: whether the end-spin field of the dipole reduction is added.

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


@dataclass(frozen=True)
class ManifoldCheck:
    """The dipole manifold of the lattice against the spectrum of the Ising chain.

    Attributes:
        weight: the least weight a compared level keeps in the dipole manifold.
        gap: the spectrum gap with the end-spin field, in units of the transverse field.
        dropped: the same gap without the end-spin field.

    """

    weight: float
    gap: float
    dropped: float


def manifold_check(bench: Bench, knobs: dict[str, float], limits: AdmissibleSet) -> ManifoldCheck:
    """Compare the dipole manifold with the Ising spectrum, with and without the end field."""
    device_levels, weight = bench.device_spectrum(knobs, limits)
    transverse = ising.transverse_map().evaluate_real(knobs)
    return ManifoldCheck(
        weight=weight,
        gap=spacing_deviation(
            device_levels, bench.magnet_spectrum(knobs, end_field=True), transverse
        ),
        dropped=spacing_deviation(
            device_levels, bench.magnet_spectrum(knobs, end_field=False), transverse
        ),
    )
