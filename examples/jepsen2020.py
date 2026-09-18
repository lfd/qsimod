"""Reference data of Jepsen et al. (2020) and the numerical checks of the Heisenberg branch.

The Methods table of the reference, its knobs, the numerical realisation in which a knob
setting is evaluated, and three checks: the forward map against every row of the table, the
free-fermion band of the Jordan-Wigner branch, and a scan of the anisotropy across the
admissible set.  ``examples/heisenberg_end_to_end.py`` uses the request and the realisation;
the test suite uses the rest.  P. N. Jepsen et al., *Spin transport in a tunable Heisenberg
model realized with ultracold atoms*, Nature **588**, 403-407 (2020).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qsimod.artifact import as_hamiltonian
from qsimod.pipeline import Pipeline
from qsimod.realise import (
    HilbertSpace,
    RealisationRequest,
    SectorBasis,
    build_operator,
    build_operator_in,
    manifold_levels,
    spacing_deviation,
)
from qsimod.solving import SolveResult, SolveStatus, realise_parameters
from qsimod.symbolic import OperatorSum
from qsimod.usecases.heisenberg import (
    FERMION_TARGET,
    KNOBS,
    build_graph,
    device_limits,
    device_start,
    field_map,
    lattice,
    longitudinal_map,
    spin_chain,
    transverse_map,
    validity,
    weighted_magnetisation_term,
)
from qsimod.usecases.heisenberg import (
    ParameterNames as P,
)

# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperRow:
    """One row of the Methods table of the reference, at a lattice depth of 11 recoils.

    Attributes:
        intraspecies_up: ``U_ud / U_uu``.
        intraspecies_down: ``U_ud / U_dd``.
        interspecies: ``U_ud / t``; negative on the attractive branch.
        anisotropy: the ``Delta`` the reference quotes for that triple.

    """

    intraspecies_up: float
    intraspecies_down: float
    interspecies: float
    anisotropy: float


#: The Methods table ("Numerical simulations") at 11 recoils.
PAPER_ROWS: tuple[PaperRow, ...] = (
    PaperRow(1.206, -0.188, -17.94, 0.020),
    PaperRow(1.406, 0.264, -24.32, 0.670),
    PaperRow(1.401, 0.459, -24.17, 0.860),
    PaperRow(1.398, 0.575, -24.08, 0.973),
    PaperRow(1.397, 0.659, -24.05, 1.055),
    PaperRow(1.392, 0.862, -23.94, 1.256),
)

#: The row at which the request is posed: the one nearest the isotropic point.
REQUEST_ROW = 3

#: The spin-exchange time ``hbar/Jxy`` at 11 recoils (Fig. 2a), which fixes the energy scale.
PAPER_EXCHANGE_TIME_MS = 2.01

#: The quoted uncertainties: ``+-0.1`` on ``Delta`` and ``+-10 %`` on the exchange time.
PAPER_ANISOTROPY_UNCERTAINTY = 0.1
PAPER_EXCHANGE_UNCERTAINTY = 0.10

#: The anisotropies of the scan; the last three lie beyond the reach of the lattice.
SCAN_ANISOTROPIES: tuple[float, ...] = (-1.43, -1.02, 0.0, 0.973, 1.58, 6.0, 20.0, 60.0)

#: The chain length at which the spectra are compared, and that of the free-fermion band.
SITES = 4
BAND_SITES = 10


def experiment_knobs(row: PaperRow, hopping: float) -> dict[str, float]:
    """One row of the Methods table as the four hardware knobs, in rad/ms."""
    mixed = row.interspecies * hopping
    return {
        P.HOPPING: hopping,
        P.INTERACTION_UP: mixed / row.intraspecies_up,
        P.INTERACTION_MIXED: mixed,
        P.INTERACTION_DOWN: mixed / row.intraspecies_down,
    }


def reference_hopping(row: PaperRow = PAPER_ROWS[REQUEST_ROW]) -> float:
    """The tunnelling at 11 recoils from the quoted exchange time, ``Jxy = -4 t / (U_ud/t)``."""
    return -row.interspecies * (1.0 / PAPER_EXCHANGE_TIME_MS) / 4.0


def solve_for(anisotropy: float, transverse: float, pipeline: Pipeline) -> SolveResult:
    """Solve the hardware model for one ``(Jxy, Delta)`` request.

    The starting point depends on the anisotropy; see
    [`device_start`][qsimod.usecases.heisenberg.device_start].
    """
    return realise_parameters(
        pipeline,
        targets={P.TRANSVERSE_XXZ_MAGNET: transverse, P.ANISOTROPY: anisotropy},
        unknowns=list(KNOBS),
        admissible_set=device_limits(),
        initial=device_start(anisotropy),
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
        transverse: the ``Jxy`` the setting realises, in rad/ms.
        longitudinal: the ``Jz`` the setting realises, in rad/ms.
        field: the longitudinal field the superexchange also produces, in rad/ms.
        margin: the weakest regime margin of the superexchange, in decades.
        deviation: the largest difference between the Mott-manifold spectrum of the hardware
            model and the spectrum of the magnet, in units of ``Jxy``, with the field included.
        dropped: the same difference with the field omitted.
        weight: the least weight any compared level keeps inside the one-atom-per-site
            manifold.

    """

    label: str
    knobs: dict[str, float]
    transverse: float
    longitudinal: float
    field: float
    margin: float
    deviation: float
    dropped: float
    weight: float

    @property
    def anisotropy(self) -> float:
        """``Delta = Jz / Jxy``."""
        return self.longitudinal / self.transverse

    @property
    def exchange_time_ms(self) -> float:
        """``hbar / Jxy`` in ms."""
        return 1.0 / self.transverse

    @property
    def expansion_parameter(self) -> float:
        """The largest ``t / |U|`` over the three interaction channels."""
        return max(
            abs(self.knobs[P.HOPPING] / self.knobs[channel])
            for channel in (P.INTERACTION_UP, P.INTERACTION_MIXED, P.INTERACTION_DOWN)
        )

    def knob_line(self) -> str:
        """The knobs and the effective parameters they realise, as one table row."""
        return (
            f"{self.knobs[P.HOPPING]:8.3f}{self.knobs[P.INTERACTION_UP]:10.2f}"
            f"{self.knobs[P.INTERACTION_MIXED]:10.2f}{self.knobs[P.INTERACTION_DOWN]:10.2f}"
            f"{self.transverse:9.4f}{self.longitudinal:9.4f}{self.anisotropy:9.4f}"
        )

    def measurement_line(self) -> str:
        """The declared margin and the measured errors, as one table row."""
        return (
            f"{self.expansion_parameter:9.4f}{self.margin:+9.2f}"
            f"{self.field / self.transverse:11.3f}{self.weight:9.4f}"
            f"{self.deviation:11.4f}{self.dropped:10.4f}"
        )


@dataclass(frozen=True)
class Bench:
    """The numerical realisation in which knob settings are evaluated.

    The hardware model is realised in its unit-filling sector and the magnet in its
    ``2**N``-dimensional space; the two are compared as spectra by
    [`spacing_deviation`][qsimod.realise.evolve.spacing_deviation].

    Attributes:
        sites: the chain length ``N``.
        basis: the unit-filling sector of the hardware model.
        manifold: the positions of the one-atom-per-site configurations in that sector.
        device: the Hamiltonian of the hardware model, with unbound parameters.
        magnet: the Hamiltonian of the magnet, with unbound parameters.
        magnet_space: the ``2**N``-dimensional Hilbert space of the magnet.

    """

    sites: int
    basis: SectorBasis
    manifold: tuple[int, ...]
    device: OperatorSum
    magnet: OperatorSum
    magnet_space: HilbertSpace

    @classmethod
    def of(cls, sites: int) -> Bench:
        """Realise both layers of the hardware branch at a given chain length."""
        device_model = lattice(sites)
        magnet_model = spin_chain(sites)
        basis = SectorBasis.of(
            device_model.structure,
            constraints=device_model.constraint_operators,
            request=RealisationRequest(boson_cutoff=2),
        )
        manifold = tuple(
            position
            for position, configuration in enumerate(basis.configurations)
            if all(configuration[2 * s] + configuration[2 * s + 1] == 1 for s in range(sites))
        )
        return cls(
            sites=sites,
            basis=basis,
            manifold=manifold,
            device=device_model.hamiltonian,
            magnet=magnet_model.hamiltonian,
            magnet_space=HilbertSpace.of(magnet_model.structure),
        )

    def manifold_levels(self, knobs: dict[str, float]) -> tuple[NDArray[np.float64], float]:
        """The one-atom-per-site levels of the hardware model, and the least weight they keep."""
        operator = build_operator_in(self.device, self.basis, knobs)
        levels, weight = manifold_levels(operator, self.manifold)
        return np.asarray(levels, dtype=np.float64), weight

    def effective_spectrum(
        self, transverse: float, longitudinal: float, field: float
    ) -> NDArray[np.float64]:
        """The spectrum of the magnet, with the derived longitudinal field added if non-zero."""
        operator = np.asarray(
            build_operator(
                self.magnet,
                self.magnet_space,
                {P.TRANSVERSE_XXZ_CHAIN: transverse, P.LONGITUDINAL_XXZ_CHAIN: longitudinal},
            )
        )
        if field:
            operator = operator + np.asarray(
                build_operator(
                    weighted_magnetisation_term(self.sites, field), self.magnet_space, {}
                )
            )
        return np.sort(np.real(np.linalg.eigvalsh(operator))).astype(np.float64)

    def assess(self, label: str, knobs: dict[str, float]) -> Outcome:
        """Evaluate one knob setting: its regime margin and its measured spectral errors."""
        transverse = transverse_map().evaluate_real(knobs)
        longitudinal = longitudinal_map().evaluate_real(knobs)
        field = field_map().evaluate_real(knobs)
        effective = {P.TRANSVERSE_XXZ_CHAIN: transverse, P.LONGITUDINAL_XXZ_CHAIN: longitudinal}
        device, weight = self.manifold_levels(knobs)
        return Outcome(
            label=label,
            knobs=dict(knobs),
            transverse=transverse,
            longitudinal=longitudinal,
            field=field,
            margin=validity().report({**knobs, **effective}).weakest_margin,
            deviation=spacing_deviation(
                device, self.effective_spectrum(transverse, longitudinal, field), transverse
            ),
            dropped=spacing_deviation(
                device, self.effective_spectrum(transverse, longitudinal, 0.0), transverse
            ),
            weight=weight,
        )


# ---------------------------------------------------------------------------
# Three checks against the reference
# ---------------------------------------------------------------------------


def methods_table(hopping: float) -> tuple[tuple[PaperRow, float], ...]:
    """Every Methods-table row with the anisotropy the forward map returns for it."""
    table = []
    for row in PAPER_ROWS:
        knobs = experiment_knobs(row, hopping)
        computed = longitudinal_map().evaluate_real(knobs) / transverse_map().evaluate_real(knobs)
        table.append((row, computed))
    return tuple(table)


@dataclass(frozen=True)
class Band:
    """The single-particle band of the Jordan-Wigner image at zero anisotropy.

    Attributes:
        sites: the chain length at which the band was computed.
        transverse: the ``Jxy`` at which it was built, in rad/ms.
        measured: the single-particle eigenvalues, ascending.
        predicted: ``-Jxy cos(k pi / (N+1))``, the open-chain band.
        exactness_gap: the largest difference between the full spectra of the spin chain and
            its fermionic image at a non-zero anisotropy, in rad/ms.

    """

    sites: int
    transverse: float
    measured: NDArray[np.float64]
    predicted: NDArray[np.float64]
    exactness_gap: float

    @property
    def bandwidth(self) -> float:
        """The measured bandwidth."""
        return float(self.measured[-1] - self.measured[0])

    @property
    def expected_bandwidth(self) -> float:
        """``2 Jxy cos(pi/(N+1))``, the bandwidth of the open chain."""
        return float(2.0 * self.transverse * np.cos(np.pi / (self.sites + 1)))

    @property
    def fermi_velocity(self) -> float:
        """``|dE/dq|`` at half filling by a finite difference; ``a Jxy`` for the free band."""
        middle = self.sites // 2
        momenta = np.pi * np.arange(1, self.sites + 1) / (self.sites + 1)
        return float(
            abs(self.measured[middle] - self.measured[middle - 1])
            / abs(momenta[middle] - momenta[middle - 1])
        )

    @property
    def band_gap(self) -> float:
        """The largest deviation of the measured band from the predicted one, in rad/ms."""
        return float(np.max(np.abs(self.measured - self.predicted)))


def free_fermion_band(sites: int, transverse: float) -> Band:
    """The free-fermion band of the analytic branch, and the exactness of the Jordan-Wigner step.

    Args:
        sites: the chain length.
        transverse: the ``Jxy`` at which to build, in rad/ms.

    Returns:
        The band.

    """
    graph = build_graph(sites)
    magnet = as_hamiltonian(graph.graph.node("xxz_chain"))
    fermions = as_hamiltonian(graph.graph.node(FERMION_TARGET))
    spin_space = HilbertSpace.of(magnet.structure)
    fermion_space = HilbertSpace.of(fermions.structure)

    free = {P.TRANSVERSE_FERMION_CHAIN: transverse, P.LONGITUDINAL_FERMION_CHAIN: 0.0}
    operator = np.asarray(build_operator(fermions.hamiltonian, fermion_space, free))
    single = [
        index
        for index in range(fermion_space.dimension)
        if sum(fermion_space.configuration(index)) == 1
    ]
    measured = np.sort(np.real(np.linalg.eigvalsh(operator[np.ix_(single, single)]))).astype(
        np.float64
    )
    momenta = np.pi * np.arange(1, sites + 1) / (sites + 1)
    predicted = np.sort(-transverse * np.cos(momenta)).astype(np.float64)

    interacting = 0.7 * transverse
    spin = np.linalg.eigvalsh(
        build_operator(
            magnet.hamiltonian,
            spin_space,
            {P.TRANSVERSE_XXZ_CHAIN: transverse, P.LONGITUDINAL_XXZ_CHAIN: interacting},
        )
    )
    image = np.linalg.eigvalsh(
        build_operator(
            fermions.hamiltonian,
            fermion_space,
            {P.TRANSVERSE_FERMION_CHAIN: transverse, P.LONGITUDINAL_FERMION_CHAIN: interacting},
        )
    )
    gap = float(np.max(np.abs(np.sort(np.asarray(spin)) - np.sort(np.asarray(image)))))
    return Band(sites, transverse, measured, predicted, gap)


def anisotropy_scan(
    bench: Bench,
    pipeline: Pipeline,
    transverse: float,
    anisotropies: tuple[float, ...],
) -> tuple[tuple[float, SolveStatus, Outcome | None], ...]:
    """Solve across a grid of anisotropies and evaluate each point found.

    Returns:
        One entry per anisotropy: the request, the solve status, and the outcome where a
        point was found.

    """
    scan: list[tuple[float, SolveStatus, Outcome | None]] = []
    for target in anisotropies:
        result = solve_for(target, transverse, pipeline)
        outcome = (
            bench.assess(f"Delta={target:g}", result.subset(list(KNOBS)))
            if result.status.is_success
            else None
        )
        scan.append((target, result.status, outcome))
    return tuple(scan)


@dataclass(frozen=True)
class Checks:
    """The three checks against the reference.

    Attributes:
        table: each Methods-table row with the anisotropy the forward map returns for it.
        band: the free-fermion band of the analytic branch.
        scan: one entry per scanned anisotropy: request, solve status, outcome if found.

    """

    table: tuple[tuple[PaperRow, float], ...]
    band: Band
    scan: tuple[tuple[float, SolveStatus, Outcome | None], ...]

    @property
    def worst_table_gap(self) -> float:
        """The largest disagreement with a quoted anisotropy, over the whole table."""
        return max(abs(computed - row.anisotropy) for row, computed in self.table)

    @property
    def reached(self) -> tuple[Outcome, ...]:
        """The outcomes of the scan where a point was found, in scan order."""
        return tuple(outcome for _, _, outcome in self.scan if outcome is not None)


def reproduce_checks(
    sites: int = SITES,
    anisotropies: tuple[float, ...] = SCAN_ANISOTROPIES,
) -> Checks:
    """Run the three checks at the request of the reference.

    Args:
        sites: the chain length ``N`` at which spectra are compared in the scan.
        anisotropies: the grid of the scan.

    Returns:
        The checks.

    """
    hopping = reference_hopping()
    transverse = transverse_map().evaluate_real(experiment_knobs(PAPER_ROWS[REQUEST_ROW], hopping))
    pipeline = build_graph(sites).device
    return Checks(
        table=methods_table(hopping),
        band=free_fermion_band(BAND_SITES, transverse),
        scan=anisotropy_scan(Bench.of(sites), pipeline, transverse, anisotropies),
    )
