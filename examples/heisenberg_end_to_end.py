"""The Heisenberg pipeline, end to end, against a reference experiment.

The parameters of the application model are entered, the pipeline is type-checked, the
hardware knobs are solved for, and both layers are realised numerically and compared.  The
request is a setting of Jepsen et al., *Spin transport in a tunable Heisenberg model realized
with ultracold atoms*, Nature **588**, 403-407 (2020), whose Eq. (1) is the artifact ``xxz_chain``
and whose Methods section "Extended Hubbard model" is the superexchange transformation of the
framework.  Section 6 checks the forward map of the superexchange against the Methods table of the
reference, section 7 the free-fermion band of the Jordan-Wigner branch, and section 8 scans the
anisotropy across the admissible set of the hardware model.

Run it with::

    uv run python examples/heisenberg_end_to_end.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from examples import Report
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
# The experiment that already answered the question
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperRow:
    """One row of the Methods table of Jepsen et al. (2020), at a lattice depth of 11 recoils.

    Attributes:
        intraspecies_up: ``U_ud / U_uu``.
        intraspecies_down: ``U_ud / U_dd``.
        interspecies: ``U_ud / t``; negative on the attractive branch at which the apparatus
            operates.
        anisotropy: the ``Delta`` the reference quotes for that triple.

    """

    intraspecies_up: float
    intraspecies_down: float
    interspecies: float
    anisotropy: float


#: The table of the reference: "We use parameters from experiments and focus on a lattice depth
#: of 11 E_R, in which case we have U_ud/U_uu = 1.206, 1.406, ... and U_ud/t = -17.94, ... for
#: anisotropies Delta = 0.020, 0.670, ... respectively" (Methods, Numerical simulations).
PAPER_ROWS: tuple[PaperRow, ...] = (
    PaperRow(1.206, -0.188, -17.94, 0.020),
    PaperRow(1.406, 0.264, -24.32, 0.670),
    PaperRow(1.401, 0.459, -24.17, 0.860),
    PaperRow(1.398, 0.575, -24.08, 0.973),
    PaperRow(1.397, 0.659, -24.05, 1.055),
    PaperRow(1.392, 0.862, -23.94, 1.256),
)

#: The row at which the request is posed: the one nearest the isotropic Heisenberg point.
REQUEST_ROW = 3

#: The spin-exchange time the reference quotes at that lattice depth: "the curves collapse when
#: times are rescaled in units of hbar/Jxy = 0.75 ms, 2.01 ms, 5.08 ms for lattice depths of
#: 9 E_R, 11 E_R, 13 E_R" (Fig. 2a).  It fixes the energy scale of the ratios of the Methods
#: table.
PAPER_EXCHANGE_TIME_MS = 2.01

#: The anisotropy uncertainty of the reference: "The uncertainty of Delta is estimated to be
#: about +-0.1" (Methods, Determination of the Heisenberg parameters).
PAPER_ANISOTROPY_UNCERTAINTY = 0.1

#: The spin-exchange-time uncertainty of the reference: "lead to an uncertainty for the
#: spin-exchange times hbar/Jxy of about +-10%" (same section).
PAPER_EXCHANGE_UNCERTAINTY = 0.10

#: The anisotropies scanned in section 8.  The first four are settings at which the reference
#: reports data (Fig. 3a-b); the last three lie beyond the admissible set of the hardware model.
SCAN_ANISOTROPIES: tuple[float, ...] = (-1.43, -1.02, 0.0, 0.973, 1.58, 6.0, 20.0, 60.0)

#: The chain length at which the spectra are compared.  The hardware model is realised in its
#: conserved unit-filling sector (266 states at ``N = 4``, against ``3**8 = 6561`` for the full
#: space).
SITES = 4

#: The chain length of the free-fermion band in section 7 (analytic branch only, ``2**N``).
BAND_SITES = 10


def experiment_knobs(row: PaperRow, hopping: float) -> dict[str, float]:
    """One row of the Methods table, as the four hardware knobs.

    The row gives three ratios; the tunnelling at 11 recoils, from
    [`reference_hopping`][examples.heisenberg_end_to_end.reference_hopping], fixes the scale.

    Args:
        row: the table row.
        hopping: the tunnelling at that lattice depth, in rad/ms.

    Returns:
        ``t``, ``U_uu``, ``U_ud`` and ``U_dd``, in rad/ms.

    """
    mixed = row.interspecies * hopping
    return {
        P.HOPPING: hopping,
        P.INTERACTION_UP: mixed / row.intraspecies_up,
        P.INTERACTION_MIXED: mixed,
        P.INTERACTION_DOWN: mixed / row.intraspecies_down,
    }


def reference_hopping(row: PaperRow = PAPER_ROWS[REQUEST_ROW]) -> float:
    """The tunnelling at 11 recoils, from the spin-exchange time quoted by the reference.

    ``Jxy = -4 t**2 / U_ud`` with ``U_ud = (U_ud/t) t`` gives ``Jxy = -4 t / (U_ud/t)``.

    Args:
        row: the table row whose spin-exchange time is quoted.

    Returns:
        The tunnelling, in rad/ms.

    """
    return -row.interspecies * (1.0 / PAPER_EXCHANGE_TIME_MS) / 4.0


# ---------------------------------------------------------------------------
# What one knob setting is worth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """One knob setting and its measured errors.

    Attributes:
        label: the origin of the setting.
        knobs: the four hardware knobs, in rad/ms.
        transverse: the spin-exchange coupling ``Jxy`` the setting realises, in rad/ms.
        longitudinal: the ``ZZ`` coupling ``Jz`` the setting realises, in rad/ms.
        field: the longitudinal field that the superexchange also produces, in rad/ms; see
            [`superexchange_field`][qsimod.transformations.perturbative.superexchange_field].
        margin: the weakest regime margin of the superexchange, in decades (positive
            inside the declared regime).
        deviation: the largest difference between the Mott-manifold spectrum of the hardware
            model and the spectrum of the magnet, in units of ``Jxy``, with the derived field
            included.
        dropped: the same difference with the field omitted.
        weight: the least weight any compared level keeps inside the one-atom-per-site
            manifold (one in the perturbative limit).

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
        """``hbar / Jxy`` in ms, the unit in which the reference reports its dynamics."""
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


# ---------------------------------------------------------------------------
# The numerical check, run identically on either setting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bench:
    """The shared numerical realisation in which both knob settings are evaluated.

    The hardware model is realised in its conserved unit-filling sector and the magnet in its
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
        return cls(
            sites=sites,
            basis=basis,
            manifold=_manifold_positions(basis, sites),
            device=device_model.hamiltonian,
            magnet=magnet_model.hamiltonian,
            magnet_space=HilbertSpace.of(magnet_model.structure),
        )

    def manifold_levels(self, knobs: dict[str, float]) -> tuple[NDArray[np.float64], float]:
        """The one-atom-per-site levels of the hardware model, and the least weight they keep.

        Levels are selected by overlap; see
        [`manifold_levels`][qsimod.realise.evolve.manifold_levels].
        """
        operator = build_operator_in(self.device, self.basis, knobs)
        levels, weight = manifold_levels(operator, self.manifold)
        return np.asarray(levels, dtype=np.float64), weight

    def effective_spectrum(
        self,
        transverse: float,
        longitudinal: float,
        field: float,
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
        """Evaluate one knob setting: its regime margin and its measured spectral errors.

        The effective parameters follow from the knobs through the forward maps of
        the superexchange.

        Args:
            label: the origin of the setting.
            knobs: the four hardware knobs, in rad/ms.

        Returns:
            The outcome.

        """
        transverse = transverse_map().evaluate_real(knobs)
        longitudinal = longitudinal_map().evaluate_real(knobs)
        field = field_map().evaluate_real(knobs)
        effective = {
            P.TRANSVERSE_XXZ_CHAIN: transverse,
            P.LONGITUDINAL_XXZ_CHAIN: longitudinal,
        }
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


def _manifold_positions(basis: SectorBasis, sites: int) -> tuple[int, ...]:
    """The positions of the ``2**N`` one-atom-per-site configurations in the unit-filling sector.

    This is the manifold that
    [`mott_manifold_operators`][qsimod.models.magnetism.mott_manifold_operators] declares.
    """
    return tuple(
        position
        for position, configuration in enumerate(basis.configurations)
        if all(configuration[2 * site] + configuration[2 * site + 1] == 1 for site in range(sites))
    )


# ---------------------------------------------------------------------------
# Section 7: the free-fermion branch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Band:
    """The single-particle band of the Jordan-Wigner image at zero anisotropy.

    Attributes:
        sites: the chain length at which the band was computed.
        transverse: the ``Jxy`` at which it was built, in rad/ms.
        measured: the single-particle eigenvalues, ascending.
        predicted: ``-Jxy cos(k pi / (N+1))``, the open-chain form of ``E(q) = -Jxy cos(qa)``.
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
        """The measured bandwidth.

        On an open chain the momenta run from ``pi/(N+1)`` to ``N pi/(N+1)``; see
        [`expected_bandwidth`][examples.heisenberg_end_to_end.Band.expected_bandwidth].
        """
        return float(self.measured[-1] - self.measured[0])

    @property
    def expected_bandwidth(self) -> float:
        """``2 Jxy cos(pi/(N+1))``, the bandwidth of the open chain at this length."""
        return float(2.0 * self.transverse * np.cos(np.pi / (self.sites + 1)))

    @property
    def fermi_velocity(self) -> float:
        """``|dE/dq|`` at half filling, ``a Jxy`` for the free-fermion band.

        It is estimated by a finite difference between the two levels straddling the middle of
        the band, at lattice spacing one.  Jepsen et al. (2020) report their spin-wave speeds in
        this unit: "a characteristic velocity v = 0.76(1) v_F" (Fig. 2d).
        """
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

    At zero anisotropy the one-particle block of the fermion chain is compared with the
    open-chain band ``-Jxy cos(k pi / (N+1))``; at a non-zero anisotropy the full spectra of the
    spin chain and its fermionic image are compared.

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

    # Jordan-Wigner exactness at a non-zero anisotropy.
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


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """The results of one run.

    Attributes:
        solved: the knob setting found by the solve of the framework.
        experiment: the knob setting of the experiment, from the Methods table.
        table: each Methods-table row with the anisotropy the forward map returns for it.
        band: the free-fermion band of the analytic branch.
        scan: one entry per scanned anisotropy: the request, the solve status and the outcome
            where a point was found.

    """

    solved: Outcome
    experiment: Outcome
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


def _solve_for(anisotropy: float, transverse: float, pipeline: Pipeline) -> SolveResult:
    """Pose one request to the hardware model and solve it.

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


def _report_methods_table(
    report: Report,
    hopping: float,
) -> tuple[tuple[PaperRow, float], ...]:
    """Section 6: the forward map of the superexchange against every row of the table."""
    report.say()
    report.section("6. the forward map against the paper's own Methods table")
    table = tuple(
        (
            entry,
            longitudinal_map().evaluate_real(experiment_knobs(entry, hopping))
            / transverse_map().evaluate_real(experiment_knobs(entry, hopping)),
        )
        for entry in PAPER_ROWS
    )
    report.say(
        f"   {'U_ud/U_uu':>10}{'U_ud/U_dd':>11}{'U_ud/t':>9}"
        f"{'Delta quoted':>14}{'Delta computed':>16}{'gap':>9}"
    )
    for entry, computed in table:
        report.say(
            f"   {entry.intraspecies_up:10.3f}{entry.intraspecies_down:11.3f}"
            f"{entry.interspecies:9.2f}{entry.anisotropy:14.3f}{computed:16.3f}"
            f"{computed - entry.anisotropy:+9.3f}"
        )
    report.say()
    report.say(
        f"   {'largest gap':<34}{max(abs(c - e.anisotropy) for e, c in table):9.4f}"
        f"   (paper quotes Delta to +-{PAPER_ANISOTROPY_UNCERTAINTY:g})"
    )
    return table


def _report_band(report: Report, transverse: float) -> Band:
    """Section 7: the free-fermion band of the analytic branch, and the Jordan-Wigner exactness."""
    report.say()
    report.section("7. the analytic branch: the free-fermion band")
    band = free_fermion_band(BAND_SITES, transverse)
    report.say(f"   chain of {band.sites} sites, at Jxy = {band.transverse:.4f} rad/ms")
    report.say(f"   {'band vs -Jxy cos(k pi/(N+1))':<40}{band.band_gap:12.3e} rad/ms")
    report.say(
        f"   {'bandwidth, 2 Jxy cos(pi/(N+1))':<40}{band.bandwidth:12.6f}"
        f"   expected {band.expected_bandwidth:.6f}"
    )
    report.say(
        f"   {'Fermi velocity at half filling':<40}{band.fermi_velocity:12.6f}"
        f"   expected {band.transverse:.6f}  (a Jxy, by a finite difference)"
    )
    report.say(
        f"   {'Jordan-Wigner exactness, full spectra':<40}{band.exactness_gap:12.3e} rad/ms"
        "   (claim: EXACT)"
    )
    return band


def _report_scan(
    report: Report,
    bench: Bench,
    pipeline: Pipeline,
    transverse: float,
    anisotropies: tuple[float, ...],
) -> tuple[tuple[float, SolveStatus, Outcome | None], ...]:
    """Section 8: solve across a grid of anisotropies and evaluate each point found."""
    report.say()
    report.section("8. anisotropy scan: how far the lattice reaches")
    report.say(
        f"   {'Delta':>8}{'status':>18}{'t/U':>9}{'margin':>9}"
        f"{'field/Jxy':>11}{'weight':>9}{'deviation':>11}"
    )
    scan: list[tuple[float, SolveStatus, Outcome | None]] = []
    for target in anisotropies:
        result = _solve_for(target, transverse, pipeline)
        outcome = (
            bench.assess(f"Delta={target:g}", result.subset(list(KNOBS)))
            if result.status.is_success
            else None
        )
        scan.append((target, result.status, outcome))
        if outcome is None:
            report.say(f"   {target:8.2f}{result.status!s:>18}" + " " * 9 + "   out of reach")
        else:
            report.say(
                f"   {target:8.2f}{result.status!s:>18}{outcome.expansion_parameter:9.4f}"
                f"{outcome.margin:+9.2f}{outcome.field / outcome.transverse:11.3f}"
                f"{outcome.weight:9.4f}{outcome.deviation:11.4f}"
            )
    report.say()
    report.say(
        "   Delta + 1 = U_ud/U_uu + U_ud/U_dd, so a large anisotropy needs a small\n"
        "   intraspecies channel -- and a small channel is a small Mott gap.  The\n"
        "   admissible set's Mott-lobe constraints are what eventually bind."
    )
    report.say()
    report.say(
        "   not reproduced: the transport exponents of Figs. 3 and 4, which need a\n"
        "   spin-helix quench on a chain far longer than a dense realisation reaches."
    )
    return tuple(scan)


def main(
    sites: int = SITES,
    *,
    anisotropies: tuple[float, ...] = SCAN_ANISOTROPIES,
    verbose: bool = True,
) -> Comparison:
    """Run the Heisenberg pipeline and compare its knob setting with that of the experiment.

    Args:
        sites: the chain length ``N`` at which the spectra are compared.
        anisotropies: the grid scanned in section 8.
        verbose: whether the report is printed.

    Returns:
        The two knob settings, the Methods-table comparison, the band and the scan.

    Raises:
        RuntimeError: if the principal solve does not return a usable point.

    """
    graph = build_graph(sites)
    pipeline = graph.device
    report = Report(verbose=verbose)
    say = report.say

    row = PAPER_ROWS[REQUEST_ROW]
    hopping = reference_hopping()
    experiment_setting = experiment_knobs(row, hopping)
    request_transverse = transverse_map().evaluate_real(experiment_setting)

    report.section("1. the graph, and the two branches")
    say(graph)
    say()
    for branch in graph.graph.branches("xxz_magnet"):
        say(f"   {branch}")
    say()
    say(pipeline)

    targets = {P.TRANSVERSE_XXZ_MAGNET: request_transverse, P.ANISOTROPY: row.anisotropy}
    say()
    report.section("2. composite relation")
    say(pipeline.classify_relation(frozenset(targets)))

    result = _solve_for(row.anisotropy, request_transverse, pipeline)
    say()
    report.section(
        f"3. solve for Jxy = 1/{PAPER_EXCHANGE_TIME_MS:g} rad/ms, Delta = {row.anisotropy:g}"
    )
    say(result)
    if not result.status.is_success:
        msg = f"the solve did not produce a usable point: {result.status}"
        raise RuntimeError(msg)

    say()
    report.section("4. error axes")
    say(pipeline.error_report(result.point, solver_residuals=result.residuals))

    say()
    report.section(f"5. knobs and errors, N = {sites}")
    bench = Bench.of(sites)
    solved = bench.assess("framework", result.subset(list(KNOBS)))
    experiment = bench.assess("experiment", experiment_setting)
    say(
        f"   device sector: {bench.basis.dimension} of "
        f"{bench.basis.space.dimension} states, "
        f"{len(bench.manifold)} manifold levels compared"
    )
    say()
    say(
        f"   {'setting':<11}{'t':>8}{'U_uu':>10}{'U_ud':>10}{'U_dd':>10}"
        f"{'Jxy':>9}{'Jz':>9}{'Delta':>9}   (rad/ms)"
    )
    for outcome in (solved, experiment):
        say(f"   {outcome.label:<11}{outcome.knob_line()}")
    say()
    say(
        f"   {'setting':<11}{'t/U':>9}{'margin':>9}{'field/Jxy':>11}"
        f"{'weight':>9}{'deviation':>11}{'dropped':>10}"
    )
    for outcome in (solved, experiment):
        say(f"   {outcome.label:<11}{outcome.measurement_line()}")

    table = _report_methods_table(report, hopping)
    band = _report_band(report, request_transverse)
    scan = _report_scan(report, bench, pipeline, request_transverse, anisotropies)
    return Comparison(
        solved=solved,
        experiment=experiment,
        table=table,
        band=band,
        scan=scan,
    )


if __name__ == "__main__":
    main()
