"""The analogue branch of the Schwinger case study, end to end.

The request of Zhou et al. (2022), a massless fermion at ``kappa = 14.5 Hz``, is posed to the
analogue pipeline: the composite relation is classified, the knobs of the Bose-Hubbard chain
are solved for, the errors are reported on their separate axes, and the solved knobs are
compared with the knobs the experiment used by realising both layers numerically.  The
reference data and the numerical realisation live in
[`examples.zhou2022`][examples.zhou2022].

Run it with::

    uv run python examples/analogue_end_to_end.py
"""

from __future__ import annotations

from dataclasses import dataclass

from examples import Report
from examples.zhou2022 import (
    ELECTRIC_GAP,
    TARGET_COUPLING_HZ,
    TARGET_MASS_HZ,
    WINDOW_MS,
    Bench,
    Outcome,
    device_box,
    experiment_knobs,
)
from qsimod.solving import realise_parameters
from qsimod.units import from_hertz
from qsimod.usecases.schwinger import ParameterNames as P
from qsimod.usecases.schwinger import build_graph

#: The hardware knobs solved for.
KNOBS = (P.TUNNELLING, P.INTERACTION, P.SUPERLATTICE, P.TILT)


@dataclass(frozen=True)
class Comparison:
    """The two knob settings of one run.

    Attributes:
        solved: the setting found by the solve of the framework.
        experiment: the prescribed setting of the experiment.

    """

    solved: Outcome
    experiment: Outcome


def main(matter_sites: int = 3, *, verbose: bool = True) -> Comparison:
    """Solve the analogue pipeline for the request and compare with the experiment's knobs.

    Args:
        matter_sites: the chain length ``N`` of the numerical comparison.
        verbose: whether the report is printed.

    Returns:
        The solved and the prescribed knob setting, each with its measured errors.

    Raises:
        RuntimeError: if the solve does not return a usable point.

    """
    report = Report(verbose=verbose)
    say = report.say
    pipeline = build_graph(matter_sites).analogue
    targets = {
        P.MASS_LATTICE_QED: from_hertz(TARGET_MASS_HZ),
        P.COUPLING_QUANTUM_LINK_STAGGERED: from_hertz(TARGET_COUPLING_HZ),
        P.ELECTRIC_GAP: ELECTRIC_GAP,
    }

    report.section("1. pipeline and composite relation")
    say(pipeline)
    say(pipeline.classify_relation(frozenset(targets)))

    report.section(f"2. solve for m = {TARGET_MASS_HZ:g} Hz, kappa = {TARGET_COUPLING_HZ:g} Hz")
    result = realise_parameters(
        pipeline, targets=targets, unknowns=list(KNOBS), admissible_set=device_box()
    )
    say(result)
    if not result.status.is_success:
        msg = f"the solve did not produce a usable point: {result.status}"
        raise RuntimeError(msg)

    report.section("3. error axes")
    say(pipeline.error_report(result.point, solver_residuals=result.residuals))

    report.section(f"4. solved against prescribed knobs, N = {matter_sites}, {WINDOW_MS:g} ms")
    bench = Bench.of(matter_sites)
    solved, _ = bench.assess("framework", result.subset(list(KNOBS)))
    experiment, _ = bench.assess("experiment", experiment_knobs())
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
    return Comparison(solved=solved, experiment=experiment)


if __name__ == "__main__":
    main()
