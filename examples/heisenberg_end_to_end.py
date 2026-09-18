"""The Heisenberg pipeline, end to end.

A setting of Jepsen et al. (2020), the row of their Methods table nearest the isotropic point,
is posed to the pipeline: the composite relation is classified, the knobs of the two-component
lattice are solved for, the errors are reported, and the solved knobs are compared with the
knobs of the experiment by realising both layers numerically.  The reference data and the
numerical realisation live in [`examples.jepsen2020`][examples.jepsen2020].

Run it with::

    uv run python examples/heisenberg_end_to_end.py
"""

from __future__ import annotations

from dataclasses import dataclass

from examples import Report
from examples.jepsen2020 import (
    PAPER_EXCHANGE_TIME_MS,
    PAPER_ROWS,
    REQUEST_ROW,
    SITES,
    Bench,
    Outcome,
    experiment_knobs,
    reference_hopping,
    solve_for,
)
from qsimod.usecases.heisenberg import KNOBS, build_graph, transverse_map
from qsimod.usecases.heisenberg import ParameterNames as P


@dataclass(frozen=True)
class Comparison:
    """The two knob settings of one run.

    Attributes:
        solved: the setting found by the solve of the framework.
        experiment: the setting of the experiment, from the Methods table.

    """

    solved: Outcome
    experiment: Outcome


def main(sites: int = SITES, *, verbose: bool = True) -> Comparison:
    """Solve the Heisenberg pipeline for the request and compare with the experiment's knobs.

    Args:
        sites: the chain length ``N`` at which the spectra are compared.
        verbose: whether the report is printed.

    Returns:
        The solved and the experiment's knob setting, each with its measured errors.

    Raises:
        RuntimeError: if the solve does not return a usable point.

    """
    report = Report(verbose=verbose)
    say = report.say
    graph = build_graph(sites)
    pipeline = graph.device

    row = PAPER_ROWS[REQUEST_ROW]
    experiment_setting = experiment_knobs(row, reference_hopping())
    transverse = transverse_map().evaluate_real(experiment_setting)
    targets = {P.TRANSVERSE_XXZ_MAGNET: transverse, P.ANISOTROPY: row.anisotropy}

    report.section("1. the graph, its two branches, and the composite relation")
    say(graph)
    say(pipeline.classify_relation(frozenset(targets)))

    report.section(
        f"2. solve for Jxy = 1/{PAPER_EXCHANGE_TIME_MS:g} rad/ms, Delta = {row.anisotropy:g}"
    )
    result = solve_for(row.anisotropy, transverse, pipeline)
    say(result)
    if not result.status.is_success:
        msg = f"the solve did not produce a usable point: {result.status}"
        raise RuntimeError(msg)

    report.section("3. error axes")
    say(pipeline.error_report(result.point, solver_residuals=result.residuals))

    report.section(f"4. solved against experiment knobs, N = {sites}")
    bench = Bench.of(sites)
    solved = bench.assess("framework", result.subset(list(KNOBS)))
    experiment = bench.assess("experiment", experiment_setting)
    say(
        f"   device sector: {bench.basis.dimension} of {bench.basis.space.dimension} states, "
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
    return Comparison(solved=solved, experiment=experiment)


if __name__ == "__main__":
    main()
