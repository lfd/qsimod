"""One tilted Bose-Hubbard chain solved for two theories, each setting evaluated against both.

The hardware model ``bose_hubbard`` is the target of the gauge theory of the case study and of
the antiferromagnetic Ising chain of Simon et al. (2011).  The same lattice is solved for both
requests, and each knob setting is evaluated against the declared regimes of both theories.
The requests and the verdicts live in [`examples.simon2011`][examples.simon2011].

Run it with::

    uv run python examples/shared_device.py
"""

from __future__ import annotations

from dataclasses import dataclass

from examples import Report
from examples.simon2011 import (
    PAPER_CRITICAL_SLOPE,
    REQUEST_TRANSVERSE_FIELD,
    SITES,
    Verdict,
    lattice_limits,
    solve_gauge,
    solve_ising,
    verdicts,
)
from qsimod.usecases import ising
from qsimod.usecases.ising import KNOBS


@dataclass(frozen=True)
class Comparison:
    """The two knob settings and their four evaluations.

    Attributes:
        ising_knobs: the knob setting realising the Ising chain.
        gauge_knobs: the knob setting realising the gauge theory.
        ising_verdicts: the Ising setting against the Ising and the gauge regime.
        gauge_verdicts: the gauge setting against the Ising and the gauge regime.

    """

    ising_knobs: dict[str, float]
    gauge_knobs: dict[str, float]
    ising_verdicts: tuple[Verdict, Verdict]
    gauge_verdicts: tuple[Verdict, Verdict]

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
        The two knob settings and the four evaluations.

    Raises:
        RuntimeError: if either solve fails to return a usable point.

    """
    report = Report(verbose=verbose)
    say = report.say
    limits = lattice_limits()

    report.section("1. one hardware node, two incoming transformations")
    graph = ising.shared_graph(SITES, admissible_set=limits)
    for edge in graph.incoming(ising.DEVICE_TARGET):
        say(f"   {edge}")
    say()
    say("   admissible set posed against:")
    for line in str(limits).split("; "):
        say(f"     {line}")

    report.section("2. the same lattice, solved for both theories")
    critical_field = 1.0 - PAPER_CRITICAL_SLOPE * REQUEST_TRANSVERSE_FIELD
    solves = {"Ising": solve_ising(limits, critical_field), "gauge": solve_gauge(limits)}
    for theory, result in solves.items():
        if not result.status.is_success:
            msg = f"the {theory} solve did not produce a usable point: {result.status}"
            raise RuntimeError(msg)
    knobs = {theory: result.subset(list(KNOBS)) for theory, result in solves.items()}
    say(f"   {'theory':<10}" + "".join(f"{knob.split('.')[-1]:>11}" for knob in KNOBS))
    for theory, setting in knobs.items():
        say(f"   {theory:<10}{_knob_line(setting)}")

    report.section("3. each setting against both declared regimes")
    judged = {theory: verdicts(setting) for theory, setting in knobs.items()}
    say(f"   {'setting':<16}{'Ising regime':<44}gauge regime")
    for theory, pair in judged.items():
        say(f"   {theory + ' setting':<16}{pair[0]!s:<44}{pair[1]!s}")
    say()
    say(
        "   the superlattice derivation needs Delta << delta, the dipole derivation\n"
        "   delta << Gamma: no setting of this lattice realises both theories at once."
    )
    return Comparison(
        ising_knobs=dict(knobs["Ising"]),
        gauge_knobs=dict(knobs["gauge"]),
        ising_verdicts=judged["Ising"],
        gauge_verdicts=judged["gauge"],
    )


if __name__ == "__main__":
    main()
