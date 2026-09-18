"""A request no knob setting of the admissible set can realise, reported as ``INFEASIBLE``.

The analogue pipeline of the Schwinger case study is posed a coupling far above the reach of a
hardware model with a capped tunnelling.  Interval arithmetic over the admissible ranges of the
knobs proves the request unreachable, and the binding constraints are reported.

Run it with::

    uv run python examples/infeasible_request.py
"""

from __future__ import annotations

from examples import Report
from qsimod.parameters import AdmissibleSet, Bound
from qsimod.solving import SolveStatus, realise_parameters
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import (
    build_graph,
    device_limits,
)

#: The requested effective parameters.
TARGET_COUPLING = 0.004525
TARGET_MASS = 0.02
ELECTRIC_GAP = 0.5

#: The cap on the tunnelling.  As ``kappa ~ J**2 / U``, it limits the coupling to order 1e-7.
TUNNELLING_CAP = 1e-4


def tightened_device() -> AdmissibleSet:
    """The admissible set of a Bose-Hubbard hardware model with a capped tunnelling.

    The ranges of the remaining knobs are narrow enough to keep the denominators of the coupling
    relation sign-definite, as the interval proof of infeasibility requires; see
    [`qsimod.solving.reachability`][qsimod.solving.reachability].
    """
    reference = device_limits()
    return AdmissibleSet(
        bounds=(
            Bound(P.TUNNELLING, 0.0, TUNNELLING_CAP, strict_lower=True),
            Bound(P.INTERACTION, 0.9, 1.1),
            Bound(P.SUPERLATTICE, 0.4, 0.6),
            Bound(P.TILT, 0.01, 0.05, strict_lower=True),
        ),
        constraints=reference.constraints,
        description="optical superlattice with a hard cap on the tunnelling",
    )


def main(*, verbose: bool = True) -> SolveStatus:
    """Pose the unreachable request to the analogue pipeline and report the result.

    Returns:
        The solve status, [`INFEASIBLE`][qsimod.solving.result.SolveStatus.INFEASIBLE].

    """
    graph = build_graph(3)
    device = tightened_device()
    say = Report(verbose=verbose).say

    say("requested:", f"m = {TARGET_MASS}, kappa = {TARGET_COUPLING}")
    say("device:   ", device)
    say()

    result = realise_parameters(
        graph.analogue,
        targets={
            P.MASS_LATTICE_QED: TARGET_MASS,
            P.COUPLING_QUANTUM_LINK_STAGGERED: TARGET_COUPLING,
            P.ELECTRIC_GAP: ELECTRIC_GAP,
        },
        unknowns=[P.TUNNELLING, P.INTERACTION, P.SUPERLATTICE, P.TILT],
        admissible_set=device,
    )
    say(result)
    say()
    say(f"status: {result.status}")
    say("binding constraint(s):")
    for constraint in result.binding_constraints:
        say(f"  - {constraint}")
    return result.status


if __name__ == "__main__":
    main()
