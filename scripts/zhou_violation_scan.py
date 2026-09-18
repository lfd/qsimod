"""The gauge-violation scan of `paper_zhou22` (Fig. 2D) at a chosen chain length.

Follows the experiment's protocol: ``U`` and ``Delta`` are held at the prescribed knobs of
``examples/analogue_end_to_end.py``, ``J`` alone varies so that ``U / J`` runs over the
paper's grid, and each point is quenched from ``|1 0 1 0 1 ...>``.  Both models are realised
in the sector of ``N`` atoms.  Records, per point, the knobs, the effective ``(m, kappa)``,
the weakest validity ratio of the pipeline, the gauge violation averaged over the paper's
scan window of 120 ms and over the 150 ms window of the trajectory runs, and the leakage.

Writes ``results/zhou_violation_scan.csv``.  Run it with::

    uv run python scripts/zhou_violation_scan.py --matter-sites 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from examples.analogue_end_to_end import (
    PAPER_CASES,
    PAPER_INTERACTION_RATIOS,
    SAMPLES,
    SCAN_WINDOW_MS,
    WINDOW_MS,
    case_knobs,
    experiment_knobs,
)

from qsimod.realise import evolve_state, expectation
from qsimod.units import to_hertz
from qsimod.usecases.schwinger import ParameterNames as P
from qsimod.usecases.schwinger import coupling_map, mass_map, validity
from scripts import write_frame
from scripts.zhou_trajectories import NumberSectorBench

__all__ = ["main", "scan_point"]


def scan_point(
    bench: NumberSectorBench,
    knobs: dict[str, float],
    label: str,
) -> dict[str, object]:
    """One point of the scan: knobs, effective parameters, validity ratio and errors."""
    effective = {
        P.MASS_EFFECTIVE_BOSONIC: mass_map().evaluate_real(knobs),
        P.COUPLING_EFFECTIVE_BOSONIC: coupling_map().evaluate_real(knobs),
    }
    device = bench.build(bench.device, knobs)
    times = np.linspace(0.0, WINDOW_MS, SAMPLES)
    states = evolve_state(device, bench.initial, list(times))
    violation = np.asarray(expectation(bench.violation, states))
    leakage = np.asarray(1.0 - expectation(bench.projector, states))
    report = validity().report({**knobs, **effective})
    weakest = report.weakest_margin
    return {
        "case": label,
        "U_over_J": knobs[P.INTERACTION] / knobs[P.TUNNELLING],
        "delta_over_J": knobs[P.SUPERLATTICE] / knobs[P.TUNNELLING],
        "J_hz": to_hertz(knobs[P.TUNNELLING]),
        "U_hz": to_hertz(knobs[P.INTERACTION]),
        "delta_hz": to_hertz(knobs[P.SUPERLATTICE]),
        "Delta_hz": to_hertz(knobs[P.TILT]),
        "m_hz": to_hertz(effective[P.MASS_EFFECTIVE_BOSONIC]),
        "kappa_hz": to_hertz(effective[P.COUPLING_EFFECTIVE_BOSONIC]),
        "weakest_margin_decades": weakest,
        "weakest_ratio": 0.1 * 10.0 ** (-weakest),
        "violation_mean_120ms": float(violation[times <= SCAN_WINDOW_MS].mean()),
        "violation_mean_150ms": float(violation.mean()),
        "leakage_max_150ms": float(leakage.max()),
    }


def main(matter_sites: int = 6, output: Path | None = None) -> Path:
    """Run the scan and write it to CSV.

    Args:
        matter_sites: the chain length ``N``.
        output: the destination, or ``None`` for ``results/zhou_violation_scan.csv``.

    Returns:
        The path written to.

    """
    bench = NumberSectorBench.of(matter_sites)
    print(f"N = {matter_sites}, number-sector realisation, dimension {bench.dimension}", flush=True)
    experiment = experiment_knobs()
    named = {2 * staggering: name for name, staggering in PAPER_CASES}
    rows = []
    for ratio in PAPER_INTERACTION_RATIOS:
        row = scan_point(bench, case_knobs(experiment, ratio / 2), named.get(ratio, ""))
        rows.append(row)
        print(
            f"  U/J = {ratio:5.1f}  eta(120ms) = {row['violation_mean_120ms']:.4f}"
            f"  ratio = {row['weakest_ratio']:.3f}  {row['case']}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.insert(0, "matter_sites", matter_sites)
    return write_frame(frame, output, "zhou_violation_scan.csv")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matter-sites", type=int, default=6, help="the chain length N")
    parser.add_argument("--output", type=Path, default=None, help="destination CSV")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse()
    print(main(arguments.matter_sites, arguments.output))
