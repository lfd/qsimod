"""Numerical validation of the case study, and the generated documentation figures.

The modules of this package are the entry points that produce the numerical results and the
figures of the accompanying article.  Each exposes a ``main`` and is runnable directly with
``uv run python scripts/<name>.py``.  The numerical studies write a tidy ``pandas`` frame to
``results/``, from which ``plots/plot.r`` draws the figures:

* [`zhou_trajectories`][scripts.zhou_trajectories]: the reproduction of the analogue reference
  experiment; the knob settings of the analogue simulator model and the trajectories of the
  mean matter occupation, the deviation, the leakage and the gauge violation over the
  experimental time window.
* [`zhou_digital`][scripts.zhou_digital]: the digital branch at the same request; the
  resources and error bounds of the product formula over a ladder of step counts, the step
  counts required to meet given deviation thresholds, and the trajectories of the
  Trotterised evolution.
* [`figures`][scripts.figures]: the model-graph figures of the documentation, generated from
  the graphs the use cases build.

Shared here: [`write_frame`][scripts.write_frame] and [`RESULTS`][scripts.RESULTS].
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    import pandas as pd

__all__ = ["RESULTS", "write_frame"]

#: Default output directory of the numerical studies, relative to the working directory.
RESULTS = Path("results")


def write_frame(frame: pd.DataFrame, output: Path | None, default_name: str) -> Path:
    """Write a frame to CSV, creating the directory, and return the path written.

    Args:
        frame: the tidy frame.
        output: the destination, or ``None`` for the default.
        default_name: the file name under [`RESULTS`][scripts.RESULTS] used when
            ``output`` is ``None``.

    Returns:
        The path written to.

    """
    destination = output or (RESULTS / default_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
