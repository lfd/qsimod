"""Use cases: model graphs assembled from the model and transformation libraries.

A use case selects models from [`qsimod.models`][qsimod.models] at given conventions, relates
them by transformations from [`qsimod.transformations`][qsimod.transformations], wires them
into a [`ModelGraph`][qsimod.pipeline.ModelGraph], and names their parameter namespaces.

* [`schwinger`][qsimod.usecases.schwinger]: the case study, a lattice Schwinger model carried
  to an analogue and a digital simulator model.
* [`heisenberg`][qsimod.usecases.heisenberg]: a Heisenberg magnet on a two-component lattice.
* [`ising`][qsimod.usecases.ising]: an Ising chain on the hardware model of the case study.

Each module defines ``ParameterNames``, ``build_graph`` and ``NAMESPACES``; only the symbols
of the first module are re-exported from this package.
"""

from qsimod.usecases import heisenberg, ising, schwinger
from qsimod.usecases.schwinger import (
    ALTERNATIVE_DIGITAL_TARGET,
    ANALOGUE_TARGET,
    DIGITAL_TARGET,
    NAMESPACES,
    SOURCE,
    ParameterNames,
    SchwingerGraph,
    analogue_pipeline,
    build_graph,
    digital_pipeline,
)

__all__ = [
    "ALTERNATIVE_DIGITAL_TARGET",
    "ANALOGUE_TARGET",
    "DIGITAL_TARGET",
    "NAMESPACES",
    "SOURCE",
    "ParameterNames",
    "SchwingerGraph",
    "analogue_pipeline",
    "build_graph",
    "digital_pipeline",
    "heisenberg",
    "ising",
    "schwinger",
]
