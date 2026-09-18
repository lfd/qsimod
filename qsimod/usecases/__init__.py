"""Use cases: model graphs assembled from the model and transformation libraries.

A use case selects models from [`qsimod.models`][qsimod.models] at given conventions, relates
them by transformations from [`qsimod.transformations`][qsimod.transformations], wires them
into a [`ModelGraph`][qsimod.pipeline.ModelGraph], and names their parameter namespaces.

* [`schwinger`][qsimod.usecases.schwinger]: the case study of the article, a lattice Schwinger
  model in 1+1 dimensions carried to an analogue and a digital simulator model from one shared
  prefix of transformations.
* [`heisenberg`][qsimod.usecases.heisenberg]: an anisotropic Heisenberg magnet carried to a
  two-component optical lattice.
* [`ising`][qsimod.usecases.ising]: an antiferromagnetic Ising chain carried to the same
  artifact ``bose_hubbard`` the gauge theory reaches; its
  [`shared_graph`][qsimod.usecases.ising.shared_graph] holds both theories.

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
