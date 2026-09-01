"""The model graph object every use case returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from qsimod.levels import AbstractionLevel
from qsimod.pipeline import ModelGraph

__all__ = ["UseCaseGraph"]


@dataclass(frozen=True)
class UseCaseGraph:
    """The model graph of a use case together with the pipelines through it.

    Attributes:
        source: the artifact from which every pipeline of the use case starts; a class
            attribute.
        graph: the model graph.

    """

    source: ClassVar[str] = ""

    graph: ModelGraph

    def targets(self) -> tuple[str, ...]:
        """Every terminal artifact reachable from the source."""
        return self.graph.terminal_targets(self.source)

    def levels(self) -> dict[str, AbstractionLevel]:
        """The abstraction level of every artifact, by name."""
        return {name: self.graph.node(name).level for name in self.graph.nodes}

    def __str__(self) -> str:
        return "\n".join(
            [
                self.graph.name,
                *(f"  {edge}" for edge in self.graph.edges),
                f"  terminal targets from {self.source}: {', '.join(self.targets())}",
            ]
        )
