"""Pipelines and the model graph.

Transformations compose into a pipeline; a composed pipeline is itself a transformation.  A
[`Pipeline`][qsimod.pipeline.Pipeline] is a composite
[`Transformation`][qsimod.transform.Transformation]: the junctions are type-checked at
composition time, the pipeline is ``EXACT`` if and only if every constituent transformation is
exact, its validity conditions are the conjunction of those of its constituents, and its
parameter relation is the union of those of its constituents with intermediate parameters left
free.  A [`ModelGraph`][qsimod.pipeline.ModelGraph] is the directed acyclic graph of artifacts
and transformations from which pipelines are enumerated.  An
[`ErrorReport`][qsimod.pipeline.ErrorReport] reports the structural approximations, the
residual of a solve, the margins of the regime conditions and the resource cost on separate
axes.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from qsimod.artifact import Artifact, ArtifactKind, ArtifactKindError
from qsimod.relations import EMPTY_RELATION, ParameterRelation, RelationClassification
from qsimod.structure import StructureMismatch, StructureType
from qsimod.transform import (
    Approximation,
    ApproximationKind,
    Exactness,
    Transformation,
)
from qsimod.validity import Conjunction, ValidityReport

__all__ = [
    "BranchSummary",
    "CompositionError",
    "Edge",
    "ErrorReport",
    "ModelGraph",
    "Pipeline",
    "ResourceCost",
    "StructuralApproximation",
]


class CompositionError(TypeError):
    """Two transformations were composed whose types do not meet.

    Attributes:
        first: the name of the producing transformation.
        second: the name of the consuming transformation.
        gaps: the structural mismatches.
        kind_problem: a description of a mismatch of artifact kinds, or empty.

    """

    def __init__(
        self,
        first: str,
        second: str,
        gaps: Sequence[StructureMismatch],
        kind_problem: str = "",
    ) -> None:
        self.first = first
        self.second = second
        self.gaps = tuple(gaps)
        self.kind_problem = kind_problem
        details = [kind_problem] if kind_problem else []
        details += [str(gap) for gap in self.gaps]
        super().__init__(f"cannot compose {first!r} with {second!r}: " + "; ".join(details))


@dataclass(frozen=True)
class StructuralApproximation:
    """One approximate transformation, as it appears in an error report.

    Attributes:
        step: the name of the transformation.
        kind: ``regime-limited`` or ``resource-controlled``.
        leading_error_order: the leading surviving correction, where known.
        note: a further caveat.

    """

    step: str
    kind: ApproximationKind
    leading_error_order: str = ""
    note: str = ""

    def __str__(self) -> str:
        order = f"; leading error {self.leading_error_order}" if self.leading_error_order else ""
        return f"{self.step}: {self.kind}{order}"


@dataclass(frozen=True)
class ResourceCost:
    """The error-against-cost entry of a resource-controlled transformation.

    Attributes:
        step: the name of the transformation.
        settings: the resource parameters and their values, for instance
            ``{"n": 32, "order": 2}``.
        error_bound: the a-priori error bound at those settings.
        cost: hardware-agnostic resource counts, for instance the number of factors and the
            depth in layers.

    """

    step: str
    settings: Mapping[str, float]
    error_bound: float
    cost: Mapping[str, int]

    def __str__(self) -> str:
        settings = ", ".join(f"{k}={v:g}" for k, v in sorted(self.settings.items()))
        cost = ", ".join(f"{k}={v}" for k, v in sorted(self.cost.items()))
        return f"{self.step}: at ({settings}) bound {self.error_bound:.3e}, cost ({cost})"


@dataclass(frozen=True)
class ErrorReport:
    """The independent error axes of a pipeline; the axes are not summable.

    Attributes:
        structural: the approximate transformations and their kinds.
        regime: the validity report at the operating point.
        solver_residuals: the per-equation residual left by the parameter solve; empty when
            the solve was exact or no solve was needed.
        resource_costs: the error-against-cost entries of the resource-controlled
            transformations.

    """

    structural: tuple[StructuralApproximation, ...] = ()
    regime: ValidityReport = field(default_factory=lambda: ValidityReport(()))
    solver_residuals: Mapping[str, float] = field(default_factory=dict)
    resource_costs: tuple[ResourceCost, ...] = ()

    @property
    def approximation_kinds(self) -> frozenset[ApproximationKind]:
        """The set of approximation kinds along the pipeline."""
        return frozenset(entry.kind for entry in self.structural)

    @property
    def max_solver_residual(self) -> float:
        """The largest absolute solver residual, or ``0.0`` if there is none."""
        return max((abs(value) for value in self.solver_residuals.values()), default=0.0)

    def __str__(self) -> str:
        lines = ["error report (three independent axes; not summable)"]
        lines.append("  structural approximation:")
        if self.structural:
            lines += [f"    - {entry}" for entry in self.structural]
        else:
            lines.append("    - none: every step is EXACT")
        lines.append("  solver residual:")
        if self.solver_residuals:
            lines += [
                f"    - {name}: {value:+.3e}"
                for name, value in sorted(self.solver_residuals.items())
            ]
        else:
            lines.append("    - none")
        lines.append("  regime margins:")
        lines += [f"    {line}" for line in str(self.regime).splitlines()]
        if self.resource_costs:
            lines.append("  resource-controlled error vs cost:")
            lines += [f"    - {entry}" for entry in self.resource_costs]
        return "\n".join(lines)


def _check_junctions(steps: Sequence[Transformation]) -> None:
    """Check that the types of each adjacent pair meet, in artifact kind and in structural type.

    Raises:
        CompositionError: at the first junction whose types do not meet.

    """
    for first, second in pairwise(steps):
        kind_problem = ""
        if first.target_kind is not second.source_kind:
            kind_problem = (
                f"{first.name} produces a {first.target_kind} but {second.name} "
                f"consumes a {second.source_kind}"
            )
        gaps = first.target_pattern.guarantee_gaps(second.source_pattern)
        if gaps or kind_problem:
            raise CompositionError(first.name, second.name, gaps, kind_problem)


def _composite_approximation(steps: Sequence[Transformation]) -> Approximation:
    """The approximation declaration of a composite with at least one approximate constituent.

    The kind is ``resource-controlled`` if any constituent is, otherwise ``regime-limited``;
    the full set is
    [`Pipeline.approximation_kinds`][qsimod.pipeline.Pipeline.approximation_kinds].
    """
    approximations = [step.approximation for step in steps if step.approximation]
    kinds = {approximation.kind for approximation in approximations}
    dominant = (
        ApproximationKind.RESOURCE_CONTROLLED
        if ApproximationKind.RESOURCE_CONTROLLED in kinds
        else ApproximationKind.REGIME_LIMITED
    )
    resources = tuple(
        dict.fromkeys(
            name for approximation in approximations for name in approximation.resource_parameters
        )
    )
    return Approximation(
        kind=dominant,
        leading_error_order="; ".join(
            approximation.leading_error_order
            for approximation in approximations
            if approximation.leading_error_order
        ),
        resource_parameters=(
            resources if dominant is ApproximationKind.RESOURCE_CONTROLLED else ()
        ),
        note=f"composite of kinds: {', '.join(sorted(str(kind) for kind in kinds))}",
    )


def _composite_declarations(
    steps: Sequence[Transformation],
) -> tuple[ParameterRelation, Conjunction]:
    """The united parameter relation and the conjoined validity conditions of a composite.

    Equation names are prefixed by the index of the constituent transformation (``s0.``,
    ``s1.``, ...); parameter names are unchanged.
    """
    relation = EMPTY_RELATION
    validity = Conjunction()
    for index, step in enumerate(steps):
        relation = relation.compose(step.relation.prefixed(f"s{index}."))
        validity = validity.and_also(step.validity)
    return relation, validity


@dataclass(frozen=True)
class Pipeline(Transformation):
    """A composite transformation: an ordered chain of transformations that type-check pairwise.

    A composed pipeline is itself a transformation.  It is exact only if every constituent
    transformation is exact.

    Attributes:
        steps: the constituent transformations, source-most first.

    """

    steps: tuple[Transformation, ...] = ()

    @classmethod
    def of(cls, steps: Sequence[Transformation], name: str = "") -> Pipeline:
        """Compose ``steps`` into a pipeline, type-checking each junction.

        Args:
            steps: the transformations, source-most first.
            name: a name for the composite; by default the names of the constituents joined.

        Returns:
            The composite transformation.

        Raises:
            ValueError: if ``steps`` is empty.
            CompositionError: if the types of two adjacent transformations do not meet.

        """
        if not steps:
            msg = "a pipeline needs at least one step"
            raise ValueError(msg)
        _check_junctions(steps)

        exactness = (
            Exactness.EXACT if all(step.is_exact for step in steps) else Exactness.APPROXIMATE
        )
        relation, validity = _composite_declarations(steps)
        return cls(
            name=name or " -> ".join(step.name for step in steps),
            source_pattern=steps[0].source_pattern,
            target_pattern=steps[-1].target_pattern,
            exactness=exactness,
            source_kind=steps[0].source_kind,
            target_kind=steps[-1].target_kind,
            approximation=(
                None if exactness is Exactness.EXACT else _composite_approximation(steps)
            ),
            relation=relation,
            validity=validity,
            source_parameters=steps[0].source_parameters,
            target_parameters=steps[-1].target_parameters,
            description=f"composite of {len(steps)} step(s)",
            steps=tuple(steps),
        )

    def then(self, other: Transformation) -> Pipeline:
        """Extend the pipeline by one further transformation.

        Raises:
            CompositionError: if the types do not meet.

        """
        extra = other.steps if isinstance(other, Pipeline) else (other,)
        return Pipeline.of(self.steps + extra)

    def target_structure(self, source: StructureType) -> StructureType:
        """Propagate a structural type through every constituent transformation."""
        current = source
        for step in self.steps:
            current = step.target_structure(current)
        return current

    def _transform(self, artifact: Artifact) -> Artifact:
        """Apply every constituent transformation in order."""
        current = artifact
        for step in self.steps:
            current = step.apply(current)
        return current

    # -- the composite axes ------------------------------------------------------

    @property
    def approximation_kinds(self) -> frozenset[ApproximationKind]:
        """The set of approximation kinds along the pipeline."""
        return frozenset(
            step.approximation.kind for step in self.steps if step.approximation is not None
        )

    @property
    def approximate_steps(self) -> tuple[Transformation, ...]:
        """The constituent transformations that are ``APPROXIMATE``, in order."""
        return tuple(step for step in self.steps if not step.is_exact)

    def structural_approximations(self) -> tuple[StructuralApproximation, ...]:
        """The approximate transformations, as entries of an error report."""
        return tuple(
            StructuralApproximation(
                step=step.name,
                kind=step.approximation.kind,
                leading_error_order=step.approximation.leading_error_order,
                note=step.approximation.note,
            )
            for step in self.steps
            if step.approximation is not None
        )

    def dropped_constants(self) -> tuple[tuple[str, str], ...]:
        """The constituents discarding an additive constant, as ``(step, description)`` pairs."""
        return tuple(
            (step.name, step.dropped_constant) for step in self.steps if step.dropped_constant
        )

    def error_report(
        self,
        point: Mapping[str, float],
        solver_residuals: Mapping[str, float] | None = None,
        resource_costs: Sequence[ResourceCost] = (),
    ) -> ErrorReport:
        """Assemble the error report at an operating point.

        Args:
            point: the parameter values.
            solver_residuals: the per-equation residuals left by a parameter solve, if any.
            resource_costs: the error-against-cost entries of the resource-controlled
                transformations.

        Returns:
            The report.

        """
        return ErrorReport(
            structural=self.structural_approximations(),
            regime=self.validity.report(dict(point)),
            solver_residuals=dict(solver_residuals or {}),
            resource_costs=tuple(resource_costs),
        )

    def classify_relation(self, known: frozenset[str]) -> RelationClassification:
        """Classify the united relation of the whole pipeline in one direction."""
        return self.relation.classify(known=known)

    def sub_pipeline(self, start: int = 0, stop: int | None = None) -> Pipeline:
        """The pipeline formed by a contiguous slice of the constituent transformations."""
        return Pipeline.of(self.steps[start:stop])

    def with_step_replaced(self, index: int, step: Transformation) -> Pipeline:
        """A copy with one constituent replaced; the junctions are type-checked again.

        Raises:
            CompositionError: if the types of the new transformation do not meet those of
                its neighbours.

        """
        steps = list(self.steps)
        steps[index] = step
        return Pipeline.of(steps, name=self.name)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Transformation]:
        return iter(self.steps)

    def summary_lines(self) -> list[str]:
        """One line per constituent transformation, for printing a pipeline in a report."""
        return [f"  {index + 1}. {step}" for index, step in enumerate(self.steps)]

    def __str__(self) -> str:
        kinds = ", ".join(sorted(str(k) for k in self.approximation_kinds)) or "none"
        head = f"{self.name}: {self.exactness} [approximation kinds: {kinds}]"
        return "\n".join([head, *self.summary_lines()])


# ---------------------------------------------------------------------------
# The DAG
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """One directed edge of the model graph.

    Attributes:
        source: the name of the source artifact.
        target: the name of the target artifact.
        transformation: the transformation the edge carries.

    """

    source: str
    target: str
    transformation: Transformation

    def __str__(self) -> str:
        return f"{self.source} --[{self.transformation.name}]--> {self.target}"


@dataclass(frozen=True)
class BranchSummary:
    """One row of the side-by-side comparison of the branches of a model graph.

    Attributes:
        target: the name of the terminal artifact.
        pipeline: the pipeline that reaches it.
        artifact_kind: the kind of artifact the branch produces.
        exactness: the exactness of the composite.
        approximation_kinds: the set of approximation kinds along the branch.
        step_names: the names of the constituent transformations, in order.

    """

    target: str
    pipeline: Pipeline
    artifact_kind: ArtifactKind
    exactness: Exactness
    approximation_kinds: frozenset[ApproximationKind]
    step_names: tuple[str, ...]

    def __str__(self) -> str:
        kinds = ", ".join(sorted(str(k) for k in self.approximation_kinds)) or "none"
        return f"{self.target:<8} {self.artifact_kind!s:<16} {self.exactness!s:<12} kinds: {kinds}"


class ModelGraph:
    """The model graph: a directed acyclic graph of artifacts, keyed by name, and transformations.

    The graph is built incrementally with [`add_node`][qsimod.pipeline.ModelGraph.add_node]
    and [`add_edge`][qsimod.pipeline.ModelGraph.add_edge]; adding an edge type-checks the
    transformation against both of its endpoints.  An artifact may have several outgoing and
    several incoming edges.

    Attributes:
        name: the name of the graph, used in diagnostics.

    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._nodes: dict[str, Artifact] = {}
        self._edges: list[Edge] = []

    # -- construction ------------------------------------------------------------

    def add_node(self, artifact: Artifact) -> Artifact:
        """Register an artifact in the graph.

        Raises:
            ValueError: if an artifact of that name is already registered.

        """
        if artifact.name in self._nodes:
            msg = f"graph {self.name!r} already has a node named {artifact.name!r}"
            raise ValueError(msg)
        self._nodes[artifact.name] = artifact
        return artifact

    def add_edge(self, source: str, target: str, transformation: Transformation) -> Edge:
        """Register a transformation as an edge, type-checking both endpoints.

        Raises:
            KeyError: if an endpoint is not a registered artifact.
            ArtifactKindError: if the artifact kinds of the transformation do not match the
                endpoints.
            StructureTypeError: if the structure of an endpoint does not match.

        """
        source_node = self.node(source)
        target_node = self.node(target)
        transformation.check(source_node)
        if target_node.kind is not transformation.target_kind:
            raise ArtifactKindError(
                transformation.name, transformation.target_kind, target_node.kind, target_node.name
            )
        transformation.target_pattern.check(
            target_node.structure, f"{transformation.name} (target)"
        )
        edge = Edge(source, target, transformation)
        self._edges.append(edge)
        return edge

    # -- inspection --------------------------------------------------------------

    def node(self, name: str) -> Artifact:
        """The artifact of the given name.

        Raises:
            KeyError: if no artifact of that name is registered.

        """
        try:
            return self._nodes[name]
        except KeyError:
            msg = f"graph {self.name!r} has no node named {name!r}; has {sorted(self._nodes)}"
            raise KeyError(msg) from None

    @property
    def nodes(self) -> tuple[str, ...]:
        """The name of every registered artifact, in registration order."""
        return tuple(self._nodes)

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Every registered edge, in registration order."""
        return tuple(self._edges)

    def outgoing(self, node: str) -> tuple[Edge, ...]:
        """The edges leaving the artifact ``node``."""
        return tuple(edge for edge in self._edges if edge.source == node)

    def incoming(self, node: str) -> tuple[Edge, ...]:
        """The edges entering the artifact ``node``."""
        return tuple(edge for edge in self._edges if edge.target == node)

    def reachable(self, source: str) -> tuple[str, ...]:
        """Every artifact reachable from ``source``, excluding ``source`` itself, sorted by name."""
        seen: set[str] = set()
        frontier = [source]
        while frontier:
            current = frontier.pop()
            for edge in self.outgoing(current):
                if edge.target not in seen:
                    seen.add(edge.target)
                    frontier.append(edge.target)
        return tuple(sorted(seen))

    def terminal_targets(self, source: str) -> tuple[str, ...]:
        """The reachable artifacts without outgoing edges: the end points of the pipelines."""
        return tuple(name for name in self.reachable(source) if not self.outgoing(name))

    def paths(self, source: str, target: str) -> tuple[tuple[Edge, ...], ...]:
        """Every simple path from ``source`` to ``target``, as a sequence of edges."""
        found: list[tuple[Edge, ...]] = []

        def walk(current: str, trail: tuple[Edge, ...], visited: frozenset[str]) -> None:
            if current == target and trail:
                found.append(trail)
                return
            for edge in self.outgoing(current):
                if edge.target in visited:
                    continue
                walk(edge.target, (*trail, edge), visited | {edge.target})

        self.node(source)
        self.node(target)
        walk(source, (), frozenset({source}))
        return tuple(found)

    def pipelines(self, source: str, target: str) -> tuple[Pipeline, ...]:
        """Every distinct pipeline from ``source`` to ``target``."""
        return tuple(
            Pipeline.of(
                [edge.transformation for edge in path],
                name=f"{source} -> {target} via {' -> '.join(edge.target for edge in path)}",
            )
            for path in self.paths(source, target)
        )

    def branches(self, source: str) -> tuple[BranchSummary, ...]:
        """Side-by-side summaries of every pipeline to every terminal artifact.

        Resource costs are not included; see
        [`Pipeline.error_report`][qsimod.pipeline.Pipeline.error_report].
        """
        summaries: list[BranchSummary] = []
        for target in self.terminal_targets(source):
            for pipeline in self.pipelines(source, target):
                summaries.append(
                    BranchSummary(
                        target=target,
                        pipeline=pipeline,
                        artifact_kind=self.node(target).kind,
                        exactness=pipeline.exactness,
                        approximation_kinds=pipeline.approximation_kinds,
                        step_names=tuple(step.name for step in pipeline.steps),
                    )
                )
        return tuple(summaries)

    def with_binding(self, node: str, **values: float) -> Artifact:
        """Bind parameters on a registered artifact and return the bound copy."""
        return self.node(node).bind(**values)

    def replace_node(self, artifact: Artifact) -> Artifact:
        """Replace a registered artifact of the same name, keeping the edges.

        Raises:
            KeyError: if no artifact of that name is registered.

        """
        if artifact.name not in self._nodes:
            msg = f"graph {self.name!r} has no node named {artifact.name!r} to replace"
            raise KeyError(msg)
        self._nodes[artifact.name] = artifact
        return artifact

    def __str__(self) -> str:
        lines = [
            f"model graph {self.name!r}: {len(self._nodes)} node(s), {len(self._edges)} edge(s)"
        ]
        lines += [f"  {edge}" for edge in self._edges]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ModelGraph(name={self.name!r}, nodes={len(self._nodes)}, edges={len(self._edges)})"
