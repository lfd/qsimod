"""Generate the model-graph figures of the documentation from the use cases' graphs.

Each use case builds a [`ModelGraph`][qsimod.pipeline.ModelGraph] of artifacts and
transformations.  This script renders those graphs as mermaid figures and writes them into
the pages that carry them: the overview figure on the README and the landing page, and the
pipeline figure at the top of each guide.  The captions follow the notation of the
accompanying article; per-node and per-edge overrides live in
[`CAPTIONS`][scripts.figures.CAPTIONS] and [`STEP_LABELS`][scripts.figures.STEP_LABELS], and
the nodes left out of the overview are listed in [`OMITTED`][scripts.figures.OMITTED].

Run it with::

    uv run python scripts/figures.py
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from qsimod.artifact import Artifact, ArtifactKind
from qsimod.levels import AbstractionLevel
from qsimod.pipeline import Edge, ModelGraph
from qsimod.transform import Exactness
from qsimod.usecases import heisenberg, ising, schwinger

__all__ = [
    "CAPTIONS",
    "FIGURE_PAGES",
    "OMITTED",
    "STEP_LABELS",
    "Caption",
    "figures",
    "main",
    "overview",
    "use_case_figure",
]

#: The pages that carry the overview figure, relative to the repository root.
FIGURE_PAGES = (Path("README.md"), Path("docs/index.md"))

#: The guide page of each use case, and the foreign ``(source, target)`` edges drawn greyed
#: for context.
GUIDE_FIGURES: tuple[tuple[Path, str, tuple[tuple[str, str], ...]], ...] = (
    (Path("docs/schwinger.md"), "schwinger", ()),
    (Path("docs/heisenberg.md"), "heisenberg", ()),
    (Path("docs/ising.md"), "ising", (("effective_bosonic", "bose_hubbard"),)),
)

#: Artifacts left out of the overview figure: the second digital branch of the case study.
OMITTED = frozenset({"qubit_register_staggered", "trotter_staggered"})

#: The caption of each abstraction layer's frame, and the module its models come from.
LEVEL_MODULES = {
    AbstractionLevel.APPLICATION: ("Application model", "models.application"),
    AbstractionLevel.INTERMEDIATE: ("Intermediate representations", "models.intermediate"),
    AbstractionLevel.HARDWARE: ("Hardware model", "models.hardware"),
}

#: The layers whose frame lays its boxes out left to right.
HORIZONTAL_LEVELS = frozenset({AbstractionLevel.APPLICATION, AbstractionLevel.HARDWARE})


@dataclass(frozen=True)
class Caption:
    """The text of an artifact's box.

    Attributes:
        title: the operator's name, in HTML.
        lines: the description, one entry per rendered line.
        parameters: the italic line naming the parameter set; empty if none.

    """

    title: str
    lines: tuple[str, ...] = ()
    parameters: str = ""

    def render(self, node: str) -> str:
        """The box label of the node carrying this caption."""
        parts = [f"<b>{node} &nbsp; {self.title}</b>", *self.lines]
        if self.parameters:
            parts.append(f"<i>{self.parameters}</i>")
        return "<br/>".join(parts)


#: Per-node caption overrides, in the article's notation.  A node with no entry is captioned
#: by its artifact's own names.
CAPTIONS: dict[str, Caption] = {
    "lattice_qed": Caption(
        "H<sub>sys</sub>",
        ("lattice QED", "(Kogut–Susskind)"),
        "Θ<sub>sys</sub> = {m, a, e}",
    ),
    "quantum_link_staggered": Caption(
        "H<sub>IR1</sub>",
        ("quantum-link model,", "staggered mass"),
        "Θ<sub>IR1</sub> = {m, κ}",
    ),
    "quantum_link_homogeneous": Caption(
        "H<sub>IR2</sub>",
        ("quantum-link model,", "pair coupling"),
        "Θ<sub>IR2</sub> = {m, κ}",
    ),
    "effective_bosonic": Caption(
        "H<sub>IR3</sub>", ("boson encoding",), "Θ<sub>IR3</sub> = {m, κ}"
    ),
    "qubit_register": Caption(
        "H<sub>IR4</sub>", ("qubit Hamiltonian",), "Θ<sub>IR4</sub> = {m, κ}"
    ),
    "bose_hubbard": Caption(
        "H<sub>sim</sub>",
        ("tilted, staggered", "Bose–Hubbard chain"),
        "Θ<sub>sim</sub> = {J, U, δ, Δ}",
    ),
    "trotter": Caption("U<sub>≈</sub>", ("Trotter product formula",), "t, n, order"),
    "qubit_register_staggered": Caption(
        "H<sub>IR4</sub><sup>st</sup>",
        ("qubit Hamiltonian,", "from the staggered form"),
        "m, κ",
    ),
    "trotter_staggered": Caption(
        "U<sub>≈</sub><sup>st</sup>",
        ("Trotter product formula,", "from the staggered form"),
        "t, n, order",
    ),
    "xxz_magnet": Caption(
        "H<sub>XXZ</sub>",
        ("Heisenberg XXZ magnet,", "by its anisotropy"),
        "Θ = {J<sub>xy</sub>, Δ}",
    ),
    "xxz_chain": Caption(
        "H<sub>XXZ</sub>",
        ("XXZ chain,", "by its two couplings"),
        "Θ = {J<sub>xy</sub>, J<sub>z</sub>}",
    ),
    "fermion_chain": Caption(
        "H<sub>tV</sub>",
        ("spinless fermions,", "nearest-neighbour interaction"),
        "Θ = {J<sub>xy</sub>, J<sub>z</sub>}",
    ),
    "two_component_bose_hubbard": Caption(
        "H<sub>2BHM</sub>",
        ("two-component", "Bose–Hubbard chain"),
        "Θ = {t, U<sub>↑↑</sub>, U<sub>↑↓</sub>, U<sub>↓↓</sub>}",
    ),
    "ising_magnet": Caption(
        "H<sub>Ising</sub>",
        ("antiferromagnetic", "Ising chain"),
        "Θ = {J<sub>z</sub>, h<sub>z</sub>, h<sub>x</sub>}",
    ),
    "ising_chain": Caption(
        "H<sub>Ising</sub>", ("Ising chain,", "by three energies"), "Θ = {J<sub>z</sub>, Γ, B}"
    ),
}

#: Per-edge caption overrides, keyed by ``(source, target)``: the transformation's name as
#: the figure prints it, and the exactness line under it.  Absent entries use the transformation's
#: own name and declared exactness.
STEP_LABELS: dict[tuple[str, str], tuple[str, str]] = {
    ("lattice_qed", "quantum_link_staggered"): (
        "quantum-link truncation",
        "approximate, regime conditions",
    ),
    ("quantum_link_staggered", "quantum_link_homogeneous"): (
        "particle–hole transformation",
        "exact",
    ),
    ("quantum_link_homogeneous", "effective_bosonic"): (
        "boson encoding",
        "exact on the encoded subspace",
    ),
    ("effective_bosonic", "bose_hubbard"): (
        "degenerate perturbation theory,<br/>solved for the knob settings",
        "approximate, regime conditions",
    ),
    ("quantum_link_homogeneous", "qubit_register"): ("Jordan–Wigner transformation", "exact"),
    ("qubit_register", "trotter"): ("Trotterisation", "approximate, resource-controlled"),
    ("quantum_link_staggered", "qubit_register_staggered"): (
        "Jordan–Wigner transformation,<br/>from the staggered form",
        "exact",
    ),
    ("qubit_register_staggered", "trotter_staggered"): (
        "Trotterisation, order 2",
        "approximate, resource-controlled",
    ),
    ("xxz_chain", "two_component_bose_hubbard"): (
        "second-order superexchange,<br/>solved for the knob settings",
        "approximate, regime conditions",
    ),
    ("ising_chain", "bose_hubbard"): (
        "resonant dipole reduction,<br/>solved for the knob settings",
        "approximate, regime conditions",
    ),
}


def use_case_graphs(matter_sites: int = 3) -> dict[str, ModelGraph]:
    """The model graph of every use case, in overview order."""
    return {
        "schwinger": schwinger.build_graph(matter_sites).graph,
        "heisenberg": heisenberg.build_graph(matter_sites).graph,
        "ising": ising.build_graph(matter_sites).graph,
    }


def _nodes(graphs: Sequence[ModelGraph], omit: frozenset[str]) -> dict[str, Artifact]:
    """Every drawn artifact, sorted by abstraction level and then in first-seen order."""
    found: dict[str, Artifact] = {}
    for graph in graphs:
        for name in graph.nodes:
            if name not in found and name not in omit:
                found[name] = graph.node(name)
    by_level = sorted(found.items(), key=lambda entry: entry[1].level.value)
    return dict(by_level)


def _edges(graphs: Sequence[ModelGraph], omit: frozenset[str]) -> Iterator[tuple[ModelGraph, Edge]]:
    """Every drawn transformation, grouped by the use case that declares it."""
    seen: set[tuple[str, str]] = set()
    for graph in graphs:
        for edge in graph.edges:
            key = (edge.source, edge.target)
            if key in seen or omit & set(key):
                continue
            seen.add(key)
            yield graph, edge


def _fallback_caption(artifact: Artifact) -> Caption:
    """The caption of an artifact without a [`CAPTIONS`][scripts.figures.CAPTIONS] entry."""
    title = re.sub(r"_(\w+)", r"<sub>\1</sub>", artifact.structure.name or artifact.name)
    parameters = ", ".join(name.rsplit(".", 1)[-1] for name in artifact.parameters.names)
    return Caption(title, (artifact.origin,) if artifact.origin else (), parameters)


def _node_lines(nodes: dict[str, Artifact]) -> Iterator[str]:
    """The frame of each abstraction layer and the boxes inside it."""
    for level, (title, module) in LEVEL_MODULES.items():
        drawn = [name for name, artifact in nodes.items() if artifact.level is level]
        if not drawn:
            continue
        frame = f"LV{level.value}"
        yield f'    subgraph {frame}["{title} &nbsp;·&nbsp; <code>{module}</code>"]'
        if level in HORIZONTAL_LEVELS:
            yield "        direction LR"
        for name in drawn:
            caption = CAPTIONS.get(name) or _fallback_caption(nodes[name])
            yield f'        {name}["{caption.render(name)}"]'
        yield "    end"


def _edge_line(edge: Edge) -> str:
    """One arrow: dotted for an approximate transformation, solid for an exact one."""
    step = edge.transformation
    caption, word = STEP_LABELS.get(
        (edge.source, edge.target), (_default_caption(step.name), _default_word(step))
    )
    label = f"<b>{caption}</b><br/><i>{word}</i>"
    if step.exactness is Exactness.EXACT:
        return f'    {edge.source} ---> |"{label}"| {edge.target}'
    return f'    {edge.source} -. "{label}" .-> {edge.target}'


def _default_caption(name: str) -> str:
    """A transformation's name, with the typographic substitutions the figure uses."""
    for plain, typeset in (
        ("spin-1/2", "spin-½"),
        ("Jordan-Wigner", "Jordan–Wigner"),
        ("Kogut-Susskind", "Kogut–Susskind"),
        ("Suzuki-Trotter", "Suzuki–Trotter"),
        ("particle-hole", "particle–hole"),
        ("Bose-Hubbard", "Bose–Hubbard"),
    ):
        name = name.replace(plain, typeset)
    return name


def _default_word(step: object) -> str:
    """The exactness line under an arrow: ``"exact"``, or the kind of approximation."""
    kind = getattr(step, "approximation_kind", None)
    return "exact" if kind is None else f"approximate, {kind}"


#: Fixed part of the overview figure: the numerical realisation and solving offered at a
#: hardware model, and the executable layer that is out of scope.  Guide figures do not carry
#: it.
CHROME = """    subgraph OUT["Numerical realisation and solving"]
        direction LR
        SOLVE["<b>parameter realisation</b><br/>a constrained solve for the knob settings\
<br/>exact · approximate · infeasible · unsolved"]
        VALID["<b>validity report</b><br/>every regime condition,<br/>with its margin"]
        NUM["<b>numerical realisation</b><br/>dense operators in JAX,<br/>spectra and dynamics"]
        SOLVE ~~~ VALID ~~~ NUM
    end
    L4(["Executable layer: gates, routing, pulses<br/><i>out of scope</i>"])
"""

#: The chrome's arrows, emitted with the chrome.
CHROME_STYLE = """    LV3 ==> |"a hardware model with bound parameters"| OUT
    LV3 -.-> L4
"""

#: Node classes: Hamiltonian, product formula, foreign context node, and the two classes of
#: the chrome.  A figure declares only the classes it uses.
#:
#: The hues are the ``lfd`` palette of the article's TikZ figures (``tikz.tex``), so a node
#: carries the same colour here as in the paper: ``lfd4`` #009371 for a Hamiltonian (the
#: ``hamiltonian`` style), ``lfd2`` #E69F00 for a product formula (``unitaryblock``, the
#: gates and pulses), ``lfd6`` #ED665A for the executable layer (``instructionset``),
#: ``lfd3`` #999999 for what the paper de-emphasises, ``lfd7`` #1F78B4 for an abstraction
#: level (``abstractbox``), and ``lfd5`` #BEAED4, the one palette colour the TikZ styles
#: leave unclaimed, for the realisation and solving lane.  Each fill is a tint of its colour
#: on white and each text a shade of it, the way ``lfd4!12`` would read in TikZ; a saturated
#: fill with white text is unreadable at the size these nodes render.
CLASS_DEFS = {
    "ham": "classDef ham fill:#e0f2ee,stroke:#009371,color:#003e2f",
    "pf": "classDef pf fill:#fbf1d9,stroke:#c48700,color:#573c00",
    "ctx": "classDef ctx fill:#f3f3f3,stroke:#919191,color:#545454,stroke-dasharray:4 3",
    "svc": "classDef svc fill:#ede8f3,stroke:#988baa,color:#4c4655",
    "oos": "classDef oos fill:#fdebea,stroke:#ed665a,color:#642b26",
}

#: Shading of a layer frame, and of the chrome's frame.
LEVEL_STYLE = "    style LV{level} fill:#f2f7fa,stroke:#b1d0e5,color:#103c5a"
OUT_STYLE = "    style OUT fill:#f6f4f9,stroke:#d8cee5,color:#5f576a"


def _class_lines(
    nodes: dict[str, Artifact], context: frozenset[str], *, chrome: bool
) -> Iterator[str]:
    """The class definitions a figure needs, followed by the nodes assigned to each."""
    buckets = {
        "ham": [
            name
            for name, artifact in nodes.items()
            if artifact.kind is ArtifactKind.HAMILTONIAN and name not in context
        ],
        "pf": [
            name
            for name, artifact in nodes.items()
            if artifact.kind is ArtifactKind.PRODUCT_FORMULA and name not in context
        ],
        "ctx": sorted(context, key=list(nodes).index),
    }
    assignments = {name: ",".join(members) for name, members in buckets.items() if members}
    if chrome:
        assignments["svc"] = "SOLVE,VALID,NUM"
        assignments["oos"] = "L4"
    for name in assignments:
        yield "    " + CLASS_DEFS[name]
    for name, members in assignments.items():
        yield f"    class {members} {name}"


def _figure(
    graphs: Sequence[ModelGraph],
    *,
    omit: frozenset[str] = frozenset(),
    context: Sequence[tuple[Artifact, Edge]] = (),
    chrome: bool = False,
) -> str:
    """Assemble one mermaid block, fences included.

    Args:
        graphs: the model graphs whose artifacts and transformations are drawn.
        omit: node names to leave out, together with every edge that touches one.
        context: foreign ``(source node, edge)`` pairs drawn greyed; the source node is
            added, the target is expected to belong to ``graphs``.
        chrome: whether to draw [`CHROME`][scripts.figures.CHROME].

    Returns:
        The fenced block.

    """
    nodes = _nodes(graphs, omit)
    foreign = {edge.source: artifact for artifact, edge in context if edge.source not in nodes}
    nodes = dict(sorted({**nodes, **foreign}.items(), key=lambda entry: entry[1].level.value))

    lines = ["```mermaid", "flowchart TB", *_node_lines(nodes)]
    if chrome:
        lines.append(CHROME.rstrip("\n"))
    lines.append("")

    current: ModelGraph | None = None
    for graph, edge in _edges(graphs, omit):
        if current is not None and graph is not current:
            lines.append("")
        current = graph
        lines.append(_edge_line(edge))
    for _, edge in context:
        lines.append("")
        lines.append(_edge_line(edge))

    lines.append("")
    if chrome:
        lines.append(CHROME_STYLE.rstrip("\n"))
        lines.append("")
    lines += _class_lines(nodes, frozenset(foreign), chrome=chrome)
    lines += [
        LEVEL_STYLE.format(level=level.value)
        for level in LEVEL_MODULES
        if any(artifact.level is level for artifact in nodes.values())
    ]
    if chrome:
        lines.append(OUT_STYLE)
    lines.append("```")
    return "\n".join(lines)


def overview(matter_sites: int = 3) -> str:
    """The overview figure: every use case, arranged by abstraction layer."""
    return _figure(list(use_case_graphs(matter_sites).values()), omit=OMITTED, chrome=True)


def use_case_figure(
    name: str,
    context: Sequence[tuple[str, str]] = (),
    matter_sites: int = 3,
) -> str:
    """The model-graph figure of one use case.

    Args:
        name: the use case, as [`use_case_graphs`][scripts.figures.use_case_graphs] keys it.
        context: foreign transformations to draw greyed, as ``(source, target)`` pairs,
            looked up in whichever other use case declares them.
        matter_sites: the chain length the graphs are built at.

    Returns:
        The fenced block.

    Raises:
        KeyError: if no use case declares a requested context edge.

    """
    graphs = use_case_graphs(matter_sites)
    foreign: list[tuple[Artifact, Edge]] = []
    for pair in context:
        found = next(
            (
                (graph, edge)
                for key, graph in graphs.items()
                if key != name
                for edge in graph.edges
                if (edge.source, edge.target) == pair
            ),
            None,
        )
        if found is None:
            msg = f"no use case other than {name!r} declares the edge {pair}"
            raise KeyError(msg)
        owner, edge = found
        foreign.append((owner.node(edge.source), edge))
    return _figure([graphs[name]], context=foreign)


def figures(matter_sites: int = 3) -> dict[Path, str]:
    """Every generated figure, keyed by the page that carries it.

    Args:
        matter_sites: the chain length the graphs are built at; the figures do not depend
            on it.

    Returns:
        One fenced block per page, keyed by path relative to the repository root.

    """
    block = overview(matter_sites)
    drawn = dict.fromkeys(FIGURE_PAGES, block)
    for page, name, context in GUIDE_FIGURES:
        drawn[page] = use_case_figure(name, context, matter_sites)
    return drawn


_BLOCK = re.compile(r"^```mermaid\n.*?^```", re.M | re.S)


def _replaced(text: str, block: str) -> str:
    """The page text with its single mermaid block replaced.

    Raises:
        ValueError: if the page does not carry exactly one mermaid block.

    """
    found = _BLOCK.findall(text)
    if len(found) != 1:
        msg = f"expected exactly one mermaid block, found {len(found)}"
        raise ValueError(msg)
    return _BLOCK.sub(lambda _: block, text, count=1)


def main(root: Path | None = None, *, verbose: bool = True) -> dict[Path, str]:
    """Write every generated figure into the page that carries it.

    Args:
        root: the repository root; defaults to the parent of this file's directory.
        verbose: print which pages changed.

    Returns:
        The figures, keyed by page.

    """
    base = root or Path(__file__).resolve().parent.parent
    drawn = figures()
    for page, block in drawn.items():
        path = base / page
        text = path.read_text(encoding="utf-8")
        updated = _replaced(text, block)
        changed = updated != text
        if changed:
            path.write_text(updated, encoding="utf-8")
        if verbose:
            print(f"{page}: {'rewritten' if changed else 'unchanged'}")
    return drawn


if __name__ == "__main__":
    main()
