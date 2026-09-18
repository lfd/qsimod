"""Guide pages under ``docs/``: every Python snippet runs, links resolve, figures are current."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from scripts.figures import figures

from qsimod.usecases.heisenberg import build_graph as heisenberg_graph
from qsimod.usecases.ising import build_graph as ising_graph
from qsimod.usecases.schwinger import build_graph as schwinger_graph

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

#: The hand-written guide pages; the API-reference pages are generated from docstrings.
GUIDE_PAGES = ("schwinger.md", "heisenberg.md", "ising.md")

#: Fenced Python blocks, including those indented inside an admonition; the fence info is
#: captured so that a block labelled ``title="continued"`` can be run after its predecessor.
_BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)```python(?P<info>[^\n]*)\n(?P<body>.*?)^(?P=indent)```",
    re.M | re.S,
)

#: The fence-info marker that makes a snippet a continuation of the one before it.
CONTINUATION = "continued"


def _snippets(page: str) -> list[tuple[str, int, str]]:
    """Every Python snippet of a page, as ``(page, line number, runnable source)``.

    A block labelled ``title="continued"`` is returned with the preceding block's source
    prepended.
    """
    text = (DOCS / page).read_text(encoding="utf-8")
    found: list[tuple[str, int, str]] = []
    previous = ""
    for match in _BLOCK.finditer(text):
        indent = match.group("indent")
        body = match.group("body")
        if indent:
            body = "\n".join(
                line.removeprefix(indent) if line.startswith(indent) else line
                for line in body.splitlines()
            )
        source = f"{previous}\n{body}" if CONTINUATION in match.group("info") else body
        line = text[: match.start()].count("\n") + 1
        found.append((page, line, source))
        previous = source
    return found


ALL_SNIPPETS = [snippet for page in GUIDE_PAGES for snippet in _snippets(page)]


def test_the_guide_pages_exist_and_carry_snippets() -> None:
    """Every guide page exists and contributes at least one snippet."""
    for page in GUIDE_PAGES:
        assert (DOCS / page).is_file(), page
    assert len(ALL_SNIPPETS) >= 15
    pages = {page for page, _, _ in ALL_SNIPPETS}
    assert pages == set(GUIDE_PAGES)


@pytest.mark.parametrize(
    ("page", "line", "source"),
    ALL_SNIPPETS,
    ids=[f"{page}:{line}" for page, line, _ in ALL_SNIPPETS],
)
def test_a_guide_snippet_runs(
    page: str,
    line: int,
    source: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each snippet executes in a fresh namespace without raising."""
    namespace: dict[str, Any] = {"__name__": "__doc_snippet__"}
    try:
        exec(compile(source, f"{page}:{line}", "exec"), namespace)  # noqa: S102
    except Exception as error:
        captured = capsys.readouterr()
        pytest.fail(
            f"{page} line {line} raised {type(error).__name__}: {error}\n"
            f"--- snippet ---\n{source}\n--- output so far ---\n{captured.out}"
        )


def test_every_guide_link_target_exists() -> None:
    """Relative links between the guide pages, and their anchors, resolve."""
    anchors: dict[str, set[str]] = {}
    for page in GUIDE_PAGES:
        text = (DOCS / page).read_text(encoding="utf-8")
        headings = re.findall(r"^#{2,4} (.+)$", text, re.M)
        anchors[page] = {_slug(heading) for heading in headings}

    problems: list[str] = []
    for page in GUIDE_PAGES:
        text = (DOCS / page).read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?!https?:)([^)]+)\)", text):
            document, _, anchor = target.partition("#")
            document = document or page
            if not (DOCS / document).is_file():
                problems.append(f"{page} links to missing page {document}")
                continue
            if anchor and anchor not in anchors.get(document, set()):
                problems.append(f"{page} links to missing anchor {document}#{anchor}")
    assert not problems, problems


def _slug(heading: str) -> str:
    """The anchor the site generator slugifies a Markdown heading into."""
    text = re.sub(r"`|\*|\[|\]|\(|\)|:|,|\.|/|\+|'", "", heading.lower())
    return re.sub(r"[^a-z0-9_]+", "-", text).strip("-")


def test_every_figure_is_the_source_its_generator_produces() -> None:
    """Each page carries the mermaid source ``scripts/figures.py`` generates, verbatim."""
    for page, block in figures().items():
        text = (REPO / page).read_text(encoding="utf-8")
        assert block in text, (
            f"the figure in {page} is not what scripts/figures.py produces; run it to regenerate"
        )


def test_every_figure_draws_the_use_case_its_page_is_about() -> None:
    """The overview draws all three use cases and each guide draws only its own."""
    drawn = figures()
    sources = {
        "schwinger": schwinger_graph(matter_sites=3).source,
        "heisenberg": heisenberg_graph(sites=3).source,
        "ising": ising_graph(matter_sites=3).source,
    }
    overview = drawn[Path("README.md")]
    for source in sources.values():
        assert f'\n        {source}["' in overview, source
    for use_case, source in sources.items():
        page = drawn[Path(f"docs/{use_case}.md")]
        assert f'\n        {source}["' in page, use_case
        assert not [
            other
            for name, other in sources.items()
            if name != use_case and f'\n        {other}["' in page
        ], use_case

    # The shared hardware node is reached from the gauge theory and from the Ising chain in the
    # overview and in the Ising guide.
    for block in (overview, drawn[Path("docs/ising.md")]):
        arrivals = [line for line in block.splitlines() if line.rstrip().endswith(" bose_hubbard")]
        assert {line.split()[0] for line in arrivals} == {"effective_bosonic", "ising_chain"}
