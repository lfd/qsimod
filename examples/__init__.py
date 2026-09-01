"""Runnable end-to-end examples of the framework.

Each module exposes a ``main`` that the test suite imports, and runs directly with
``uv run python examples/<name>.py``.  [`Report`][examples.Report] prints the report of a run.
"""

from __future__ import annotations

__all__ = ["RULE_WIDTH", "Report"]

#: The width of the horizontal rules around a section heading.
RULE_WIDTH = 78


class Report:
    """The printed report of an example; ``verbose=False`` suppresses all output.

    Attributes:
        verbose: whether anything is printed.

    """

    def __init__(self, *, verbose: bool = True) -> None:
        self.verbose = verbose

    def say(self, *parts: object) -> None:
        """Print one line unless the report is silenced."""
        if self.verbose:
            print(*parts)

    def section(self, title: str) -> None:
        """Print a section heading between two horizontal rules."""
        rule = "=" * RULE_WIDTH
        self.say(rule)
        self.say(title)
        self.say(rule)
