# Contributing to Q-SiMod

Contributions are welcome.  [`AGENTS.md`](AGENTS.md) explains the architecture, the invariants
the test suite enforces, and where a new model, transformation or use case goes.

## Workflow

1. Open an issue using one of the templates (*Bug report*, *Feature request*, *Code health*) so
   we can agree on the change first.  Small fixes such as a typo need no issue.
2. Fork the repository and work on a branch.
3. Open a pull request against `main` and link the issue.  CI runs the checks below; a pull
   request has to pass them before it is merged.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) with a committed `uv.lock`:

```sh
uv sync --all-extras     # the environment, including the dev and docs tools
```

## Checks

Run these before opening a pull request; they are what CI runs.

```sh
uv run ruff format                                   # format, in place
uv run ruff check                                    # lint
uv run ty check                                      # type check
uv run pytest -m "not slow"                          # tests; omit the marker for all
uv run --extra docs zensical build --clean --strict  # the documentation site
```

Warnings are errors in pytest and in ty, and the source carries no type-checker suppressions.

The same ruff and ty checks can run on every commit through
[pre-commit](https://pre-commit.com/), which is not a project dependency:

```sh
uvx pre-commit install
```

## Conventions

- Keep the layering: the declaration layer imports neither `numpy` nor `jax`, and nothing under
  `qsimod/` imports a vendor SDK.  The executable layer (gates, routing, pulses) is out of scope.
- Google-style docstrings on every public name, full type annotations, British spelling.
- The mermaid figures are generated: run `uv run python scripts/figures.py` after adding an
  artifact or renaming a transformation, never edit them by hand.
- Every Python block in the guide pages under `docs/` is executed by the tests.
- A new use case needs a guide page under `docs/` and an entry in `nav` in `zensical.toml`.

## Documentation

```sh
uv run --extra docs zensical serve   # local preview with automatic rebuilds
```

Pull requests build the site with `--strict`; a push to `main` deploys it to GitHub Pages.

Contributions are licensed under the project's [MIT licence](LICENSE).
