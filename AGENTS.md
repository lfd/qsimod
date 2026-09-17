# Working on Q-SiMod

Guidance for coding agents. `README.md` explains *what* the project is and the physics it
models; this file covers *how to change it without breaking it*. Read the README's "Layout"
and "Out of scope" sections before touching anything under `qsimod/`.

## In one paragraph

Q-SiMod is a typed metamodel for the model-driven engineering of quantum simulation.
A **transformation** maps one typed **artifact** (a symbolic operator plus a structural type
plus a parameter set) to another; a chain of them is a **pipeline**, which is type-checked
symbolically before anything numerical happens. A solver then finds the hardware knob settings
that realise a request, and a separate realisation layer validates the declarations by exact
diagonalisation in JAX at small chain length. Nearly everything is declarative and symbolic;
the numerics are a validation lane, not the main road.

## Environment and commands

The project uses [uv](https://docs.astral.sh/uv/). There is a committed `uv.lock`; use it.

```sh
uv sync --all-extras                                   # environment, including dev and docs
uv run pytest                                          # the whole suite
uv run pytest -m "not slow"                            # skip the docs build and the wheel build
uv run ruff check                                      # lint
uv run ruff format                                     # format, in place
uv run ty check                                        # type check
uv run --extra docs zensical build --clean --strict    # the documentation site, into site/
```

Run all four checks before declaring a change finished. The dev dependency group carries
`pytest`, `ruff` and `ty` and uv installs it by default, so those need no `--extra`; the
documentation site needs `--extra docs`.

Python is `>=3.11`. CI tests 3.11, 3.12 and 3.13.

## The layers, and the direction of dependency

The package is stratified, and the stratification is the point of the project. Respect it.

| Layer | Modules | May import |
|---|---|---|
| Declaration | `levels`, `scalar`, `affine`, `units`, `structure`, `symbolic`, `normal_form`, `parameters`, `artifact`, `relations`, `validity`, `transform`, `pipeline` | each other, and nothing else: no `numpy`, no `jax`, no solver |
| Structural | `pauli`, `trotter/` | the declaration layer; `trotter/schedule.py` is the one module here that reaches for `jax` |
| Solving | `solving/` | the layers above it; `numpy`, and `scipy` only inside `solving/backends/` |
| Realisation | `realise/`, `jax_setup` | the layers above it; `jax` |
| Libraries | `models/`, `transformations/` | the declaration and structural layers |
| Use cases | `usecases/` | everything; this is where models, conventions and transformations are assembled |

The declaration layer is pure Python today — it imports neither `numpy` nor `jax` nor any
solver — and a change that introduces such an import is almost certainly a design error rather
than a shortcut. `solving/backends/scipy_nlp.py` is the only `scipy` consumer. `jax` appears
only in `realise/`, `jax_setup.py`, `trotter/schedule.py` and that one backend.

## Invariants the test suite enforces

These fail CI, and several of them are non-obvious:

- **No vendor SDK.** No module under `qsimod/` may import `qiskit`, `cirq`, `pulser`,
  `braket`, `pennylane`, `pytket`, `qutip` or `projectq` (`tests/test_digital.py`).
- **No level-4 vocabulary in public names.** A public name containing `cnot`, `gate_count`,
  `gate_set`, `transpile`, `routing`, `clifford`, `qasm`, `pulse_schedule` and friends fails
  the scope test. The digital branch stops at an ordered product of exponentials of k-local
  Pauli terms; the resource model counts k-local factors and layers, never gates.
- **The mermaid figures are generated.** The diagrams in `README.md`, `docs/index.md` and the
  three guide pages are emitted by `scripts/figures.py` from the model graphs themselves, and
  a test asserts the file contains that output verbatim. Never hand-edit a generated mermaid
  block; run `uv run python scripts/figures.py` after adding an artifact or renaming a
  transformation.
- **Every Python snippet in the guide pages runs.** `tests/test_docs_guide.py` extracts the
  fenced `python` blocks from `docs/schwinger.md`, `heisenberg.md` and `ising.md` and executes
  them. A block whose fence info contains `title="continued"` is run with the preceding
  block's source prepended. Intra-page links are checked against real heading anchors too.
- **`pyproject.toml` is self-consistent.** `project.version` must equal `qsimod.__version__`,
  and every path named under `[tool.hatch]`, `[tool.ty.src]` and `[tool.ruff]` must exist.
- **Warnings are errors.** `filterwarnings = ["error"]` in the pytest config, and
  `[tool.ty.terminal] error-on-warning = true`.

## Style

- `ruff` is both linter and formatter, with a broad rule set: pydocstyle (`D`),
  flake8-annotations (`ANN`), flake8-bandit (`S`), pylint (`PL`) and more. Line length 100.
  It formats the code blocks inside docstrings as well as the sources.
- Google-style docstrings on every public module, class and function — `mkdocstrings` renders
  them into the API reference, so a docstring is user-facing documentation.
- Cross-reference with ``[`Name`][qsimod.module.Name]``, never with a Sphinx role such as
  ``:class:`Name` ``; a test rejects Sphinx role syntax because mkdocstrings renders it as
  literal text.
- Full type annotations. **The source carries no type-checker suppressions** — if `ty`
  complains, fix the types rather than silencing it. `blanket-ignore-comment` is an error, so
  any suppression must at minimum name its rule.
- British spelling throughout: *realisation*, *Trotterisation*, *normalise*, *truncation*.
  Match it in new code and prose. (The one `realized` in the tree is inside a quoted paper
  title and is correct.)
- Test names are full sentences describing the property:
  `test_the_jordan_wigner_string_is_empty_between_adjacent_matter_sites`. Tests are grouped by
  property, not by module — see the README's list of which file holds which.

## Gotchas

- **JAX 64-bit mode is process-global.** Importing `qsimod` enables it, because comparisons
  run at tolerances around `1e-10`. `qsimod.jax_setup.x64_enabled()` reports the state and
  `require_x64()` raises if it is off. Do not disable it in a test.
- **Stale zensical cache.** `zensical` caches into `.cache/`, and a cache written by an older
  version is reused after an upgrade and renders cross-references as literal
  `[name][qsimod.thing]`. `--clean` or `rm -rf .cache` fixes it. Always build with `--clean`.
- **The slow tests are slow for a reason.** `-m slow` covers the documentation build and the
  wheel build; the docs test skips itself unless the docs extra is installed, so run it as
  `uv run --extra docs pytest -m slow` or it will pass without checking anything.
- **Do not run the numerical studies casually.** `scripts/zhou_trajectories.py` and
  `scripts/zhou_digital.py` at the article's `N = 6` take several hours. Use a small
  `--matter-sites` when exercising them.
- **Gitignored outputs.** `results/`, `plots/img-*/`, `site/`, `.cache/` and `docs/references/`
  are generated or external; never commit them, and do not assume `docs/references/` exists on
  a fresh clone.
- **The `plots/` scripts are R**, drawing the article's figures from `results/`. They are not
  part of the Python checks.

## Adding things

- **A new model** goes in `qsimod/models/`, filed by abstraction level (`application`,
  `intermediate`, `hardware`), with what a family shares in `gauge.py` or `magnetism.py`.
- **A new transformation** goes in `qsimod/transformations/`, filed by the kind of operation
  (truncations, basis changes, encodings, reparametrisations, perturbative reductions,
  Trotterisation). Declare the structural types it requires and produces, whether it is exact
  or approximate, and its parameter relation. An `EXACT` claim is checked by
  `normal_form.py`, which decides whether two symbolic sums are the same operator — so the
  claim must actually hold.
- **A new use case** goes in `qsimod/usecases/`, and then: regenerate the figures, add a guide
  page under `docs/`, and add it to `nav` in `zensical.toml`. `tests/test_extensibility.py`
  shows a hardware model being added from outside the package; it is the best worked example
  of the extension points.
- **Anything at the executable layer** — gate sets, routing, transpilation, pulses, noise
  models, vendor backends — is deliberately out of scope. Do not add it, even helpfully.

## CI

Three workflows in `.github/workflows/`, all triggered on pull requests to `main`, on pushes
to `main`, and manually:

| Workflow | Does |
|---|---|
| `quality.yml` | `ruff check`, `ruff format --check`, `ty check` |
| `tests.yml` | `pytest -m "not slow"` on Python 3.11/3.12/3.13, plus the slow tests once |
| `docs.yml` | builds the site; uploads it as an artifact on pull requests, deploys it to GitHub Pages on push to `main` |
