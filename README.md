# Q-SiMod

A typed metamodel for the model-driven engineering (MDE) of quantum simulation.

**Documentation: <https://lfd.github.io/qsimod/>** guides to the three use cases, the layers
of the package and the API reference.

Quantum simulation lets a controllable quantum system stand in for a physical quantum system
whose dynamics are classically intractable.  In practice it is a chain of modelling steps: the
Hamiltonian of a physical theory is subjected to truncations, changes of basis, encodings and
perturbative mappings until it arrives at a hardware model, which is either analogue, meaning
that a device realises the Hamiltonian natively, or digital, meaning a product formula to be
executed on a gate-based quantum computer.  Q-SiMod frames each of these steps as a
**transformation** between typed **artifacts**.  Every artifact carries a symbolic operator, a
structural type and a parameter set; every transformation declares the structural types it
requires and produces, whether it is **exact** or **approximate**, and its parameter relation.
A pipeline of transformations is **type-checked** before anything is computed, the **knob
settings** of a hardware model are found by a **solver** over the whole pipeline, with the
margin of every regime condition reported, and a **numerical realisation** layer validates the
declarations by exact diagonalisation (JAX) at small chain length.

The package implements the metamodel of the article it accompanies; see
the [citation](#citation).  The [documentation site](https://lfd.github.io/qsimod/) walks
through each use case artifact by artifact and transformation by transformation.

## Digital and analogue simulation

<img src="docs/figures/analogue_vs_digital.svg" alt="Digital and analogue quantum simulation side by side: the evolution under H_sys Trotterised into n slices of gates on a universal gate set, and the same evolution mapped onto H_sim and compiled into control pulses." width="100%">

Both modes start from the same object: the dynamics of a physical system, governed by a
Hamiltonian $`\hat H_\text{sys}`$ and carried out by the unitary $`\hat U_\text{sys}`$ from the initial state at time $`0`$
to the target state at time $`t`$.  **Digital** simulation discretises that evolution into `n`
time slices, a *Trotterisation*, whose one- and two-qubit gates run on universal gate-based
hardware and realise an implicit, approximate Hamiltonian $`\hat H_\approx`$.  **Analogue** simulation maps
$`\hat H_\text{sys}`$ onto a Hamiltonian $`\hat H_\text{sim}`$ that a device realises natively and compiles it into
time-dependent control fields.  Both chains terminate in an instruction set: a universal gate
set, portable across gate-based machines, or an analogue instruction set tailored to one
platform.  Q-SiMod covers the chain down to the hardware model; the instruction sets
themselves are [out of scope](#out-of-scope).

## The metamodel

<img src="docs/figures/overview.svg" alt="The metamodel: three abstraction layers from an application model over intermediate representations to a hardware model that splits into an analogue and a digital simulator model, with the attributes of an artifact and of a transformation and a numerical realisation lane." width="100%">

The chain of modelling steps is organised in three abstraction layers: an application model,
the intermediate representations it passes through, and a hardware model, which simulates
either in analogue mode, $`\hat H_\text{sim}`$, or in digital mode, $`\hat U_\approx`$.  The nodes of that chain are the
artifacts and its edges the transformations, with the attributes each of them declares shown
beside them.  Because artifacts and transformations are declarative, nothing is computed to
type-check a pipeline; the **numerical realisation** layer on the right is separate and
optional, and it is what realises the operators, validates the transformations, solves the
parameter relations and measures the errors.

## The model graph

The artifacts and transformations the package implements, arranged by abstraction layer:

```mermaid
flowchart TB
    subgraph LV1["Application model &nbsp;·&nbsp; <code>models.application</code>"]
        direction LR
        lattice_qed["<b>lattice_qed &nbsp; H<sub>sys</sub></b><br/>lattice QED<br/>(Kogut–Susskind)<br/><i>Θ<sub>sys</sub> = {m, a, e}</i>"]
        xxz_magnet["<b>xxz_magnet &nbsp; H<sub>XXZ</sub></b><br/>Heisenberg XXZ magnet,<br/>by its anisotropy<br/><i>Θ = {J<sub>xy</sub>, Δ}</i>"]
        ising_magnet["<b>ising_magnet &nbsp; H<sub>Ising</sub></b><br/>antiferromagnetic<br/>Ising chain<br/><i>Θ = {J<sub>z</sub>, h<sub>z</sub>, h<sub>x</sub>}</i>"]
    end
    subgraph LV2["Intermediate representations &nbsp;·&nbsp; <code>models.intermediate</code>"]
        quantum_link_staggered["<b>quantum_link_staggered &nbsp; H<sub>IR1</sub></b><br/>quantum-link model,<br/>staggered mass<br/><i>Θ<sub>IR1</sub> = {m, κ}</i>"]
        quantum_link_homogeneous["<b>quantum_link_homogeneous &nbsp; H<sub>IR2</sub></b><br/>quantum-link model,<br/>pair coupling<br/><i>Θ<sub>IR2</sub> = {m, κ}</i>"]
        effective_bosonic["<b>effective_bosonic &nbsp; H<sub>IR3</sub></b><br/>boson encoding<br/><i>Θ<sub>IR3</sub> = {m, κ}</i>"]
        qubit_register["<b>qubit_register &nbsp; H<sub>IR4</sub></b><br/>qubit Hamiltonian<br/><i>Θ<sub>IR4</sub> = {m, κ}</i>"]
        xxz_chain["<b>xxz_chain &nbsp; H<sub>XXZ</sub></b><br/>XXZ chain,<br/>by its two couplings<br/><i>Θ = {J<sub>xy</sub>, J<sub>z</sub>}</i>"]
        fermion_chain["<b>fermion_chain &nbsp; H<sub>tV</sub></b><br/>spinless fermions,<br/>nearest-neighbour interaction<br/><i>Θ = {J<sub>xy</sub>, J<sub>z</sub>}</i>"]
        ising_chain["<b>ising_chain &nbsp; H<sub>Ising</sub></b><br/>Ising chain,<br/>by three energies<br/><i>Θ = {J<sub>z</sub>, Γ, B}</i>"]
    end
    subgraph LV3["Hardware model &nbsp;·&nbsp; <code>models.hardware</code>"]
        direction LR
        bose_hubbard["<b>bose_hubbard &nbsp; H<sub>sim</sub></b><br/>tilted, staggered<br/>Bose–Hubbard chain<br/><i>Θ<sub>sim</sub> = {J, U, δ, Δ}</i>"]
        trotter["<b>trotter &nbsp; U<sub>≈</sub></b><br/>Trotter product formula<br/><i>t, n, order</i>"]
        two_component_bose_hubbard["<b>two_component_bose_hubbard &nbsp; H<sub>2BHM</sub></b><br/>two-component<br/>Bose–Hubbard chain<br/><i>Θ = {t, U<sub>↑↑</sub>, U<sub>↑↓</sub>, U<sub>↓↓</sub>}</i>"]
    end
    subgraph OUT["Numerical realisation and solving"]
        direction LR
        SOLVE["<b>parameter realisation</b><br/>a constrained solve for the knob settings<br/>exact · approximate · infeasible · unsolved"]
        VALID["<b>validity report</b><br/>every regime condition,<br/>with its margin"]
        NUM["<b>numerical realisation</b><br/>dense operators in JAX,<br/>spectra and dynamics"]
        SOLVE ~~~ VALID ~~~ NUM
    end
    L4(["Executable layer: gates, routing, pulses<br/><i>out of scope</i>"])

    lattice_qed -. "<b>quantum-link truncation</b><br/><i>approximate, regime conditions</i>" .-> quantum_link_staggered
    quantum_link_staggered ---> |"<b>particle–hole transformation</b><br/><i>exact</i>"| quantum_link_homogeneous
    quantum_link_homogeneous ---> |"<b>boson encoding</b><br/><i>exact on the encoded subspace</i>"| effective_bosonic
    effective_bosonic -. "<b>degenerate perturbation theory,<br/>solved for the knob settings</b><br/><i>approximate, regime conditions</i>" .-> bose_hubbard
    quantum_link_homogeneous ---> |"<b>Jordan–Wigner transformation</b><br/><i>exact</i>"| qubit_register
    qubit_register -. "<b>Trotterisation</b><br/><i>approximate, resource-controlled</i>" .-> trotter

    xxz_magnet ---> |"<b>anisotropy resolution</b><br/><i>exact</i>"| xxz_chain
    xxz_chain ---> |"<b>Jordan–Wigner to spinless fermions</b><br/><i>exact</i>"| fermion_chain
    xxz_chain -. "<b>second-order superexchange,<br/>solved for the knob settings</b><br/><i>approximate, regime conditions</i>" .-> two_component_bose_hubbard

    ising_magnet ---> |"<b>field resolution</b><br/><i>exact</i>"| ising_chain
    ising_chain -. "<b>resonant dipole reduction,<br/>solved for the knob settings</b><br/><i>approximate, regime conditions</i>" .-> bose_hubbard

    LV3 ==> |"a hardware model with bound parameters"| OUT
    LV3 -.-> L4

    classDef ham fill:#e0f2ee,stroke:#009371,color:#003e2f
    classDef pf fill:#fbf1d9,stroke:#c48700,color:#573c00
    classDef svc fill:#ede8f3,stroke:#988baa,color:#4c4655
    classDef oos fill:#fdebea,stroke:#ed665a,color:#642b26
    class lattice_qed,xxz_magnet,ising_magnet,quantum_link_staggered,quantum_link_homogeneous,effective_bosonic,qubit_register,xxz_chain,fermion_chain,ising_chain,bose_hubbard,two_component_bose_hubbard ham
    class trotter pf
    class SOLVE,VALID,NUM svc
    class L4 oos
    style LV1 fill:#f2f7fa,stroke:#b1d0e5,color:#103c5a
    style LV2 fill:#f2f7fa,stroke:#b1d0e5,color:#103c5a
    style LV3 fill:#f2f7fa,stroke:#b1d0e5,color:#103c5a
    style OUT fill:#f6f4f9,stroke:#d8cee5,color:#5f576a
```

<!-- Generated by scripts/figures.py from the graphs themselves; do not edit by hand. -->

A **solid** arrow is an exact transformation and a **dotted** arrow an approximate one, with
the kind of approximation in italics.  Every box is headed by the name under which the model
graph holds the artifact, followed by the notation of the article.  Three physical theories
descend from the application layer through intermediate representations to the hardware layer,
and two of them, the gauge theory via `effective_bosonic` and the Ising chain via
`ising_chain`, arrive at the same hardware model `bose_hubbard`.

The chain from `lattice_qed` forms the case study of the article, a **one-dimensional lattice
quantum electrodynamics** carried to an analogue simulator model $`\hat H_\text{sim}`$
and a digital simulator model $`\hat U_\approx`$
([Zhou et al., Science **377**, 311 (2022)](https://doi.org/10.1126/science.abl6277)); the
chain from `xxz_magnet` is a **Heisenberg magnet**
([Jepsen et al., Nature **588**, 403 (2020)](https://doi.org/10.1038/s41586-020-3033-y)), and
the chain from `ising_magnet` an **Ising chain**
([Simon et al., Nature **472**, 307 (2011)](https://doi.org/10.1038/nature09994)).

---

## Install

The project uses [uv](https://docs.astral.sh/uv/) for the environment, and
[ruff](https://docs.astral.sh/ruff/), [ty](https://docs.astral.sh/ty/) and
[zensical](https://zensical.org/) for linting and formatting, type checking and the
documentation site.

```sh
uv sync --all-extras     # create the environment, including docs and dev tools
```

The runtime dependencies are `jax`, `jaxlib`, `numpy`, `pandas` and `scipy`.  `scipy` provides
the solver backend and is imported only by `qsimod/solving/backends/scipy_nlp.py`.

## Quickstart

The application model is initialised with its parameters, the pipeline is
type-checked, and the knob settings of the hardware model that realise the
request are found by the solver:

```python
from qsimod.usecases.schwinger import ParameterNames as P
from qsimod.usecases.schwinger import build_graph, device_limits
from qsimod.solving import realise_parameters

graph = build_graph(matter_sites=3)
print(graph.analogue)  # the pipeline, and what kind of approximation it is

targets = {
    P.MASS_LATTICE_QED: 0.02,
    P.COUPLING_QUANTUM_LINK_STAGGERED: 0.004525,
    P.ELECTRIC_GAP: 0.5,
}
print(graph.analogue.classify_relation(frozenset(targets)))
# UNDER_DETERMINED: 7 equation(s), 9 unknown(s) ...; 2 residual degree(s) of freedom

result = realise_parameters(
    graph.analogue,
    targets=targets,
    unknowns=[P.TUNNELLING, P.INTERACTION, P.SUPERLATTICE, P.TILT],
    admissible_set=device_limits(),
)
print(result)  # a status, a point, residuals, and every margin
```

The result carries a status: `EXACT_SOLUTION`, `APPROXIMATE_SOLUTION` with a
per-parameter residual, `UNSOLVED`, or `INFEASIBLE` with the binding constraint
named, which is proved by interval and affine arithmetic over the declared
boxes before any search runs.

The digital branch, with the step count of the product formula found as an integer solve
against the a-priori error bound:

```python
from qsimod.solving.stepcount import minimal_steps
from qsimod.transformations import interleaved_layers_of
from qsimod.usecases.schwinger import trotterisation

qubits = ...  # a bound qubit_register model; see the examples
layers = interleaved_layers_of(qubits)
print(layers.commutation_report())  # which layer pairs commute, derived from the Pauli algebra
n = int(minimal_steps(layers, order=2, time=2.0, target_error=1e-3).point["trotter.n"])
print(trotterisation(time=2.0, steps=n, order=2).apply(qubits).resources())
```

## Layout

| Path | Contents |
|---|---|
| `qsimod/levels.py`, `scalar.py`, `affine.py`, `units.py`, `structure.py`, `symbolic.py`, `normal_form.py`, `parameters.py`, `artifact.py`, `relations.py`, `validity.py`, `transform.py`, `pipeline.py` | **Declaration Layer**: symbolic artifacts, parameters, parameter relations, validity conditions, transformations and pipelines.  `normal_form.py` decides whether two symbolic sums are the same operator, which is how the claim of an `EXACT` transformation is checked.  No operator matrix is constructed and no solver is imported. |
| `qsimod/pauli.py`, `qsimod/trotter/` | **Structural Layer**: operations on the term structure without constructing operators, namely commutation, layer decompositions, product formulas, a-priori error bounds and resource counts. |
| `qsimod/solving/` | **Solving Layer**: parameter realisation, feasibility and step counts. |
| `qsimod/realise/`, `qsimod/jax_setup.py` | **Realisation Layer**: dense `complex128` operators in JAX. |
| `qsimod/models/` | **Model Library**, indexed by abstraction layer: `application`, `intermediate` and `hardware`, with `gauge` and `magnetism` for what each family of models shares. |
| `qsimod/transformations/` | **Transformation Library**, indexed by the kind of operation: truncations, basis changes, encodings, reparametrisations, perturbative reductions and Trotterisation. |
| `qsimod/usecases/` | **Use Cases**: which models, at which conventions, are related by which transformations and assembled into which model graph. |
| `examples/` | Five runnable end-to-end examples, with the reference data of the three experiments in `zhou2022.py`, `jepsen2020.py` and `simon2011.py`. |
| `scripts/` | The two numerical studies of the article, which write `pandas` frames to `results/`, and the generator of the documentation figures. |
| `plots/` | The R scripts that draw the figures of the article from `results/`. |
| `docs/` | The sources of the [documentation site](https://lfd.github.io/qsimod/): one guide per application model ([`schwinger.md`](docs/schwinger.md), [`heisenberg.md`](docs/heisenberg.md), [`ising.md`](docs/ising.md)) and an API reference generated from the docstrings. |

## Examples

```sh
uv run python examples/analogue_end_to_end.py    # the analogue branch, posed against Zhou et al. 2022
uv run python examples/infeasible_request.py     # an unreachable request, with the binding constraint named
uv run python examples/digital_end_to_end.py     # the digital branch, with the step count solved for
uv run python examples/heisenberg_end_to_end.py  # the Heisenberg magnet, posed against Jepsen et al. 2020
uv run python examples/shared_device.py          # two theories on one lattice, posed against Simon et al. 2011
```

## Numerical validation of the case study

The tables and figures of the article are produced by two scripts, at `N = 6` matter sites,
`m = 0`, `kappa = 14.5 Hz` and the experimental time window of 150 ms of Zhou et al. (2022):

```sh
uv run python scripts/zhou_trajectories.py --matter-sites 6 --realisation number-sector
uv run python scripts/zhou_digital.py --matter-sites 6
```

The first solves for the knob settings of the analogue simulator model and records the
trajectories of the mean matter occupation, the deviation from the effective theory, the
leakage and the gauge violation for the free and the prescribed knob settings; the second
records the resources, the a-priori error bounds and the measured deviations of the digital
simulator model over a ladder of step counts, and the step counts required to stay below given
deviation thresholds.  Each writes CSV files to `results/` (gitignored); `plots/plot.r` draws
the figures from a precomputed copy of these files in `plots/results_paper`.  A run at
`N = 6` takes several hours.

### Drawing the figures with R

The figures of the article are drawn by the R scripts in `plots/`.  They need R with three
packages; the TikZ route additionally needs a LaTeX installation providing `lualatex` and the
`standalone`, `IEEEtran`, `tikz` and `quantikz` packages.

```sh
# R itself: apt install r-base (Debian/Ubuntu), brew install r (macOS)
Rscript -e 'install.packages(c("tidyverse", "tikzDevice", "scales"))'
```

Then, from `plots/`:

```sh
make        # plot.r writes img-tikz/*.tex, lualatex renders them to img-gen/*.pdf
make pdf    # standalone PDFs into img-pdf/ instead, no LaTeX needed
```

Both read `results_paper/zhou_trajectories.csv` and
`results_paper/zhou_digital_trajectories.csv`; to redraw from your own run, copy those two
files over from `results/`.

The documentation figures are generated from the model graphs the use cases build; they are
regenerated after adding an artifact or renaming a transformation (a test checks that they are
current):

```sh
uv run python scripts/figures.py                # rewrite every generated figure, in place
```

## Tests, linting, typing, docs

```sh
uv run pytest                     # the whole suite
uv run pytest -m "not slow"       # skip the documentation build and the wheel build
uv run ruff check                 # lint
uv run ruff format                # format, in place
uv run ty check                   # type check
uv run --extra docs zensical build --clean --strict   # the documentation site, into site/
```

`zensical` keeps a build cache in `.cache/`; a cache from an older version is reused after an
upgrade and renders cross-references as literal `[name][qsimod.thing]`.  `--clean` (used by
the test suite) or `rm -rf .cache` fixes it.

Every code snippet in the guide pages is extracted and executed by
`tests/test_docs_guide.py`.  `ruff` runs a broad rule set including pydocstyle and
flake8-annotations and is also the formatter, for the sources, for the code blocks in the
docstrings, and for the Python blocks in the Markdown pages.  `ty` runs with the rules it
leaves off by default turned on and its warnings promoted to errors, and `pytest` uses
`filterwarnings = ["error"]`.  The source carries no type-checker suppressions.

The tests are grouped by property: `tests/test_type_checking.py` (structural and kind type
checking), `test_regime_checking.py` (validity conditions and margins),
`test_semantic_preservation.py` (numerical agreement between abstraction layers),
`test_extensibility.py` (adding a hardware model from outside the package),
`test_engineering.py` (build and runnable examples), `test_solver.py` (parameter
realisation), `test_digital.py` (the digital branch), `test_core_conventions.py` (conventions
the other tests assume) and `test_docs_guide.py` (the guide snippets).

**JAX precision.**  Importing `qsimod` enables the 64-bit mode of JAX for the whole process,
since comparisons are made at tolerances of `1e-10`.  `qsimod.jax_setup` documents the switch,
`x64_enabled()` reports it, and `require_x64()` raises if it is off.

## Contributing

Bug reports, corrections and new models, transformations or use cases are welcome.
[`CONTRIBUTING.md`](CONTRIBUTING.md) describes the workflow, the checks a pull request has to
pass and the conventions of the code base; [`AGENTS.md`](AGENTS.md) explains the architecture and
the invariants the test suite enforces.

## Out of scope

The executable layer: the decomposition into a concrete gate set, qubit routing,
transpilation, scheduling and pulse schedules.  The digital branch ends at an ordered product
of exponentials of k-local Pauli terms, so the resource model counts k-local unitaries and
layers, not CNOTs.  Also absent are hardware backends and vendor SDKs, noise models, error
mitigation or correction, and a textual modelling language; the modelling language is the
Python API.  A test inspects the imports and the public API of the package to keep these
boundaries.

## Citation

The package implements the metamodel of this [paper](https://arxiv.org/abs/XXXX.XXXXX)

```bibtex
@misc{franz26_qsimod,
  author       = {Franz, Maja and Mauerer, Wolfgang},
  title        = {Gotta Model them All: Model-Driven Engineering of Quantum Simulation},
  year         = {2026},
  eprint       = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  url          = {https://arxiv.org/abs/XXXX.XXXXX},
}
```

which in turn builds on the vision presented in this paper [article](https://doi.org/10.1145/3786150.3788611)

```bibtex
@inproceedings{franz26_qse,
  author    = {Franz, Maja and Schmidbauer, Lukas and Ammermann, Joshua and
               Schaefer, Ina and Mauerer, Wolfgang},
  title     = {Towards Quantum Software for Quantum Simulation},
  year      = {2026},
  booktitle = {Proceedings of the 7th IEEE/ACM International Workshop on Quantum
               Software Engineering},
  pages     = {50--54},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  isbn      = {9798400723834},
  doi       = {10.1145/3786150.3788611},
  url       = {https://doi.org/10.1145/3786150.3788611},
}
```
