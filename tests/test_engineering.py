"""Engineering checks: project metadata, examples, the docs build and the wheel.

Slow checks are marked ``slow``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from examples.analogue_end_to_end import main as run_analogue_example
from examples.digital_end_to_end import main as run_digital_example
from examples.heisenberg_end_to_end import main as run_heisenberg_example
from examples.infeasible_request import main as run_infeasible_example
from examples.shared_device import main as run_shared_device_example

import qsimod
from qsimod.solving import SolveStatus
from qsimod.units import to_hertz
from qsimod.usecases.schwinger import ParameterNames as P

ROOT = Path(__file__).resolve().parent.parent


def test_the_project_metadata_is_valid_toml_and_self_consistent() -> None:
    """``pyproject.toml`` parses, and everything it names exists."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    project = config["project"]
    assert project["name"] == "qsimod"
    assert project["version"] == qsimod.__version__
    assert (ROOT / project["readme"]).is_file()

    for package in config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]:
        assert (ROOT / package).is_dir(), package
    for included in config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]:
        assert (ROOT / included).exists(), included

    # Solver backends are runtime dependencies.
    dependencies = " ".join(project["dependencies"])
    assert "scipy" in dependencies
    assert "jax" in dependencies


def test_the_declared_source_paths_all_exist() -> None:
    """The tool configuration's paths exist, so ``ty`` and ``ruff`` have targets."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    for path in config["tool"]["ty"]["src"]["include"]:
        assert (ROOT / path).is_dir(), path
    for path in config["tool"]["ruff"]["src"]:
        assert (ROOT / path).is_dir(), path


def test_the_analogue_example_agrees_with_the_experiment_it_is_posed_against() -> None:
    """``examples/analogue_end_to_end.py`` realises the request and ranks the settings."""
    comparison = run_analogue_example(matter_sites=3, verbose=False)
    solved, experiment = comparison.solved, comparison.experiment

    # Both settings realise the request.
    for outcome in (solved, experiment):
        assert to_hertz(outcome.coupling) == pytest.approx(14.5, rel=1e-6), outcome.label
        assert to_hertz(outcome.mass) == pytest.approx(0.0, abs=1e-9), outcome.label

    # The experiment's knobs: delta/J = 16 (Fig. 2A, case 1) and a 57 Hz tilt.
    assert experiment.staggering == pytest.approx(16.0, rel=1e-6)
    assert to_hertz(experiment.knobs[P.TILT]) == pytest.approx(57.0, rel=1e-9)

    # Errors are nonzero and bounded.
    for outcome in (solved, experiment):
        for measured in (outcome.deviation, outcome.leakage, outcome.violation):
            assert 0.0 < measured < 1.0, outcome.label

    # The wider declared margin is the more faithful setting on all three measures.
    assert solved.margin > experiment.margin
    assert solved.deviation < experiment.deviation
    assert solved.leakage < experiment.leakage
    assert solved.violation < experiment.violation


def test_the_analogue_example_reproduces_two_measured_results_of_that_experiment() -> None:
    """``examples/analogue_end_to_end.py`` reproduces two measured results of `paper_zhou22`."""
    comparison = run_analogue_example(
        matter_sites=3, interaction_ratios=(2.0, 13.0, 32.0), verbose=False
    )

    # Fig. S3a: f_exp = 21 Hz; Fig. 3: 1/gamma = 63 +- 9 ms; both on the converged row.
    assert comparison.converged.frequency_hz == pytest.approx(21.0, rel=0.05)
    assert comparison.converged.damping_ms == pytest.approx(63.0, rel=0.3)

    # Fig. 2D: the violation falls monotonically with U/J above the curve's peak.
    scan = comparison.scan
    assert [point.interaction_ratio for point in scan] == pytest.approx([2.0, 13.0, 32.0])
    violations = [point.violation for point in scan]
    margins = [point.margin for point in scan]
    assert violations == sorted(violations, reverse=True)
    assert margins == sorted(margins)
    assert violations[0] > 0.2, "U/J = 2 is far outside the gauge regime"
    assert violations[-1] < 0.02, "U/J = 32 is the gauge-theory regime"

    # Fig. 2A's cases, in the paper's order: case 1 the most constrained.
    cases = comparison.labelled_cases
    assert [case.label for case in cases] == ["case 1", "case 2", "case 3"]
    assert cases[0].violation < cases[-1].violation

    # The window is fixed in milliseconds (Fig. 2C), not on a common t*kappa axis.
    dimensionless = [point.dimensionless_time for point in scan]
    assert all(value > 0.0 for value in dimensionless)
    assert len(set(dimensionless)) == len(dimensionless)

    # At U/J = 2 the second-order coupling exceeds the tunnelling it is derived from.
    assert scan[0].coupling_per_tunnelling > 1.0
    assert scan[-1].coupling_per_tunnelling < 1.0


def test_the_heisenberg_example_agrees_with_the_experiment_it_is_posed_against() -> None:
    """``examples/heisenberg_end_to_end.py`` reproduces `paper_jepsen20`'s stated numbers."""
    comparison = run_heisenberg_example(sites=3, anisotropies=(0.973, 6.0, 60.0), verbose=False)
    solved, experiment = comparison.solved, comparison.experiment

    # Both settings realise the request.
    for outcome in (solved, experiment):
        assert outcome.anisotropy == pytest.approx(0.973, rel=1e-6), outcome.label
        assert outcome.exchange_time_ms == pytest.approx(2.01, rel=1e-6), outcome.label

    # Section 6: the Methods-table anisotropies, quoted to +-0.1.
    assert comparison.worst_table_gap < 3e-3

    # Section 7: the free-fermion band and step (b)'s exactness.
    band = comparison.band
    assert band.band_gap < 1e-12
    assert band.exactness_gap < 1e-10
    assert band.bandwidth == pytest.approx(band.expected_bandwidth, rel=1e-9)
    assert band.fermi_velocity == pytest.approx(band.transverse, rel=0.02)

    # The wider declared margin is the more faithful setting.
    assert solved.margin > experiment.margin
    assert solved.deviation < experiment.deviation

    # The solve balances the two t << U conditions, where the derived field is zero.
    assert solved.field == pytest.approx(0.0, abs=1e-9)
    assert abs(experiment.field) > 0.3 * experiment.transverse
    assert experiment.dropped > 20 * experiment.deviation

    # Section 8: the unreachable anisotropy is UNSOLVED.
    statuses = {target: status for target, status, _ in comparison.scan}
    assert statuses[0.973] is SolveStatus.EXACT_SOLUTION
    assert statuses[60.0] is SolveStatus.UNSOLVED
    reached = comparison.reached
    assert [point.deviation for point in reached] == sorted(point.deviation for point in reached)
    assert [point.margin for point in reached] == sorted(
        (point.margin for point in reached), reverse=True
    )


def test_the_shared_device_example_shows_two_theories_on_one_lattice() -> None:
    """``examples/shared_device.py`` reproduces the transition point and finds disjoint windows."""
    comparison = run_shared_device_example(verbose=False)

    # hz = 1 - 0.66 hx requested; E = U + 1.85 t read back.
    assert comparison.critical_slope == pytest.approx(0.66, rel=1e-6)

    # Each setting is valid for its own theory and out of regime for the other.
    assert comparison.windows_are_disjoint
    ising_here, gauge_here = comparison.ising_verdicts
    ising_there, gauge_there = comparison.gauge_verdicts
    assert gauge_here.weakest == "Delta << delta"
    assert ising_there.weakest == "delta << Gamma"
    assert ising_here.margin > 0.0 > gauge_here.margin
    assert gauge_there.margin > 0.0 > ising_there.margin

    # The two theories want opposite superlattices.
    assert comparison.ising_knobs[P.SUPERLATTICE] < 0.01
    assert comparison.gauge_knobs[P.SUPERLATTICE] > 1.0

    # Manifold weight, spectrum gap, and the cost of dropping the end-spin field.
    assert comparison.weight > 0.99
    assert comparison.gap < 0.05
    assert comparison.dropped > 100 * comparison.gap


def test_the_infeasible_example_runs_to_completion() -> None:
    """``examples/infeasible_request.py`` runs and reports INFEASIBLE."""
    assert run_infeasible_example(verbose=False) is SolveStatus.INFEASIBLE


def test_the_digital_example_runs_to_completion() -> None:
    """``examples/digital_end_to_end.py`` runs and chooses a step count."""
    steps = run_digital_example(matter_sites=3, verbose=False)
    assert steps >= 1


@pytest.mark.slow
def test_the_documentation_site_builds(tmp_path: Path) -> None:
    """``zensical build`` succeeds against ``zensical.toml`` and renders the API reference."""
    pytest.importorskip("zensical")
    pytest.importorskip("mkdocstrings")
    executable = shutil.which("zensical", path=str(Path(sys.executable).parent))
    if executable is None:  # pragma: no cover - depends on how the extra was installed
        pytest.skip("the zensical executable is not on this environment's path")
    _ = tmp_path
    # --clean rebuilds from the current sources; --strict makes any warning fail the build.
    outcome = subprocess.run(  # noqa: S603 - a fixed executable resolved above
        [executable, "build", "--clean", "--strict"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = outcome.stdout + outcome.stderr
    assert outcome.returncode == 0, combined
    assert "Build finished" in combined
    assert "unresolved" not in combined, combined

    site = ROOT / "site"
    index = (site / "index.html").read_text(encoding="utf-8")
    assert "Q-SiMod" in index
    reference = (site / "api" / "declaration" / "index.html").read_text(encoding="utf-8")
    assert "from_kilohertz" in reference
    # The API reference must actually link, not merely mention.
    assert reference.count("#qsimod.") > 500

    # Every guide page links into the reference.
    for page in ("schwinger", "heisenberg", "ising"):
        rendered = (site / page / "index.html").read_text(encoding="utf-8")
        assert "#qsimod." in rendered, page

    # No literal `[x][qsimod.y]` may remain; this fails when the autorefs plugin is not running.
    literal = {
        str(rendered.relative_to(site)): len(
            re.findall(r"\]\[qsimod\.", rendered.read_text(encoding="utf-8"))
        )
        for rendered in site.rglob("index.html")
    }
    assert not {page: count for page, count in literal.items() if count}


@pytest.mark.slow
def test_no_docstring_uses_sphinx_role_syntax() -> None:
    """No docstring uses Sphinx role syntax, which mkdocstrings renders as literal text."""
    role = re.compile(r":(mod|class|meth|func|attr|data|exc|obj|term):`")
    offenders = [
        f"{source.relative_to(ROOT)}:{number}"
        for source in sorted((ROOT / "qsimod").rglob("*.py"))
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1)
        if role.search(line)
    ]
    assert not offenders, (
        "Sphinx roles render as literal text under mkdocstrings; use "
        "[`Name`][qsimod.module.Name] instead: " + ", ".join(offenders)
    )


@pytest.mark.slow
def test_the_package_builds_a_wheel_and_an_sdist(tmp_path: Path) -> None:
    """``uv build`` produces both distributions with the code in place."""
    executable = shutil.which("uv")
    if executable is None:  # pragma: no cover - depends on the environment
        pytest.skip("uv is not available on this machine")
    outcome = subprocess.run(  # noqa: S603 - a fixed executable resolved above
        [executable, "build", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    assert list(tmp_path.glob("*.whl")), sorted(p.name for p in tmp_path.iterdir())
    assert list(tmp_path.glob("*.tar.gz")), sorted(p.name for p in tmp_path.iterdir())
