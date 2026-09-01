"""The digital branch of the Schwinger case study, end to end.

The qubit Hamiltonian ``L2d`` (``H_IR4`` of the article) is bound at the reference parameters,
the step count of the Trotter product formula ``L3b`` (the digital simulator model
``U_approx``) that meets a target error bound is solved for, and the resource summary, the
measured error against the a-priori error bound, and the Gauss-law violation before and after
the product formula are reported.  The parameters ``m = 0.41``, ``g = 0.83``, ``t = 2`` are
those of the published reference tables.

Run it with::

    uv run python examples/digital_end_to_end.py
"""

from __future__ import annotations

import jax.numpy as jnp

from examples import Report
from qsimod.artifact import HamiltonianModel, as_hamiltonian
from qsimod.realise import (
    HilbertSpace,
    build_operator,
    gauss_violation,
    max_abs_deviation,
    propagator,
    spectral_norm,
)
from qsimod.solving import SolveStatus
from qsimod.solving.backends.integer import minimal_resource_setting
from qsimod.solving.stepcount import STEPS_SYMBOL, minimal_steps, resource_candidates
from qsimod.transformations import interleaved_layers_of
from qsimod.trotter import ProductFormulaModel, SpectralNormEstimator, as_product_formula
from qsimod.usecases.schwinger import (
    ParameterNames as P,
)
from qsimod.usecases.schwinger import (
    build_graph,
    canonical_state_configuration,
    particle_hole,
    to_qubits,
    trotterisation,
)

#: The parameters of the published reference tables.
TARGET_MASS = 0.41
TARGET_COUPLING = 0.83
SIMULATED_TIME = 2.0
TARGET_ERROR = 1e-3


def main(matter_sites: int = 4, *, verbose: bool = True) -> int:
    """Run the digital pipeline and return the step count found.

    Args:
        matter_sites: the chain length ``N``; the reference tables use 4.
        verbose: whether the report is printed.

    Returns:
        The Trotter step count found by the integer solve.

    Raises:
        RuntimeError: if the step-count solve fails.

    """
    graph = build_graph(matter_sites)
    report = Report(verbose=verbose)
    say = report.say

    report.section("1. the digital pipeline")
    say(graph.digital)

    # kappa is a free parameter from L2a on, so it is bound there.
    staggered = graph.graph.node("L2a").bind(
        **{P.MASS_L2A: TARGET_MASS, P.COUPLING_L2A: TARGET_COUPLING}
    )
    homogeneous = particle_hole().apply(staggered)
    qubits = as_hamiltonian(to_qubits().apply(homogeneous), "the digital branch")
    say()
    report.section("2. the qubit Hamiltonian")
    say(qubits.pretty())

    layers = interleaved_layers_of(qubits)
    say()
    report.section("3. the layer decomposition, and which pairs commute")
    say(layers)
    say(layers.commutation_report())

    estimator = SpectralNormEstimator(len(qubits.structure.sites))
    say()
    report.section(f"4. the step count: the least n with bound <= {TARGET_ERROR:g}")
    candidates = resource_candidates(
        layers, SIMULATED_TIME, TARGET_ERROR, orders=(1, 2, 4), estimator=estimator
    )
    for candidate in candidates:
        say(f"   {candidate}")
    if not candidates:
        msg = "no order met the accuracy target"
        raise RuntimeError(msg)
    best = minimal_resource_setting(candidates)
    say(f"   resource-minimal: {best}")

    outcome = minimal_steps(layers, best.order, SIMULATED_TIME, TARGET_ERROR, estimator=estimator)
    if outcome.status is not SolveStatus.EXACT_SOLUTION:
        msg = f"the step-count solve failed: {outcome.status}"
        raise RuntimeError(msg)
    steps = round(outcome.point[STEPS_SYMBOL])
    say()
    say(f"   {outcome.notes['minimality']}")
    say(f"   {outcome.termination}")

    product_formula = as_product_formula(
        trotterisation(time=SIMULATED_TIME, steps=steps, order=best.order).apply(qubits)
    )
    say()
    report.section("5. the resource summary")
    say(product_formula.resources(estimator))

    _report_error(product_formula, qubits, estimator, report)
    _report_gauge_invariance(product_formula, qubits, matter_sites, report)
    return steps


def _report_error(
    product_formula: ProductFormulaModel,
    qubits: HamiltonianModel,
    estimator: SpectralNormEstimator,
    report: Report,
) -> None:
    """Print the measured error against the a-priori error bound, in both norms."""
    say = report.say
    say()
    report.section("6. measured error against the a-priori bound")
    space = HilbertSpace.of(qubits.structure)
    exact_hamiltonian = build_operator(qubits.hamiltonian, space, qubits.environment())
    exact = propagator(exact_hamiltonian, product_formula.time)
    approximate = product_formula.matrix()
    bound = product_formula.error_bound(estimator)
    spectral = spectral_norm(approximate - exact)
    largest_entry = max_abs_deviation(approximate, exact)
    say(f"   measured error, spectral norm             {spectral:.6e}")
    say(f"   measured error, largest absolute entry    {largest_entry:.6e}")
    say(f"   a-priori bound (a spectral-norm bound)    {bound.value:.6e}")
    say(f"   slack against the matched (spectral) norm {bound.value / spectral:.2f}x")
    say(f"   bound family used                         {bound.method}")


def _report_gauge_invariance(
    product_formula: ProductFormulaModel,
    qubits: HamiltonianModel,
    matter_sites: int,
    report: Report,
) -> None:
    """Print the Gauss-law violation before and after the product formula."""
    say = report.say
    say()
    report.section("7. gauge invariance")
    space = HilbertSpace.of(qubits.structure)
    environment = qubits.environment()
    constraints = [
        build_operator(constraint.operator, space, environment)
        for constraint in qubits.constraint_operators
    ]
    initial = space.basis_state(canonical_state_configuration(matter_sites))
    evolved = product_formula.apply_to_state(initial)
    before = float(gauss_violation(constraints, initial))
    after = float(gauss_violation(constraints, evolved))
    say(f"   sum_l <G_l^2> initially                 {before:.15f}")
    say(f"   sum_l <G_l^2> after the product formula {after:.15f}")
    say(f"   norm of the evolved state               {float(jnp.linalg.norm(evolved)):.15f}")


if __name__ == "__main__":
    main()
