"""Time evolution of a spin chain with the existing realisation layer, in two ways.

A bound XXZ chain is realised densely, a Neel state is evolved under it exactly, and the same
evolution is repeated one Trotter step at a time with a second-order product formula, recording
the magnetisation of one site after every step.  The two trajectories are compared.

This is the smallest complete path from a symbolic model to a measured trajectory, and the
starting point for a tensor-network realisation.  Section 3,
[`evolve_state`][qsimod.realise.evolve.evolve_state], is what a TDVP backend replaces: a matrix
product state evolved under a matrix product operator built from ``model.hamiltonian``.  Section
4, the loop over [`apply_to_state`][qsimod.trotter.schedule.ProductFormulaModel.apply_to_state],
is what a TEBD backend replaces: the same k-local factors applied to a matrix product state, with
a truncation of the bond dimension after each one.  The model, the observable, the initial
state and the comparison stay as they are.

Run it with::

    uv run python examples/time_evolution.py
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from examples import Report
from qsimod.artifact import HamiltonianModel, as_hamiltonian
from qsimod.realise import HilbertSpace, build_operator, evolve_state, expectation
from qsimod.symbolic import spin_z, word
from qsimod.transformations.trotterisation import derived_layers_of, product_formula_from
from qsimod.usecases.heisenberg import ParameterNames as P
from qsimod.usecases.heisenberg import spin_chain

#: The two couplings of the chain, in the energy unit of the use case (rad/ms).
TRANSVERSE = 1.0
LONGITUDINAL = 0.5
#: The Trotter step size and the number of steps; the simulated time is their product.
STEP_SIZE = 0.05
STEPS = 40
#: The site whose magnetisation ``<S^z>`` is recorded.
PROBE_SITE = 0
#: The order of the product formula.
ORDER = 2


@dataclass(frozen=True)
class Trajectories:
    """The recorded magnetisation under both evolutions.

    Attributes:
        times: the sample times, one per Trotter step including ``t = 0``.
        exact: ``<S^z>`` at the probe site under the exact evolution.
        trotter: the same under the product formula.

    """

    times: NDArray[np.float64]
    exact: NDArray[np.float64]
    trotter: NDArray[np.float64]

    @property
    def deviation(self) -> float:
        """The largest gap between the two trajectories."""
        return float(np.max(np.abs(self.exact - self.trotter)))


def bound_chain(sites: int) -> HamiltonianModel:
    """The XXZ chain with both couplings bound; every realisation starts from a bound model."""
    values = {P.TRANSVERSE_XXZ_CHAIN: TRANSVERSE, P.LONGITUDINAL_XXZ_CHAIN: LONGITUDINAL}
    return as_hamiltonian(spin_chain(sites).bind(**values), "the time evolution example")


def neel_configuration(sites: int) -> tuple[int, ...]:
    """The occupation configuration ``|0 1 0 1 ...>`` of the Neel state."""
    return tuple(site % 2 for site in range(sites))


def main(sites: int = 6, *, verbose: bool = True) -> Trajectories:
    """Evolve the Neel state of an XXZ chain exactly and by a product formula.

    Args:
        sites: the chain length ``N``.
        verbose: whether the report is printed.

    Returns:
        The two trajectories of ``<S^z>`` at the probe site.

    """
    report = Report(verbose=verbose)
    say = report.say

    report.section("1. the model")
    model = bound_chain(sites)
    say(model)
    say(model.pretty())

    report.section("2. the dense realisation")
    space = HilbertSpace.of(model.structure)
    say(space)
    hamiltonian = build_operator(model.hamiltonian, space, model.environment())
    observable = build_operator(word(1.0, spin_z(PROBE_SITE)), space, {})
    configuration = neel_configuration(sites)
    initial = space.basis_state(configuration)
    say(f"   initial state {space.label(space.index_of(configuration))}")
    times = STEP_SIZE * np.arange(STEPS + 1, dtype=np.float64)

    report.section("3. exact evolution, from one eigendecomposition")
    # A TDVP backend replaces this call.
    exact_states = evolve_state(hamiltonian, initial, times)
    exact = np.asarray(expectation(observable, exact_states))
    say(f"   {len(times)} states of dimension {space.dimension}")

    report.section("4. Trotter evolution, one product-formula step at a time")
    step = product_formula_from(
        model, time=STEP_SIZE, steps=1, order=ORDER, layers=derived_layers_of(model)
    )
    say(step)
    say(step.layers)
    factors = step.step_factors()
    say(f"   {len(factors)} k-local factors per step; the first three:")
    for factor in factors[:3]:
        say(f"      {factor}")
    # A TEBD backend replaces this loop.
    state = initial
    trotter = [float(expectation(observable, state))]
    for _ in range(STEPS):
        state = step.apply_to_state(state)
        trotter.append(float(expectation(observable, state)))
    say(f"   norm of the final state {float(jnp.linalg.norm(state)):.12f}")

    result = Trajectories(times=times, exact=exact, trotter=np.asarray(trotter))
    report.section(f"5. <S^z_{PROBE_SITE}> over t in [0, {times[-1]:g}]")
    say(f"   {'t':>6}{'exact':>12}{'Trotter':>12}")
    for index in range(0, STEPS + 1, 5):
        say(f"   {times[index]:6.2f}{exact[index]:12.6f}{result.trotter[index]:12.6f}")
    say(f"   largest deviation {result.deviation:.3e}")
    return result


if __name__ == "__main__":
    main()
