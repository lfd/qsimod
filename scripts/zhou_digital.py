"""The digital branch posed against the request of Zhou et al. (2022), at the analogue setup.

The same theory, chain length, initial state, observable and time window as
``scripts/zhou_trajectories.py``: ``N`` matter sites, ``m = 0``, ``kappa = 14.5 Hz``, the quench
from ``|1 0 1 0 1 ...>`` and the mean matter occupation over 150 ms.  The digital simulator is
the Suzuki-Trotter product formula of the qubit Hamiltonian ``qubit_register``; its knobs are
the order and the step count ``n``.

For every ``(order, n)`` of a ladder the script records the resources (factor count, depth),
the a-priori error bound, and the measured deviation of the mean matter occupation from the
exact evolution of ``qubit_register``, taken over the trajectory at every Trotter step.  For a
list of target accuracies, e.g. the deviations the analogue device reaches, it records the smallest
``n`` per order by the bound (the framework's integer solve) and by the measurement.

Writes ``results/zhou_digital_ladder.csv`` and ``results/zhou_digital_targets.csv``, and the
full trajectories of the mean matter occupation, digital against exact at every Trotter step,
to ``results/zhou_digital_trajectories.csv`` for the orders named by ``--trajectory-orders``
(default: order 2 only).  Run it with::

    uv run python scripts/zhou_digital.py --matter-sites 6 --targets 0.064 0.053 0.01 0.001
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from examples.analogue_end_to_end import SAMPLES, TARGET_COUPLING_HZ, TARGET_MASS_HZ, WINDOW_MS
from jax import Array

from qsimod.artifact import HamiltonianModel, as_hamiltonian
from qsimod.pauli import PauliSum
from qsimod.realise import HilbertSpace, build_operator, evolve_state, expectation
from qsimod.solving.stepcount import resource_candidates
from qsimod.transformations import interleaved_layers_of
from qsimod.trotter import SpectralNormEstimator, as_product_formula
from qsimod.trotter.bounds import NormEstimator
from qsimod.trotter.layers import LayerDecomposition
from qsimod.units import from_hertz
from qsimod.usecases.schwinger import ParameterNames as P
from qsimod.usecases.schwinger import (
    build_graph,
    canonical_state_configuration,
    matter_occupation_observable,
    particle_hole,
    to_qubits,
    trotterisation,
)
from scripts import write_frame

__all__ = [
    "ORDERS",
    "STEP_LADDER",
    "CachedSpectralNormEstimator",
    "DigitalBench",
    "main",
    "measure",
    "qubit_model",
]

#: The product-formula orders compared.
ORDERS = (1, 2, 4)

#: The step counts the measured deviation is taken at.
STEP_LADDER = (5, 10, 20, 50, 100, 200, 500, 1000)

#: The default accuracy targets: the analogue deviations at ``N = 6``, and two round numbers.
DEFAULT_TARGETS = (0.064, 0.053, 1e-2, 1e-3)


@dataclass(frozen=True)
class CachedSpectralNormEstimator(NormEstimator):
    """The exact spectral norm, computed once per distinct Pauli sum.

    The error bounds of a product formula need the norms of the same layer commutators at
    every step count; on a register of eleven qubits each norm is a dense diagonalisation,
    so they are memoised on the Pauli sum's terms.

    Attributes:
        inner: the estimator that computes a norm the first time it is asked for.

    """

    inner: SpectralNormEstimator
    name: str = field(default="spectral norm (exact, cached)", init=False)
    _cache: dict[frozenset[tuple[object, complex]], float] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def norm(self, operator: PauliSum) -> float:
        """``||operator||`` in the spectral norm, from the cache where possible."""
        key = frozenset(
            (string, complex(round(value.real, 12), round(value.imag, 12)))
            for string, value in operator.terms.items()
        )
        if key not in self._cache:
            self._cache[key] = self.inner.norm(operator)
        return self._cache[key]

    def is_exact(self) -> bool:
        """The cached values are exact spectral norms."""
        return True


def qubit_model(matter_sites: int, mass: float, coupling: float) -> HamiltonianModel:
    """``qubit_register`` bound to ``(m, kappa)``, via the particle-hole and Jordan-Wigner steps."""
    staggered = (
        build_graph(matter_sites)
        .graph.node("quantum_link_staggered")
        .bind(**{P.MASS_QUANTUM_LINK_STAGGERED: mass, P.COUPLING_QUANTUM_LINK_STAGGERED: coupling})
    )
    return as_hamiltonian(to_qubits().apply(particle_hole().apply(staggered)))


@dataclass(frozen=True)
class DigitalBench:
    """The qubit Hamiltonian realised, with the exact evolution as reference.

    Attributes:
        qubits: the bound qubit Hamiltonian model.
        layers: its internally commuting layer decomposition.
        space: the qubit register's Hilbert space.
        hamiltonian: the realised Hamiltonian.
        observable: the realised ``<n_matter>``.
        initial: the ``|1 0 1 0 1 ...>`` initial state.
        estimator: the norm estimator the bounds use.

    """

    qubits: HamiltonianModel
    layers: LayerDecomposition
    space: HilbertSpace
    hamiltonian: Array
    observable: Array
    initial: Array
    estimator: CachedSpectralNormEstimator

    @classmethod
    def of(cls, matter_sites: int, mass: float, coupling: float) -> DigitalBench:
        """Realise the qubit model at a given chain length and ``(m, kappa)``."""
        qubits = qubit_model(matter_sites, mass, coupling)
        space = HilbertSpace.of(qubits.structure)
        return cls(
            qubits=qubits,
            layers=interleaved_layers_of(qubits),
            space=space,
            hamiltonian=build_operator(qubits.hamiltonian, space, qubits.environment()),
            observable=build_operator(matter_occupation_observable(matter_sites), space, {}),
            initial=space.basis_state(canonical_state_configuration(matter_sites)),
            estimator=CachedSpectralNormEstimator(
                SpectralNormEstimator(len(qubits.structure.sites))
            ),
        )

    def exact_occupation(self, times: Sequence[float] | np.ndarray) -> np.ndarray:
        """``<n_matter>(t)`` under the exact evolution of the qubit Hamiltonian."""
        states = evolve_state(self.hamiltonian, self.initial, times)
        return np.asarray(expectation(self.observable, states))


def measure(
    bench: DigitalBench, order: int, steps: int, time: float
) -> tuple[dict[str, object], pd.DataFrame]:
    """Resources, bound and measured deviation of one ``(order, n)``, and its trajectory.

    Returns:
        The summary row, and a frame with one row per Trotter step holding the time, the
        digital and the exact mean matter occupation.

    """
    formula = as_product_formula(
        trotterisation(time=time, steps=steps, order=order).apply(bench.qubits)
    )
    step = formula.step_matrix()
    states = [bench.initial]
    for _ in range(steps):
        states.append(step @ states[-1])
    digital = np.asarray(expectation(bench.observable, jnp.stack(states)))
    times = np.linspace(0.0, time, steps + 1)
    exact = bench.exact_occupation(times)
    difference = np.abs(digital - exact)
    trajectory = pd.DataFrame(
        {
            "order": order,
            "steps": steps,
            "step_index": np.arange(steps + 1),
            "time_ms": times,
            "occupation_hw": digital,
            "occupation_theory": exact,
        }
    )
    resources = formula.resources(bench.estimator)
    bound = formula.error_bound(bench.estimator)
    return {
        "order": order,
        "steps": steps,
        "step_ms": time / steps,
        "factors_per_step": resources.factors_per_step,
        "factor_count": resources.factor_count,
        "depth_in_layers": resources.depth_in_layers,
        "error_bound": bound.value,
        "bound_method": bound.method,
        "deviation_max": float(difference.max()),
        "deviation_final": float(difference[-1]),
    }, trajectory


def main(
    matter_sites: int = 6,
    targets: Sequence[float] = DEFAULT_TARGETS,
    trajectories: Path | None = None,
    output_dir: Path | None = None,
    orders: Sequence[int] = ORDERS,
    trajectory_orders: Sequence[int] = (2,),
) -> tuple[Path, Path, Path]:
    """Run the ladder and the target solves and write both CSVs.

    Args:
        matter_sites: the chain length ``N``.
        targets: the accuracy targets on the mean matter occupation.
        trajectories: a ``zhou_trajectories.csv`` to check the exact reference against; its
            ``occupation_theory`` column is the same theory in the bosonic encoding.
        output_dir: where to write, or ``None`` for ``results/``.
        orders: the product-formula orders to run.
        trajectory_orders: the orders whose full trajectories are written.

    Returns:
        The paths of the ladder, the targets and the trajectories CSV.

    """
    mass, coupling, time = from_hertz(TARGET_MASS_HZ), from_hertz(TARGET_COUPLING_HZ), WINDOW_MS
    bench = DigitalBench.of(matter_sites, mass, coupling)
    print(
        f"N = {matter_sites}: {len(bench.qubits.structure.sites)} qubits, "
        f"dimension {bench.space.dimension}",
        flush=True,
    )

    if trajectories is not None and trajectories.is_file():
        frame = pd.read_csv(trajectories)
        frame = frame[frame.matter_sites == matter_sites]
        if not frame.empty:
            theory = frame[frame.setting == frame.setting.iloc[0]].occupation_theory.to_numpy()
            exact = bench.exact_occupation(np.linspace(0.0, time, SAMPLES))
            print(
                "exact qubit_register vs analogue theory effective_bosonic: "
                f"max |diff| = {np.abs(exact - theory).max():.2e}",
                flush=True,
            )

    rows = []
    paths: list[pd.DataFrame] = []
    for order in orders:
        for steps in STEP_LADDER:
            row, path = measure(bench, order, steps, time)
            rows.append(row)
            if order in trajectory_orders:
                paths.append(path)
            print(
                f"  order {order}  n = {steps:5d}  bound {row['error_bound']:.3e}  "
                f"deviation {row['deviation_max']:.3e}",
                flush=True,
            )
    ladder = pd.DataFrame(rows)

    target_rows = []
    for target in targets:
        by_bound = {
            c.order: c
            for c in resource_candidates(
                bench.layers, time, target, orders=orders, estimator=bench.estimator
            )
        }
        for order in orders:
            met = ladder[(ladder.order == order) & (ladder.deviation_max <= target)]
            choice = by_bound.get(order)
            target_rows.append(
                {
                    "target": target,
                    "order": order,
                    "steps_by_bound": choice.steps if choice else float("nan"),
                    "factor_count_by_bound": choice.cost if choice else float("nan"),
                    "depth_by_bound": choice.depth_in_layers if choice else float("nan"),
                    "error_bound": choice.error_bound if choice else float("nan"),
                    "steps_by_measurement": int(met.steps.min()) if not met.empty else float("nan"),
                    "factor_count_by_measurement": int(
                        met.sort_values("steps").factor_count.iloc[0]
                    )
                    if not met.empty
                    else float("nan"),
                }
            )
    targets_frame = pd.DataFrame(target_rows)
    columns = ["order", "steps", "step_index", "time_ms", "occupation_hw", "occupation_theory"]
    trajectory_frame = (
        pd.concat(paths, ignore_index=True) if paths else pd.DataFrame(columns=columns)
    )
    for frame_ in (ladder, targets_frame):
        frame_.insert(0, "matter_sites", matter_sites)
        frame_.insert(1, "time_ms", time)
    trajectory_frame.insert(0, "matter_sites", matter_sites)  # it carries time_ms per step
    base = output_dir or Path("results")
    return (
        write_frame(ladder, base / "zhou_digital_ladder.csv", "zhou_digital_ladder.csv"),
        write_frame(targets_frame, base / "zhou_digital_targets.csv", "zhou_digital_targets.csv"),
        write_frame(
            trajectory_frame,
            base / "zhou_digital_trajectories.csv",
            "zhou_digital_trajectories.csv",
        ),
    )


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matter-sites", type=int, default=6, help="the chain length N")
    parser.add_argument(
        "--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS), help="accuracy targets"
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("results/zhou_trajectories.csv"),
        help="analogue trajectories to check the reference against",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="destination directory")
    parser.add_argument(
        "--orders", type=int, nargs="+", default=list(ORDERS), help="product-formula orders to run"
    )
    parser.add_argument(
        "--trajectory-orders",
        type=int,
        nargs="+",
        default=[2],
        help="orders whose full trajectories are written",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse()
    for path in main(
        arguments.matter_sites,
        arguments.targets,
        arguments.trajectories,
        arguments.output_dir,
        orders=arguments.orders,
        trajectory_orders=arguments.trajectory_orders,
    ):
        print(path)
