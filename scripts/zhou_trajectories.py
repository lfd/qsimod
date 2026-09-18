"""Time-resolved trajectories of the two knob settings posed against `paper_zhou22`.

For the free-knob setting (the framework's solve) and the prescribed-knob setting (the
experiment's own knobs), records at every sample time over the experiment's window the mean
matter occupation under the device Hamiltonian ``bose_hubbard`` (``occupation_hw``) and under the
effective theory ``effective_bosonic`` (``occupation_theory``),
the weight outside the declared occupation subspace (leakage) and the gauge violation
``eta``.  The settings, request and window are those of ``examples/analogue_end_to_end.py``.

Two realisations are offered.  ``full`` builds both models in the full bosonic space with
cutoff ``n_max = 2``, of dimension ``3**(2N - 1)``.  ``number-sector`` builds them in the
sector of ``N`` atoms, which the Bose-Hubbard model conserves; it holds every state the device
can reach from ``|1 0 1 0 1 ...>``, leaked ones included, and is far smaller, so longer chains
are within reach of a dense eigendecomposition.

Writes ``results/zhou_trajectories.csv``, one row per setting and sample time.  Run it with::

    uv run python scripts/zhou_trajectories.py --matter-sites 6 --realisation number-sector
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import jax.numpy as jnp
import numpy as np
import pandas as pd
from examples.analogue_end_to_end import (
    ELECTRIC_GAP,
    SAMPLES,
    TARGET_COUPLING_HZ,
    TARGET_MASS_HZ,
    WINDOW_MS,
    Bench,
    device_box,
    experiment_knobs,
)
from jax import Array

from qsimod.artifact import ConstraintOperator
from qsimod.models.gauge import (
    canonical_state_configuration,
    gauge_violation_observable,
    local_occupation_subspace,
    matter_occupation_observable,
)
from qsimod.realise import (
    RealisationRequest,
    SectorBasis,
    build_operator,
    build_operator_in,
    evolve_state,
    expectation,
    sandwich,
)
from qsimod.solving import realise_parameters
from qsimod.symbolic import OperatorSum, number, word
from qsimod.units import from_hertz, to_hertz
from qsimod.usecases.schwinger import ParameterNames as P
from qsimod.usecases.schwinger import (
    build_graph,
    coupling_map,
    effective_bosonic,
    mass_map,
    superlattice,
)
from scripts import write_frame

__all__ = [
    "FullSpaceBench",
    "NumberSectorBench",
    "Realisation",
    "free_knobs",
    "main",
    "total_atom_number",
    "trajectories",
]

Realisation: TypeAlias = Literal["full", "number-sector"]

#: The four hardware knobs, in the order they are solved for.
KNOBS = (P.TUNNELLING, P.INTERACTION, P.SUPERLATTICE, P.TILT)


def total_atom_number(matter_sites: int) -> ConstraintOperator:
    """The total atom number over the interleaved register, with target ``N``.

    The Bose-Hubbard model conserves it, and the initial state carries one atom per matter
    site.

    Args:
        matter_sites: the chain length ``N``.

    Returns:
        The constraint operator.

    """
    total = OperatorSum()
    for site in range(2 * matter_sites - 1):
        total = total + word(1.0, number(site))
    return ConstraintOperator(
        name="N_atoms",
        index=0,
        operator=total.renamed("N_atoms"),
        target_value=float(matter_sites),
    )


@dataclass(frozen=True)
class FullSpaceBench:
    """The example's full-space realisation, behind the interface `trajectories` needs.

    Attributes:
        bench: the example's bench.

    """

    bench: Bench

    @classmethod
    def of(cls, matter_sites: int) -> FullSpaceBench:
        """Realise both models in the full space at a given chain length."""
        return cls(Bench.of(matter_sites))

    @property
    def dimension(self) -> int:
        """The dimension both models are realised in."""
        return self.bench.space.dimension

    @property
    def projector(self) -> Array:
        """The projector onto the declared occupation subspace."""
        return self.bench.projector

    @property
    def initial(self) -> Array:
        """The ``|1 0 1 0 1 ...>`` initial state."""
        return self.bench.initial

    @property
    def occupation(self) -> Array:
        """The realised ``<n_matter>`` observable."""
        return self.bench.occupation

    @property
    def violation(self) -> Array:
        """The realised gauge-violation observable."""
        return self.bench.violation

    @property
    def device(self) -> OperatorSum:
        """The Hamiltonian of ``bose_hubbard``, unbound."""
        return self.bench.device

    @property
    def theory(self) -> OperatorSum:
        """The Hamiltonian of ``effective_bosonic``, unbound."""
        return self.bench.theory

    def build(self, operator: OperatorSum, values: Mapping[str, float]) -> Array:
        """Realise an operator with its parameters bound."""
        return build_operator(operator, self.bench.space, dict(values))


@dataclass(frozen=True)
class NumberSectorBench:
    """Both models realised in the sector of ``N`` atoms.

    Attributes:
        basis: the sector's basis.
        projector: the projector onto the declared occupation subspace, in the sector.
        initial: the ``|1 0 1 0 1 ...>`` initial state.
        occupation: the realised ``<n_matter>`` observable.
        violation: the realised gauge-violation observable.
        device: the Hamiltonian of ``bose_hubbard``, unbound.
        theory: the Hamiltonian of ``effective_bosonic``, unbound.

    """

    basis: SectorBasis
    projector: Array
    initial: Array
    occupation: Array
    violation: Array
    device: OperatorSum
    theory: OperatorSum

    @classmethod
    def of(cls, matter_sites: int) -> NumberSectorBench:
        """Realise both models in the atom-number sector at a given chain length."""
        device_model = superlattice(matter_sites)
        theory_model = effective_bosonic(matter_sites)
        subspace = local_occupation_subspace(matter_sites)
        basis = SectorBasis.of(
            device_model.structure,
            constraints=(total_atom_number(matter_sites),),
            request=RealisationRequest(boson_cutoff=subspace.required_cutoff()),
        )
        allowed = subspace.allowed_occupations
        mask = [
            float(
                all(
                    occupation in allowed[site]
                    for site, occupation in zip(basis.space.sites, configuration, strict=True)
                )
            )
            for configuration in basis.configurations
        ]
        return cls(
            basis=basis,
            projector=jnp.diag(jnp.asarray(mask, dtype=jnp.complex128)),
            initial=basis.state(canonical_state_configuration(matter_sites)),
            occupation=build_operator_in(matter_occupation_observable(matter_sites), basis, {}),
            violation=build_operator_in(gauge_violation_observable(matter_sites), basis, {}),
            device=device_model.hamiltonian,
            theory=theory_model.hamiltonian,
        )

    @property
    def dimension(self) -> int:
        """The dimension both models are realised in."""
        return self.basis.dimension

    def build(self, operator: OperatorSum, values: Mapping[str, float]) -> Array:
        """Realise an operator with its parameters bound."""
        return build_operator_in(operator, self.basis, dict(values))


def free_knobs(matter_sites: int) -> dict[str, float]:
    """The framework's solution for the request of `paper_zhou22`, in rad/ms.

    Args:
        matter_sites: the chain length ``N`` the pipeline is built at.

    Returns:
        ``J``, ``U``, ``delta`` and ``Delta``.

    Raises:
        RuntimeError: if the solve does not return a usable point.

    """
    pipeline = build_graph(matter_sites).analogue
    targets = {
        P.MASS_LATTICE_QED: from_hertz(TARGET_MASS_HZ),
        P.COUPLING_QUANTUM_LINK_STAGGERED: from_hertz(TARGET_COUPLING_HZ),
        P.ELECTRIC_GAP: ELECTRIC_GAP,
    }
    result = realise_parameters(
        pipeline, targets=targets, unknowns=list(KNOBS), admissible_set=device_box()
    )
    if not result.status.is_success:
        msg = f"the solve did not produce a usable point: {result.status}"
        raise RuntimeError(msg)
    return result.subset(list(KNOBS))


def trajectories(
    bench: FullSpaceBench | NumberSectorBench,
    label: str,
    knobs: Mapping[str, float],
) -> pd.DataFrame:
    """The four observables of one knob setting at every sample time of the window.

    Args:
        bench: the shared realisation.
        label: the setting's name, written to the ``setting`` column.
        knobs: the four hardware knobs, in rad/ms.

    Returns:
        A tidy frame with one row per sample time; the knobs are repeated per row, in Hz.

    """
    effective = {
        P.MASS_EFFECTIVE_BOSONIC: mass_map().evaluate_real(knobs),
        P.COUPLING_EFFECTIVE_BOSONIC: coupling_map().evaluate_real(knobs),
    }
    device = bench.build(bench.device, knobs)
    theory = sandwich(bench.build(bench.theory, effective), bench.projector)

    times = np.linspace(0.0, WINDOW_MS, SAMPLES)
    device_states = evolve_state(device, bench.initial, list(times))
    theory_states = evolve_state(theory, bench.initial, list(times))
    return pd.DataFrame(
        {
            "setting": label,
            "time_ms": times,
            "occupation_hw": np.asarray(expectation(bench.occupation, device_states)),
            "occupation_theory": np.asarray(expectation(bench.occupation, theory_states)),
            "leakage": np.asarray(1.0 - expectation(bench.projector, device_states)),
            "violation": np.asarray(expectation(bench.violation, device_states)),
            "J_hz": to_hertz(knobs[P.TUNNELLING]),
            "U_hz": to_hertz(knobs[P.INTERACTION]),
            "delta_hz": to_hertz(knobs[P.SUPERLATTICE]),
            "Delta_hz": to_hertz(knobs[P.TILT]),
        }
    )


def main(
    matter_sites: int = 3,
    realisation: Realisation = "full",
    output: Path | None = None,
) -> Path:
    """Compute both settings' trajectories and write them to CSV.

    Args:
        matter_sites: the chain length ``N``.
        realisation: ``"full"`` for the full bosonic space, ``"number-sector"`` for the
            sector of ``N`` atoms.
        output: the destination, or ``None`` for ``results/zhou_trajectories.csv``.

    Returns:
        The path written to.

    """
    bench = (
        NumberSectorBench.of(matter_sites)
        if realisation == "number-sector"
        else FullSpaceBench.of(matter_sites)
    )
    print(f"N = {matter_sites}, {realisation} realisation, dimension {bench.dimension}")
    frame = pd.concat(
        [
            trajectories(bench, "free knobs", free_knobs(matter_sites)),
            trajectories(bench, "prescribed knobs", experiment_knobs()),
        ],
        ignore_index=True,
    )
    frame.insert(0, "matter_sites", matter_sites)
    frame.insert(1, "realisation", realisation)
    return write_frame(frame, output, "zhou_trajectories.csv")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matter-sites", type=int, default=3, help="the chain length N")
    parser.add_argument(
        "--realisation",
        choices=("full", "number-sector"),
        default="full",
        help="full bosonic space, or the sector of N atoms",
    )
    parser.add_argument("--output", type=Path, default=None, help="destination CSV")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse()
    print(main(arguments.matter_sites, arguments.realisation, arguments.output))
