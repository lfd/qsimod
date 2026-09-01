"""The product-formula artifact, the digital simulator model ``U_approx`` of the article.

The artifact is an ordered sequence of k-local unitaries on a qubit register, parametrised by
the simulated time ``t``, the step count ``n`` and the order of the product formula.
[`ProductFormulaModel`][qsimod.trotter.schedule.ProductFormulaModel] stores the layer
decomposition and the stage schedule; the ``(Pauli string, angle)`` factors of one step are
derived on demand and the step is repeated ``n`` times.  Factors are listed from left to right
as written, so that the rightmost factor acts first on a state.

Depth convention: a depth layer is a maximal run of consecutive factors with pairwise disjoint
support; the depth is computed for one step and multiplied by ``n``, without packing across
step boundaries.  Resources are counted as k-local unitaries and depth layers only; no gate-set
decomposition, CNOT count or routing is performed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
from jax import Array

from qsimod.artifact import Artifact, ArtifactKind, ArtifactKindError
from qsimod.jax_setup import COMPLEX_DTYPE, require_x64
from qsimod.pauli import PauliAxis, PauliString
from qsimod.trotter.bounds import ErrorBound, NormEstimator, error_bound
from qsimod.trotter.formulas import ProductFormulaSchedule
from qsimod.trotter.layers import LayerDecomposition

__all__ = [
    "ProductFormulaModel",
    "ResourceSummary",
    "UnitaryFactor",
    "as_product_formula",
    "factors_for",
    "global_phase_for",
    "layer_depth_of",
]


@dataclass(frozen=True)
class UnitaryFactor:
    """One k-local unitary ``exp(-i angle * string)`` of a product formula.

    Attributes:
        string: the Hermitian Pauli string that is exponentiated.
        angle: the real rotation angle.
        layer: the name of the Hamiltonian layer from which the factor originates.

    """

    string: PauliString
    angle: float
    layer: str

    @property
    def support(self) -> frozenset[int]:
        """The qubits on which the factor acts."""
        return self.string.support

    @property
    def locality(self) -> int:
        """The number ``k`` of qubits on which the factor acts."""
        return self.string.weight

    def __str__(self) -> str:
        return f"exp(-i {self.angle:+.6g} {self.string})  [{self.layer}]"


def factors_for(
    decomposition: LayerDecomposition,
    schedule: ProductFormulaSchedule,
    step_size: float,
) -> tuple[UnitaryFactor, ...]:
    """The k-local unitary factors of one step of ``schedule`` at step size ``step_size``.

    ``result[0]`` is the leftmost factor of the product.  Identity Pauli strings are excluded;
    their contribution is reported by
    [`global_phase_for`][qsimod.trotter.schedule.global_phase_for].
    """
    factors: list[UnitaryFactor] = []
    for stage in schedule.stages:
        layer = decomposition.layers[stage.layer]
        for string in layer.operator.strings:
            if string.is_identity:
                continue
            factors.append(
                UnitaryFactor(
                    string=string,
                    angle=stage.coefficient * step_size * layer.operator.terms[string].real,
                    layer=layer.name,
                )
            )
    return tuple(factors)


def global_phase_for(
    decomposition: LayerDecomposition,
    schedule: ProductFormulaSchedule,
    step_size: float,
) -> float:
    """The angle contributed by identity Pauli strings in one step."""
    total = 0.0
    for stage in schedule.stages:
        layer = decomposition.layers[stage.layer]
        total += stage.coefficient * step_size * layer.operator.identity_coefficient().real
    return total


def layer_depth_of(factors: Sequence[UnitaryFactor]) -> int:
    """The depth of an ordered factor sequence, counted in layers of factors with disjoint support.

    The assignment is greedy and respects the order: a factor joins the current layer if its
    support is disjoint from every support in that layer; otherwise a new layer is started.
    """
    depth = 0
    occupied: set[int] = set()
    for factor in factors:
        support = factor.support
        if not support:
            continue
        if occupied & support:
            depth += 1
            occupied = set(support)
            continue
        if not occupied:
            depth += 1
        occupied |= support
    return depth


@dataclass(frozen=True)
class ResourceSummary:
    """The resource summary of a product formula, independent of any particular hardware.

    Attributes:
        qubits: the register size.
        steps: the Trotter step count ``n``.
        order: the order of the product formula.
        time: the simulated time ``t``.
        factors_per_step: the number of k-local unitary factors in one step.
        factor_count: the total number of factors, ``factors_per_step * steps``.
        factors_by_locality: the number of factors acting on each number of qubits, totalled
            over all steps.
        depth_per_step: the depth of one step in layers of factors with disjoint support.
        depth_in_layers: the total depth ``depth_per_step * steps``, without packing across
            step boundaries.
        hamiltonian_layers: the number of layers of mutually commuting terms in the
            Hamiltonian.
        stages_per_step: the number of exponentials in one step of the formula.
        global_phase_per_step: the angle contributed by identity Pauli strings.
        error_bound: the a-priori error bound at the given settings.
        error_bound_method: the family of bounds that produced ``error_bound``.
        exact_norms: whether the norms entering the bound were exact spectral norms.

    """

    qubits: int
    steps: int
    order: int
    time: float
    factors_per_step: int
    factor_count: int
    factors_by_locality: Mapping[int, int]
    depth_per_step: int
    depth_in_layers: int
    hamiltonian_layers: int
    stages_per_step: int
    global_phase_per_step: float
    error_bound: float
    error_bound_method: str
    exact_norms: bool

    def as_row(self) -> dict[str, float | int | str]:
        """A flat mapping, suitable for instance as a ``pandas`` row."""
        row: dict[str, float | int | str] = {
            "qubits": self.qubits,
            "steps": self.steps,
            "order": self.order,
            "time": self.time,
            "factors_per_step": self.factors_per_step,
            "factor_count": self.factor_count,
            "depth_per_step": self.depth_per_step,
            "depth_in_layers": self.depth_in_layers,
            "hamiltonian_layers": self.hamiltonian_layers,
            "stages_per_step": self.stages_per_step,
            "error_bound": self.error_bound,
            "error_bound_method": self.error_bound_method,
            "exact_norms": self.exact_norms,
        }
        for locality, count in sorted(self.factors_by_locality.items()):
            row[f"factors_k{locality}"] = count
        return row

    def __str__(self) -> str:
        breakdown = ", ".join(
            f"k={locality}: {count}" for locality, count in sorted(self.factors_by_locality.items())
        )
        return "\n".join(
            [
                f"resource summary (t={self.time:g}, n={self.steps}, order {self.order}):",
                f"  qubits                 {self.qubits}",
                f"  k-local factors        {self.factor_count} "
                f"({self.factors_per_step} per step) [{breakdown}]",
                f"  depth in layers        {self.depth_in_layers} "
                f"({self.depth_per_step} per step; no packing across steps)",
                f"  Hamiltonian layers     {self.hamiltonian_layers}",
                f"  stages per step        {self.stages_per_step}",
                f"  global phase per step  {self.global_phase_per_step:+.6g}",
                f"  a-priori error bound   {self.error_bound:.6e} "
                f"[{self.error_bound_method}, "
                f"{'exact' if self.exact_norms else 'bounded'} norms]",
            ]
        )


@dataclass(frozen=True)
class ProductFormulaModel(Artifact):
    """An ordered product of k-local unitaries on a qubit register, the digital simulator model.

    Attributes:
        layers: the partition of the Hamiltonian into layers of mutually commuting terms.
        formula: the stage schedule of the product formula.
        time: the simulated time ``t``.
        steps: the Trotter step count ``n``.

    """

    layers: LayerDecomposition = field(kw_only=True)
    formula: ProductFormulaSchedule = field(kw_only=True)
    time: float = 1.0
    steps: int = 1

    def __post_init__(self) -> None:
        if self.steps < 1:
            msg = f"{self.name}: step count must be positive, got {self.steps}"
            raise ValueError(msg)
        if self.formula.layer_count != len(self.layers):
            msg = (
                f"{self.name}: the schedule addresses {self.formula.layer_count} layer(s) "
                f"but the decomposition has {len(self.layers)}"
            )
            raise ValueError(msg)

    @property
    def kind(self) -> ArtifactKind:
        """Always [`PRODUCT_FORMULA`][qsimod.artifact.ArtifactKind.PRODUCT_FORMULA]."""
        return ArtifactKind.PRODUCT_FORMULA

    @property
    def order(self) -> int:
        """The order of the product formula."""
        return self.formula.order

    @property
    def qubit_count(self) -> int:
        """The register size, from the structural type."""
        return len(self.structure.sites)

    @property
    def step_size(self) -> float:
        """The step size ``tau = t / n``."""
        return self.time / self.steps

    # -- the factor list ---------------------------------------------------------

    def step_factors(self) -> tuple[UnitaryFactor, ...]:
        """The k-local unitary factors of one step, from left to right as written.

        ``result[0]`` is the leftmost factor of the product.  Identity strings are excluded;
        their contribution is reported by `global_phase_per_step`.
        """
        return factors_for(self.layers, self.formula, self.step_size)

    def global_phase_per_step(self) -> float:
        """The angle contributed by identity Pauli strings in one step."""
        return global_phase_for(self.layers, self.formula, self.step_size)

    def support_structure(self) -> tuple[frozenset[int], ...]:
        """The support of each factor of one step, in order."""
        return tuple(factor.support for factor in self.step_factors())

    # -- resources ---------------------------------------------------------------

    def resources(self, estimator: NormEstimator | None = None) -> ResourceSummary:
        """The complete resource summary; no operator is constructed."""
        factors = self.step_factors()
        histogram: dict[int, int] = {}
        for factor in factors:
            histogram[factor.locality] = histogram.get(factor.locality, 0) + self.steps
        bound = self.error_bound(estimator)
        depth_per_step = layer_depth_of(factors)
        return ResourceSummary(
            qubits=self.qubit_count,
            steps=self.steps,
            order=self.order,
            time=self.time,
            factors_per_step=len(factors),
            factor_count=len(factors) * self.steps,
            factors_by_locality=dict(sorted(histogram.items())),
            depth_per_step=depth_per_step,
            depth_in_layers=depth_per_step * self.steps,
            hamiltonian_layers=len(self.layers),
            stages_per_step=self.formula.stage_count,
            global_phase_per_step=self.global_phase_per_step(),
            error_bound=bound.value,
            error_bound_method=bound.method,
            exact_norms=bound.exact_norms,
        )

    def error_bound(self, estimator: NormEstimator | None = None) -> ErrorBound:
        """The a-priori error bound at the ``(t, n, order)`` of this artifact."""
        return error_bound(self.layers, self.formula, self.time, self.steps, estimator)

    def with_steps(self, steps: int) -> ProductFormulaModel:
        """A copy with a different step count."""
        return replace(self, steps=steps)

    def with_schedule(self, schedule: ProductFormulaSchedule) -> ProductFormulaModel:
        """A copy with a different product formula, that is, a different order."""
        return replace(self, formula=schedule)

    # -- numerical realisation ---------------------------------------------------

    def step_matrix(self) -> Array:
        """The unitary of one step, as a dense array.

        Each factor is ``cos(angle) I - i sin(angle) P``, since a Pauli string squares to
        the identity.
        """
        require_x64()
        # Deferred: only the numerical path needs the realisation layer.
        from qsimod.realise.build import build_pauli_string  # noqa: PLC0415

        qubits = self.qubit_count
        dimension = 2**qubits
        total = jnp.eye(dimension, dtype=COMPLEX_DTYPE)
        for factor in self.step_factors():
            string = build_pauli_string(factor.string, qubits)
            gate = (
                jnp.cos(factor.angle) * jnp.eye(dimension, dtype=COMPLEX_DTYPE)
                - 1j * jnp.sin(factor.angle) * string
            )
            total = total @ gate
        phase = jnp.exp(-1j * self.global_phase_per_step())
        return phase * total

    def matrix(self) -> Array:
        """The full propagator ``S_p(t/n)**n``, by binary exponentiation of the step unitary."""
        return _matrix_power(self.step_matrix(), self.steps)

    def apply_to_state(self, state: Array) -> Array:
        """Apply the product formula to a state vector.

        Each factor is applied as ``cos(angle)|psi> - i sin(angle) P|psi>`` at cost
        ``O(k * dimension)``; the ``n`` steps are executed under `jax.lax.fori_loop`.  The
        rightmost factor acts first, as in the written product.
        """
        require_x64()
        qubits = self.qubit_count
        factors = tuple(reversed(self.step_factors()))
        shape = (2,) * qubits
        phase = jnp.exp(-1j * self.global_phase_per_step())

        def one_step(_: Array, current: Array) -> Array:
            nested = current.reshape(shape)
            for factor in factors:
                nested = _apply_factor(nested, factor)
            return phase * nested.reshape(-1)

        evolved: Array = jax.lax.fori_loop(0, self.steps, one_step, state.astype(COMPLEX_DTYPE))
        return evolved

    def __str__(self) -> str:
        return (
            f"{self.name} <{self.kind}> {self.qubit_count} qubits, order {self.order}, "
            f"n={self.steps}, t={self.time:g}, {len(self.step_factors())} factors/step"
        )


def as_product_formula(
    artifact: Artifact,
    subject: str = "this operation",
) -> ProductFormulaModel:
    """Narrow an artifact to a product formula, verifying its kind.

    Args:
        artifact: the artifact to narrow.
        subject: the name of the operation reported in the diagnostic.

    Returns:
        The artifact, typed as a product formula.

    Raises:
        ArtifactKindError: if the artifact is not a product formula.

    """
    if isinstance(artifact, ProductFormulaModel):
        return artifact
    raise ArtifactKindError(subject, ArtifactKind.PRODUCT_FORMULA, artifact.kind, artifact.name)


#: The single-qubit Pauli matrices in the occupation-ordered basis of
#: [`qsimod.realise.hilbert`][qsimod.realise.hilbert], with ``Z = diag(-1, +1)``.
_PAULI_MATRICES: dict[PauliAxis, tuple[tuple[complex, complex], tuple[complex, complex]]] = {
    PauliAxis.X: ((0.0, 1.0), (1.0, 0.0)),
    PauliAxis.Y: ((0.0, 1.0j), (-1.0j, 0.0)),
    PauliAxis.Z: ((-1.0, 0.0), (0.0, 1.0)),
}


def _apply_factor(nested: Array, factor: UnitaryFactor) -> Array:
    """Apply ``cos(angle) I - i sin(angle) P`` to a state reshaped to ``(2,) * qubits``.

    ``P|psi>`` is formed by contracting one single-qubit Pauli matrix into each axis on which
    the string acts, at cost ``O(k * dimension)``.
    """
    rotated = nested
    for qubit, axis in factor.string.axes:
        matrix = jnp.asarray(_PAULI_MATRICES[axis], dtype=COMPLEX_DTYPE)
        rotated = jnp.moveaxis(jnp.tensordot(matrix, rotated, axes=([1], [qubit])), 0, qubit)
    return jnp.cos(factor.angle) * nested - 1j * jnp.sin(factor.angle) * rotated


def _matrix_power(matrix: Array, exponent: int) -> Array:
    """The power ``matrix ** exponent``, by binary exponentiation."""
    result = jnp.eye(matrix.shape[0], dtype=COMPLEX_DTYPE)
    base = matrix
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = result @ base
        remaining >>= 1
        if remaining:
            base = base @ base
    return result
