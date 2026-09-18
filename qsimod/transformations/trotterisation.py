"""Trotterisation: a Hamiltonian is mapped to an ordered product of k-local unitaries.

This is the only transformation that changes the artifact kind, from a Hamiltonian model to a
product formula, and the only one whose approximation error is controlled by a resource
parameter: the error decreases as the step count grows.  The layer decomposition is derived
from the supports of the terms (on-site terms, even-link terms, odd-link terms); whether the
layers commute is computed as an exact Pauli sum and is not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from qsimod.artifact import Artifact, ArtifactKind, HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models import names
from qsimod.models.intermediate import QUBIT_CHAIN_PATTERN
from qsimod.parameters import Namespace
from qsimod.pauli import pauli_sum_from_operator
from qsimod.structure import StructureType
from qsimod.symbolic import OperatorSum, Term
from qsimod.transform import Approximation, ApproximationKind, Exactness
from qsimod.transformations.base import NamespacedTransformation
from qsimod.transformations.basis_changes import carried_over
from qsimod.trotter import (
    LayerDecomposition,
    ProductFormulaModel,
    declared_layers,
    derive_layers,
    suzuki,
)

__all__ = [
    "INTERLEAVED_LAYER_NAMES",
    "SuzukiTrotter",
    "derived_layers_of",
    "interleaved_layers_of",
    "product_formula_from",
    "suzuki_trotter",
]

#: The layer names, in product-formula order, of a Hamiltonian on an interleaved register:
#: on-site terms, even-link terms, odd-link terms.  The order enters the a-priori error bound.
INTERLEAVED_LAYER_NAMES = ("H_M", "H_even", "H_odd")


def _layer_of(term: Term) -> str:
    """The interleaved layer a term belongs to, determined by its support."""
    links = sorted(site for site in term.support if site % 2 == 1)
    if not links:
        return INTERLEAVED_LAYER_NAMES[0]
    return INTERLEAVED_LAYER_NAMES[1] if (links[0] // 2) % 2 == 0 else INTERLEAVED_LAYER_NAMES[2]


def interleaved_layers_of(model: HamiltonianModel) -> LayerDecomposition:
    """The ``(on-site, even links, odd links)`` partition of a bound interleaved qubit model.

    Each layer is internally commuting: the on-site terms trivially, and the link layers
    because their terms have pairwise disjoint supports.

    Args:
        model: a bound qubit Hamiltonian.

    Returns:
        The declared decomposition.

    Raises:
        ValueError: if the model has free parameters.

    """
    environment = model.environment()
    grouped: dict[str, list[Term]] = {name: [] for name in INTERLEAVED_LAYER_NAMES}
    for term in model.hamiltonian.terms:
        grouped[_layer_of(term)].append(term)
    return declared_layers(
        [
            (
                layer,
                pauli_sum_from_operator(
                    OperatorSum(tuple(grouped[layer]), layer),
                    environment,
                    model.structure,
                    layer,
                ),
            )
            for layer in INTERLEAVED_LAYER_NAMES
            if grouped[layer]
        ]
    )


def derived_layers_of(model: HamiltonianModel) -> LayerDecomposition:
    """The greedy first-fit partition of a bound model, derived from its Pauli terms.

    Raises:
        ValueError: if the model has free parameters.

    """
    pauli = pauli_sum_from_operator(
        model.hamiltonian, model.environment(), model.structure, model.name
    )
    return derive_layers(pauli, name_prefix="derived")


def product_formula_from(
    model: HamiltonianModel,
    *,
    time: float,
    steps: int,
    order: int,
    layers: LayerDecomposition | None = None,
    name: str = "U_Trotter",
    structure: StructureType | None = None,
) -> ProductFormulaModel:
    """Build a product-formula artifact from a bound qubit Hamiltonian.

    Args:
        model: the bound Hamiltonian.
        time: the simulated time ``t``.
        steps: the step count ``n``.
        order: the order of the product formula: 1, 2, or any even ``2k >= 4``.
        layers: the layer decomposition; defaults to
            [`interleaved_layers_of`][qsimod.transformations.trotterisation.interleaved_layers_of].
        name: the name of the artifact.
        structure: the structural type of the target; defaults to that of the source.

    Returns:
        The product formula, at the hardware layer.

    """
    decomposition = layers if layers is not None else interleaved_layers_of(model)
    return ProductFormulaModel(
        name=name,
        structure=structure or model.structure,
        level=AbstractionLevel.HARDWARE,
        parameters=model.parameters,
        binding=model.binding,
        origin=(
            f"Suzuki-Trotter order {order} over the {decomposition.provenance} layer "
            f"partition of {model.name}"
        ),
        layers=decomposition,
        formula=suzuki(order, len(decomposition)),
        time=time,
        steps=steps,
    )


@dataclass(frozen=True)
class SuzukiTrotter(NamespacedTransformation):
    """A Suzuki-Trotter product formula over the layer decomposition of a Hamiltonian.

    The resource settings are fields of the transformation; the choice of the step count for a
    requested accuracy is the discrete solve of
    [`qsimod.solving.stepcount`][qsimod.solving.stepcount].  In the case study this is
    the step from ``H_IR4`` to ``U_approx``.

    Attributes:
        time: the simulated time ``t``.
        steps: the step count ``n``.
        order: the order of the product formula.
        derive_partition: whether the greedy derived partition is used instead of the
            interleaved one.

    """

    time: float = 1.0
    steps: int = 1
    order: int = 2
    derive_partition: bool = False

    def target_structure(self, source: StructureType) -> StructureType:
        """The register is unchanged; only the artifact kind changes."""
        return source

    def _transform(self, artifact: Artifact) -> Artifact:
        """Build the product formula from a bound Hamiltonian.

        Raises:
            TypeError: if the source is not a Hamiltonian model.
            ValueError: if the source has free parameters.

        """
        if not isinstance(artifact, HamiltonianModel):
            msg = f"{self.name} needs a Hamiltonian model, got {type(artifact).__name__}"
            raise TypeError(msg)
        layers = (
            derived_layers_of(artifact)
            if self.derive_partition
            else interleaved_layers_of(artifact)
        )
        return product_formula_from(
            artifact,
            time=self.time,
            steps=self.steps,
            order=self.order,
            layers=layers,
            name=self.target_name or "U_Trotter",
        )

    def retimed(
        self,
        *,
        time: float | None = None,
        steps: int | None = None,
        order: int | None = None,
    ) -> SuzukiTrotter:
        """A copy with different resource settings."""
        return replace(
            self,
            time=self.time if time is None else time,
            steps=self.steps if steps is None else steps,
            order=self.order if order is None else order,
        )


def suzuki_trotter(
    source: Namespace,
    target: Namespace,
    *,
    time: float = 1.0,
    steps: int = 1,
    order: int = 2,
    derive_partition: bool = False,
    target_name: str = "U_Trotter",
    name: str = "",
) -> SuzukiTrotter:
    """Build a Suzuki-Trotter transformation at the given resource settings.

    Args:
        source: the namespace of the qubit Hamiltonian.
        target: the namespace of the product formula.
        time: the simulated time.
        steps: the step count.
        order: the order of the product formula.
        derive_partition: whether the greedy derived layer partition is used.
        target_name: the name of the artifact produced.
        name: the name of the transformation; by default the name states the order.

    Returns:
        The transformation.

    """
    return SuzukiTrotter(
        name=name or f"Suzuki-Trotter product formula, order {order}",
        source_pattern=QUBIT_CHAIN_PATTERN,
        target_pattern=QUBIT_CHAIN_PATTERN,
        exactness=Exactness.APPROXIMATE,
        source_kind=ArtifactKind.HAMILTONIAN,
        target_kind=ArtifactKind.PRODUCT_FORMULA,
        approximation=Approximation(
            kind=ApproximationKind.RESOURCE_CONTROLLED,
            leading_error_order="O(t^(p+1) / n^p) at order p; see qsimod.trotter.bounds",
            resource_parameters=("steps", "order"),
            note=(
                "no validity window; the error falls as n grows, at a cost in k-local unitary "
                "factors and depth"
            ),
        ),
        relation=carried_over(source, target, names.MASS, names.COUPLING),
        source_parameters=(source(names.MASS), source(names.COUPLING)),
        target_parameters=(target(names.MASS), target(names.COUPLING)),
        source_level=AbstractionLevel.INTERMEDIATE,
        target_level=AbstractionLevel.HARDWARE,
        description=(
            "the Hamiltonian is split into internally commuting layers and exponentiated by "
            "the product formula of the requested order; each layer factorises exactly into "
            "one k-local rotation per Pauli string"
        ),
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
        time=time,
        steps=steps,
        order=order,
        derive_partition=derive_partition,
    )
