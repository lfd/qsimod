"""Layer decompositions of a Hamiltonian and the commutation structure between layers.

A layer is a group of Pauli terms that mutually commute; each layer therefore exponentiates
exactly into a product of one rotation per term.  A decomposition is either declared
([`declared_layers`][qsimod.trotter.layers.declared_layers]) or derived by first-fit greedy
grouping ([`derive_layers`][qsimod.trotter.layers.derive_layers]).  Whether two layers commute
is decided by computing their commutator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from qsimod.pauli import PauliAxis, PauliString, PauliSum, commutator

__all__ = [
    "Layer",
    "LayerDecomposition",
    "declared_layers",
    "derive_layers",
]


@dataclass(frozen=True)
class Layer:
    """One group of mutually commuting Pauli terms.

    Attributes:
        name: a label, used in reports and in resource summaries.
        operator: the terms of the layer, as a Pauli sum.

    """

    name: str
    operator: PauliSum

    @property
    def support(self) -> frozenset[int]:
        """The qubits on which the layer acts."""
        return self.operator.support

    @property
    def term_count(self) -> int:
        """The number of Pauli strings in the layer."""
        return len(self.operator)

    @property
    def one_norm(self) -> float:
        """The one-norm ``sum |coefficient|``, an upper bound on the spectral norm of the layer."""
        return self.operator.one_norm

    def is_internally_commuting(self) -> bool:
        """Whether every pair of strings in the layer commutes."""
        return self.operator.internally_commuting()

    def is_diagonal(self) -> bool:
        """Whether every string is a product of ``Z`` operators only."""
        return all(
            all(axis is PauliAxis.Z for _, axis in string.axes) for string in self.operator.terms
        )

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.term_count} string(s), support {sorted(self.support)}, "
            f"one-norm {self.one_norm:.4g}"
        )


@dataclass(frozen=True)
class LayerDecomposition:
    """An ordered partition of a Hamiltonian into layers of mutually commuting terms.

    Attributes:
        layers: the layers, in the order in which a product formula visits them.
        provenance: ``"declared"`` or ``"derived"``.

    """

    layers: tuple[Layer, ...]
    provenance: str = "declared"

    def __post_init__(self) -> None:
        if not self.layers:
            msg = "a layer decomposition needs at least one layer"
            raise ValueError(msg)
        offenders = [layer.name for layer in self.layers if not layer.is_internally_commuting()]
        if offenders:
            msg = (
                f"layer(s) {offenders} are not internally commuting, so they do not "
                "exponentiate exactly as a product of one rotation per term"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def names(self) -> tuple[str, ...]:
        """The layer names, in order."""
        return tuple(layer.name for layer in self.layers)

    def total(self) -> PauliSum:
        """The Hamiltonian that the decomposition partitions, the sum of all layers."""
        total = PauliSum.zero("H")
        for layer in self.layers:
            total = total + layer.operator
        return total

    @property
    def support(self) -> frozenset[int]:
        """The union of the supports of all layers."""
        return frozenset().union(*(layer.support for layer in self.layers))

    @property
    def qubit_count(self) -> int:
        """The register size implied by the support, ``max qubit index + 1``."""
        return max(self.support) + 1 if self.support else 0

    @property
    def one_norm(self) -> float:
        """The sum ``sum_gamma ||H_gamma||_1`` of layer one-norms, an upper bound on ``||H||``."""
        return sum(layer.one_norm for layer in self.layers)

    # -- commutation structure ---------------------------------------------------

    def layer_commutator(self, first: int, second: int) -> PauliSum:
        """The commutator ``[H_first, H_second]`` as an exact Pauli sum."""
        return commutator(self.layers[first].operator, self.layers[second].operator)

    def commuting_pairs(self) -> frozenset[tuple[int, int]]:
        """The index pairs ``(i, j)``, ``i < j``, whose layers commute exactly."""
        return frozenset(
            (i, j)
            for i, j in combinations(range(len(self.layers)), 2)
            if self.layer_commutator(i, j).is_zero
        )

    def non_commuting_pairs(self) -> tuple[tuple[int, int], ...]:
        """The index pairs ``(i, j)``, ``i < j``, whose layers do not commute, sorted."""
        commuting = self.commuting_pairs()
        return tuple(
            pair for pair in combinations(range(len(self.layers)), 2) if pair not in commuting
        )

    def commutation_report(self) -> str:
        """A textual table stating which pairs of layers commute."""
        lines = [f"layer commutation ({self.provenance} partition, {len(self)} layers):"]
        for i, j in combinations(range(len(self.layers)), 2):
            bracket = self.layer_commutator(i, j)
            verdict = "commute" if bracket.is_zero else f"do NOT commute ({len(bracket)} strings)"
            lines.append(f"  [{self.layers[i].name}, {self.layers[j].name}]: {verdict}")
        return "\n".join(lines)

    def layer_named(self, name: str) -> Layer:
        """The layer with the given name.

        Raises:
            KeyError: if no layer has that name.

        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        msg = f"no layer named {name!r}; have {self.names}"
        raise KeyError(msg)

    def __str__(self) -> str:
        lines = [f"layer decomposition ({self.provenance}), {len(self)} layer(s):"]
        lines += [f"  {layer}" for layer in self.layers]
        commuting = self.commuting_pairs()
        if commuting:
            pairs = ", ".join(f"[{self.names[i]},{self.names[j]}]" for i, j in sorted(commuting))
            lines.append(f"  commuting pairs: {pairs}")
        else:
            lines.append("  commuting pairs: none -- every pair contributes to the error bound")
        return "\n".join(lines)


def declared_layers(groups: Sequence[tuple[str, PauliSum]]) -> LayerDecomposition:
    """Construct a decomposition from an explicitly declared, ordered partition.

    Raises:
        ValueError: if any group is not internally commuting.

    """
    return LayerDecomposition(
        layers=tuple(Layer(name, operator) for name, operator in groups),
        provenance="declared",
    )


def derive_layers(hamiltonian: PauliSum, name_prefix: str = "layer") -> LayerDecomposition:
    """Partition a Pauli sum into layers of mutually commuting terms by a greedy algorithm.

    The strings are visited in `PauliString.sort_key` order and placed by first fit: each string
    joins the first existing layer with all of whose strings it commutes, or opens a new layer.

    Args:
        hamiltonian: the Pauli sum to partition.
        name_prefix: the stem for the generated layer names.

    Returns:
        The decomposition, with provenance ``"derived"``.

    Raises:
        ValueError: if ``hamiltonian`` is empty.

    """
    if hamiltonian.is_zero:
        msg = "cannot derive a layer decomposition of the zero operator"
        raise ValueError(msg)
    buckets: list[list[tuple[PauliString, complex]]] = []
    for string in hamiltonian.strings:
        coefficient = hamiltonian.terms[string]
        placed = False
        for bucket in buckets:
            if all(string.commutes_with(other) for other, _ in bucket):
                bucket.append((string, coefficient))
                placed = True
                break
        if not placed:
            buckets.append([(string, coefficient)])
    layers = tuple(
        Layer(
            name=f"{name_prefix}{index}",
            operator=PauliSum.from_terms(bucket, f"{name_prefix}{index}"),
        )
        for index, bucket in enumerate(buckets)
    )
    return LayerDecomposition(layers=layers, provenance="derived")
