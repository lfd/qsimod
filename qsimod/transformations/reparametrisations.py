"""Exact reparametrisations: the same Hamiltonian stated by different parameters.

An application model is stated by an energy scale and dimensionless parameters; the
derivation from a hardware model produces energies.  The relation is ``CLOSED_FORM`` in both
directions, that is invertible: one equation carries the energy scale across, and one
equation per dimensionless parameter states that the target energy is the product of that
parameter and the scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qsimod.artifact import HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models import names
from qsimod.models.application import HEISENBERG_MAGNET_PATTERN, ISING_MAGNET_PATTERN
from qsimod.models.intermediate import (
    ISING_CHAIN_PATTERN,
    SPIN_HALF_CHAIN_PATTERN,
    ising_spin_chain,
    xxz_spin_chain,
)
from qsimod.parameters import Namespace
from qsimod.relations import Definition, ParameterRelation, equation
from qsimod.structure import StructureType
from qsimod.symbolic import OperatorSum
from qsimod.transform import Exactness
from qsimod.transformations.base import ModelTransformation

__all__ = [
    "AnisotropyResolution",
    "FieldResolution",
    "Reparametrisation",
    "anisotropy_resolution",
    "field_resolution",
]


@dataclass(frozen=True)
class Reparametrisation(ModelTransformation):
    """A transformation that restates a model by different parameters and keeps its operators."""

    def target_structure(self, source: StructureType) -> StructureType:
        """The structural type, name included, is unchanged."""
        return source

    def image(self, source: HamiltonianModel) -> OperatorSum:
        """The source Hamiltonian itself, since a reparametrisation changes no operator."""
        return source.hamiltonian


def _dial_relation(
    name: str,
    source: Namespace,
    target: Namespace,
    scale: str,
    dials: Sequence[tuple[str, str, str]],
) -> ParameterRelation:
    """The relation of a reparametrisation by dimensionless parameters, solved in both directions.

    Args:
        name: the name of the relation.
        source: the namespace that states the model by dimensionless parameters.
        target: the namespace that states the model by energies.
        scale: the local name of the energy scale both sides carry.
        dials: one ``(dial, energy, note)`` triple per dimensionless parameter.

    Returns:
        The relation.

    """
    carried = f"{scale} carried over"
    equations = [
        equation(
            carried,
            target.symbol(scale),
            source.symbol(scale),
            "the overall scale is the same number on both sides",
        )
    ]
    definitions = [
        Definition(target(scale), source.symbol(scale), carried),
        Definition(source(scale), target.symbol(scale), carried),
    ]
    for dial, energy, note in dials:
        label = f"{energy} from {dial}"
        product = source.symbol(dial) * source.symbol(scale)
        equations.append(equation(label, target.symbol(energy), product, note))
        definitions += [
            Definition(target(energy), product, label),
            Definition(
                source(dial),
                target.symbol(energy) / target.symbol(scale),
                label,
                note=f"{dial} = {energy} / {scale}, wherever {scale} is non-zero",
            ),
        ]
    return ParameterRelation(name=name, equations=tuple(equations), definitions=tuple(definitions))


@dataclass(frozen=True)
class AnisotropyResolution(Reparametrisation):
    """The resolution of the anisotropy of an XXZ magnet into a second coupling energy.

    ```
    Jxy -> Jxy        (carried over)
    Jz  =  Delta * Jxy
    ```
    """

    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The XXZ chain carrying two coupling energies."""
        return xxz_spin_chain(matter_sites, self.target_namespace, name=self.target_name or "H_XXZ")


def anisotropy_resolution(
    source: Namespace,
    target: Namespace,
    *,
    target_name: str = "H_XXZ",
    name: str = "anisotropy resolution",
) -> AnisotropyResolution:
    """Build an anisotropy resolution between two namespaces.

    Args:
        source: the namespace of the magnet as stated, with ``Jxy`` and ``Delta``.
        target: the namespace of the two-coupling form, with ``Jxy`` and ``Jz``.
        target_name: the name of the model produced.
        name: the name of the transformation.

    Returns:
        The transformation.

    """
    return AnisotropyResolution(
        name=name,
        source_pattern=HEISENBERG_MAGNET_PATTERN,
        target_pattern=SPIN_HALF_CHAIN_PATTERN,
        exactness=Exactness.EXACT,
        relation=_dial_relation(
            "anisotropy resolution",
            source,
            target,
            names.TRANSVERSE_COUPLING,
            [
                (
                    names.ANISOTROPY,
                    names.LONGITUDINAL_COUPLING,
                    "Jz = Delta * Jxy: the anisotropy is the ratio of the two couplings",
                )
            ],
        ),
        source_parameters=(source(names.TRANSVERSE_COUPLING), source(names.ANISOTROPY)),
        target_parameters=(
            target(names.TRANSVERSE_COUPLING),
            target(names.LONGITUDINAL_COUPLING),
        ),
        source_level=AbstractionLevel.APPLICATION,
        target_level=AbstractionLevel.INTERMEDIATE,
        description=(
            "the same spin chain, stated by two coupling energies instead of one energy and "
            "one dimensionless anisotropy; no operator and no algebra changes"
        ),
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
    )


@dataclass(frozen=True)
class FieldResolution(Reparametrisation):
    """The resolution of the two dimensionless fields of an Ising chain into energies.

    ```
    Jz    -> Jz               (carried over)
    Gamma =  hx * Jz
    B     =  hz * Jz
    ```
    """

    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The Ising chain carrying three energies."""
        return ising_spin_chain(
            matter_sites, self.target_namespace, name=self.target_name or "H_Ising"
        )


def field_resolution(
    source: Namespace,
    target: Namespace,
    *,
    target_name: str = "H_Ising",
    name: str = "field resolution",
) -> FieldResolution:
    """Build a field resolution between two namespaces.

    Args:
        source: the namespace of the magnet as stated, with ``Jz``, ``hz`` and ``hx``.
        target: the namespace of the three-energy form, with ``Jz``, ``B`` and ``Gamma``.
        target_name: the name of the model produced.
        name: the name of the transformation.

    Returns:
        The transformation.

    """
    return FieldResolution(
        name=name,
        source_pattern=ISING_MAGNET_PATTERN,
        target_pattern=ISING_CHAIN_PATTERN,
        exactness=Exactness.EXACT,
        relation=_dial_relation(
            "field resolution",
            source,
            target,
            names.LONGITUDINAL_COUPLING,
            [
                (
                    names.LONGITUDINAL_FIELD,
                    names.LONGITUDINAL_BIAS,
                    "B = hz * Jz: the longitudinal field, in units of the coupling",
                ),
                (
                    names.TRANSVERSE_FIELD,
                    names.TRANSVERSE_AMPLITUDE,
                    "Gamma = hx * Jz: the transverse field, in units of the coupling",
                ),
            ],
        ),
        source_parameters=(
            source(names.LONGITUDINAL_COUPLING),
            source(names.LONGITUDINAL_FIELD),
            source(names.TRANSVERSE_FIELD),
        ),
        target_parameters=(
            target(names.LONGITUDINAL_COUPLING),
            target(names.LONGITUDINAL_BIAS),
            target(names.TRANSVERSE_AMPLITUDE),
        ),
        source_level=AbstractionLevel.APPLICATION,
        target_level=AbstractionLevel.INTERMEDIATE,
        description=(
            "the same Ising chain, stated by three coupling energies instead of one energy "
            "and two dimensionless fields; no operator and no algebra changes"
        ),
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
    )
