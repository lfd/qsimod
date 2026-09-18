"""Encodings: the degrees of freedom of one algebra are rewritten in another algebra.

An encoding is exact on the encoded subspace: the operator identities hold as ``A = P B P``,
and the target model declares the projector ``P`` onto the declared occupation subspace.
"""

from __future__ import annotations

from dataclasses import dataclass

from qsimod.artifact import HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models import names
from qsimod.models.intermediate import (
    BOSONIC_CHAIN_PATTERN,
    FERMION_SPIN_CHAIN_PATTERN,
    bosonic_pair_coupling_model,
)
from qsimod.normal_form import constant_word, normal_form, rewrite
from qsimod.parameters import Namespace
from qsimod.scalar import sqrt
from qsimod.structure import Algebra
from qsimod.symbolic import (
    OperatorSum,
    OpSymbol,
    SiteOperator,
    annihilate,
    create,
    number,
    word,
)
from qsimod.transform import Exactness
from qsimod.transformations.base import ModelTransformation
from qsimod.transformations.basis_changes import (
    carried_over,
    fermions_to_qubits_image,
    staggering_constant,
)

__all__ = ["HardcoreBosonEncoding", "hardcore_boson_encoding", "qubits_to_bosons_image"]


def qubits_to_bosons_image(qubit_hamiltonian: OperatorSum, source: HamiltonianModel) -> OperatorSum:
    """The bosonic encoding of a qubit-register Hamiltonian, exact on the encoded subspace.

    ```
    matter (n in {0, 1}):  sigma^- -> a              sigma^+ -> a^dag              Z -> 2n - 1
    link   (n in {0, 2}):  sigma^- -> d^2 / sqrt 2   sigma^+ -> (d^dag)^2 / sqrt 2   Z -> n - 1
    ```

    where ``X = sigma^+ + sigma^-``, ``Y = -i (sigma^+ - sigma^-)`` and ``n = (1 + Z)/2`` on
    the qubit side.  The identities hold as ``P B P``.  The map is linear with one factor per
    site and is therefore applied to a sum already in the normal form of the qubit algebra.
    ``source`` determines which register sites are matter sites (fermions) and which are links
    (spins).
    """
    structure = source.structure
    links = {site for site in structure.sites if structure.algebra_at(site) is Algebra.SPIN_HALF}
    root_two = sqrt(2)

    def raising(site: int) -> OperatorSum:
        if site in links:
            return word(1 / root_two, create(site), create(site))
        return word(1, create(site))

    def lowering(site: int) -> OperatorSum:
        if site in links:
            return word(1 / root_two, annihilate(site), annihilate(site))
        return word(1, annihilate(site))

    def pauli_z_image(site: int) -> OperatorSum:
        if site in links:
            return word(1, number(site)) + constant_word(-1)
        return word(2, number(site)) + constant_word(-1)

    def image(operator: SiteOperator) -> OperatorSum:
        site, symbol = operator.site, operator.symbol
        if symbol is OpSymbol.SIGMA_PLUS:
            return raising(site)
        if symbol is OpSymbol.SIGMA_MINUS:
            return lowering(site)
        if symbol is OpSymbol.PAULI_X:
            return raising(site) + lowering(site)
        if symbol is OpSymbol.PAULI_Y:
            return (-1j) * raising(site) + 1j * lowering(site)
        if symbol is OpSymbol.PAULI_Z:
            return pauli_z_image(site)
        if symbol is OpSymbol.NUMBER:
            return 0.5 * (constant_word(1) + pauli_z_image(site))
        msg = f"{operator} has no bosonic encoding on the {{0,1}} / {{0,2}} subspace"
        raise ValueError(msg)

    return rewrite(qubit_hamiltonian, image)


@dataclass(frozen=True)
class HardcoreBosonEncoding(ModelTransformation):
    """The boson encoding of spin-1/2 degrees of freedom by the two lowest bosonic occupations.

    ```
    sigma^-_l = P_l a_l P_l                 S^- = (1/sqrt(2)) P d^2 P
    sigma^z_l = P_l (2 a^dag_l a_l - 1) P_l  S^z = (1/2) P (d^dag d - 1) P
    ```

    Matter positions carry ``n in {0, 1}`` and link positions ``n in {0, 2}``, so a numerical
    realisation needs an occupation cutoff of at least two.

    Attributes:
        retain_staggering_constant: whether the constant ``-m * floor(N/2)`` carried by the
            source is added to the target.

    """

    retain_staggering_constant: bool = True

    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The encoded bosonic model."""
        return bosonic_pair_coupling_model(
            matter_sites,
            self.target_namespace,
            name=self.target_name or "H_eff",
            additive_constant=(
                staggering_constant(matter_sites, self.target_namespace.symbol(names.MASS))
                if self.retain_staggering_constant
                else None
            ),
        )

    def image(self, source: HamiltonianModel) -> OperatorSum:
        """The Jordan-Wigner image on qubits in normal form, followed by the boson encoding."""
        on_qubits = normal_form(
            fermions_to_qubits_image(source),
            dict.fromkeys(source.structure.sites, Algebra.QUBIT),
        )
        return qubits_to_bosons_image(on_qubits, source)


def hardcore_boson_encoding(
    source: Namespace,
    target: Namespace,
    *,
    retain_staggering_constant: bool = True,
    target_name: str = "H_eff",
    name: str = "Jordan-Wigner + hardcore-boson encoding",
) -> HardcoreBosonEncoding:
    """Build a hardcore-boson encoding between two namespaces.

    Args:
        source: the namespace of the quantum-link model.
        target: the namespace of the bosonic model.
        retain_staggering_constant: whether the source carries the staggering constant.
        target_name: the name of the model produced.
        name: the name of the transformation.

    Returns:
        The transformation.

    """
    return HardcoreBosonEncoding(
        name=name,
        source_pattern=FERMION_SPIN_CHAIN_PATTERN,
        target_pattern=BOSONIC_CHAIN_PATTERN,
        exactness=Exactness.EXACT,
        relation=carried_over(source, target, names.MASS, names.COUPLING),
        source_parameters=(source(names.MASS), source(names.COUPLING)),
        target_parameters=(target(names.MASS), target(names.COUPLING)),
        source_level=AbstractionLevel.INTERMEDIATE,
        target_level=AbstractionLevel.INTERMEDIATE,
        description=(
            "the spins become the two lowest occupations of a boson, matter site l at "
            "register position 2l and link (l, l+1) at 2l+1.  Exact as an identity between "
            "P B P operators; n_max >= 2 is required of any realisation."
        ),
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
        retain_staggering_constant=retain_staggering_constant,
    )
