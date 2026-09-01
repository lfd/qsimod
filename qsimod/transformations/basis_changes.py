"""Exact basis changes: the particle-hole transformation and two Jordan-Wigner transformations.

Each rewrites a Hamiltonian in different operators; the operator form of the target is
unitarily equivalent to that of the source, and the spectrum is preserved.  None of them
changes the abstraction level.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qsimod.artifact import HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models import names
from qsimod.models.intermediate import (
    FERMION_CHAIN_PATTERN,
    FERMION_SPIN_CHAIN_PATTERN,
    HOMOGENEOUS_CONVENTION,
    QUBIT_CHAIN_PATTERN,
    SPIN_HALF_CHAIN_PATTERN,
    QuantumLinkConvention,
    interacting_fermion_chain,
    k_local_qubit_model,
    quantum_link_model,
)
from qsimod.normal_form import constant_word, product, rewrite
from qsimod.parameters import Namespace
from qsimod.relations import ParameterRelation, identity_relation
from qsimod.scalar import Scalar
from qsimod.structure import Algebra, StructureType
from qsimod.symbolic import (
    OperatorSum,
    OpSymbol,
    SiteOperator,
    annihilate,
    create,
    number,
    parity,
    pauli_z,
    sigma_minus,
    sigma_plus,
    spin_minus,
    spin_plus,
    spin_z,
    word,
)
from qsimod.transform import Exactness
from qsimod.transformations.base import ModelTransformation

__all__ = [
    "JordanWignerToFermions",
    "JordanWignerToQubits",
    "ParticleHoleTransformation",
    "carried_over",
    "fermions_to_qubits_image",
    "jordan_wigner_ladder",
    "jordan_wigner_to_fermions",
    "jordan_wigner_to_qubits",
    "parity_string",
    "particle_hole_transformation",
    "staggering_constant",
]


def carried_over(source: Namespace, target: Namespace, *locals_: str) -> ParameterRelation:
    """An identity relation on the named local parameters between two namespaces."""
    return identity_relation(
        [(source(local), target(local)) for local in locals_],
        name="carried over",
    )


def staggering_constant(matter_sites: int, mass: Scalar) -> Scalar:
    """The constant ``-m * floor(N/2)`` that ``n_l -> 1 - n_l`` on the odd sites leaves behind."""
    return -mass * (matter_sites // 2)


def parity_string(sites: Sequence[int]) -> OperatorSum:
    """The Jordan-Wigner string ``prod_q P_q``, ``P = 1 - 2n``, over the given register sites.

    The result is one word of ``len(sites)`` parity factors; the normal form expands a parity
    only where a site retains an odd number of them.
    """
    return word(1, *(parity(site) for site in sites))


def jordan_wigner_ladder(
    position: int,
    string_sites: Sequence[int],
    ladder: SiteOperator,
) -> OperatorSum:
    """A fermionic ladder operator at Jordan-Wigner position ``p`` as ``(-1)**p (prod P) ladder``.

    The convention is that of [`qsimod.realise.build`][qsimod.realise.build]: the plain
    Jordan-Wigner transformation with alternating minus signs, the string running over the
    sites with a strictly smaller position.  ``ladder`` is the two-level operator that
    represents ``psi`` or ``psi^dag`` in the target algebra, that is ``sigma^-+`` on a qubit,
    or, in the inverse direction, ``c`` or ``c^dag`` representing a spin ``S^-+``.
    """
    return product(parity_string(string_sites), word((-1) ** position, ladder))


def _matter_sites_of(model: HamiltonianModel) -> tuple[int, ...]:
    """The fermionic register sites of a model in ascending order, the Jordan-Wigner order."""
    return tuple(
        site
        for site in model.structure.sites
        if model.structure.algebra_at(site) is Algebra.FERMION
    )


def fermions_to_qubits_image(source: HamiltonianModel) -> OperatorSum:
    """The Jordan-Wigner image of a fermion/spin-1/2 model on a qubit register.

    ```
    psi_l -> (-1)**p prod_{q<p} (1 - 2 n_q) sigma^-_l      n_l -> n_l
    S^+- -> sigma^+-                                        S^z -> Z / 2
    ```

    where ``p`` is the position of the fermionic site ``l`` in ascending site order.  The
    register positions are unchanged.
    """
    fermions = _matter_sites_of(source)
    position_of = {site: index for index, site in enumerate(fermions)}

    def image(operator: SiteOperator) -> OperatorSum:
        site, symbol = operator.site, operator.symbol
        if site in position_of and symbol in {OpSymbol.CREATE, OpSymbol.ANNIHILATE}:
            ladder = sigma_plus(site) if symbol is OpSymbol.CREATE else sigma_minus(site)
            return jordan_wigner_ladder(position_of[site], fermions[: position_of[site]], ladder)
        if symbol is OpSymbol.SPIN_Z:
            return word(0.5, pauli_z(site))
        if symbol is OpSymbol.SPIN_PLUS:
            return word(1, sigma_plus(site))
        if symbol is OpSymbol.SPIN_MINUS:
            return word(1, sigma_minus(site))
        return word(1, operator)

    return rewrite(source.hamiltonian, image)


@dataclass(frozen=True)
class ParticleHoleTransformation(ModelTransformation):
    """The particle-hole transformation on the odd matter sites and the even links.

    ```
    on matter sites l = 1, 3, 5, ... (odd):     n_l -> 1 - n_l,  psi_l -> psi^dag_l
    on links (l, l+1) with l = 0, 2, ... (even): S^z -> -S^z,    S^+- -> S^-+
    ```

    The transformation is exact and unitary.  The hopping coupling becomes the pair coupling
    with a sign ``(-1)**(l+1)`` on link ``l``; the substitution ``S^+- -> -S^+-`` on the even
    links, a rotation about ``z`` that leaves ``S^z`` and every ``G_l`` unchanged, removes this
    sign.  The mass staggering cancels and leaves the constant of
    [`staggering_constant`][qsimod.transformations.basis_changes.staggering_constant], which is
    retained.  The two parities must be opposite: odd matter sites with even links pass ``+m``
    to the target, even matter sites with odd links ``-m``.  In the case study this is
    transformation (b), from ``H_IR1`` to ``H_IR2``.

    Attributes:
        convention: the quantum-link convention produced, which is the homogeneous convention.

    """

    convention: QuantumLinkConvention = HOMOGENEOUS_CONVENTION

    def target_structure(self, source: StructureType) -> StructureType:
        """The structural type, name included, is unchanged."""
        return source

    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The homogeneous quantum-link model, carrying the staggering constant."""
        return quantum_link_model(
            matter_sites,
            self.target_namespace,
            self.convention,
            name=self.target_name or "H_QLM",
            additive_constant=staggering_constant(
                matter_sites, self.target_namespace.symbol(names.MASS)
            ),
        )

    def image(self, source: HamiltonianModel) -> OperatorSum:
        """The source rewritten under the particle-hole transformation, on the same register.

        On the odd matter sites ``n -> 1 - n`` and ``psi <-> psi^dag``; on the even links
        ``S^z -> -S^z`` and ``S^+- -> -S^-+``.  The latter includes the ``z`` rotation that
        removes the ``(-1)**(l+1)`` sign of the pair coupling.
        """
        structure = source.structure
        matter = _matter_sites_of(source)
        matter_index = {site: index for index, site in enumerate(matter)}
        links = tuple(
            site for site in structure.sites if structure.algebra_at(site) is Algebra.SPIN_HALF
        )
        link_index = {site: index for index, site in enumerate(links)}

        def image(operator: SiteOperator) -> OperatorSum:
            site, symbol = operator.site, operator.symbol
            if site in matter_index and matter_index[site] % 2 == 1:
                if symbol is OpSymbol.NUMBER:
                    return constant_word(1) + word(-1, number(site))
                if symbol is OpSymbol.CREATE:
                    return word(1, annihilate(site))
                if symbol is OpSymbol.ANNIHILATE:
                    return word(1, create(site))
            if site in link_index and link_index[site] % 2 == 0:
                if symbol is OpSymbol.SPIN_Z:
                    return word(-1, spin_z(site))
                if symbol is OpSymbol.SPIN_PLUS:
                    return word(-1, spin_minus(site))
                if symbol is OpSymbol.SPIN_MINUS:
                    return word(-1, spin_plus(site))
            return word(1, operator)

        return rewrite(source.hamiltonian, image)


def particle_hole_transformation(
    source: Namespace,
    target: Namespace,
    *,
    target_name: str = "H_QLM",
    name: str = "particle-hole transformation",
) -> ParticleHoleTransformation:
    """Build a particle-hole transformation between two namespaces.

    Args:
        source: the namespace of the staggered model.
        target: the namespace of the homogeneous model.
        target_name: the name of the model produced.
        name: the name of the transformation.

    Returns:
        The transformation.

    """
    return ParticleHoleTransformation(
        name=name,
        source_pattern=FERMION_SPIN_CHAIN_PATTERN,
        target_pattern=FERMION_SPIN_CHAIN_PATTERN,
        exactness=Exactness.EXACT,
        relation=carried_over(source, target, names.MASS, names.COUPLING),
        source_parameters=(source(names.MASS), source(names.COUPLING)),
        target_parameters=(target(names.MASS), target(names.COUPLING)),
        source_level=AbstractionLevel.INTERMEDIATE,
        target_level=AbstractionLevel.INTERMEDIATE,
        description=(
            "n_l -> 1 - n_l and psi_l -> psi^dag_l on the odd matter sites; S^z -> -S^z and "
            "S^+- -> S^-+ on the even links.  The hopping term becomes a pair "
            "creation/annihilation term and the mass staggering cancels."
        ),
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
    )


@dataclass(frozen=True)
class JordanWignerToQubits(ModelTransformation):
    """The Jordan-Wigner transformation of the matter fermions of a quantum-link model to qubits.

    ```
    psi^dag_l psi_l  ->  (sigma^z_{2l} + 1) / 2      S^+-_{l,l+1}  ->  sigma^+-_{2l+1}
    psi_l psi_{l+1}  ->  sigma^-_{2l} sigma^-_{2l+2}  S^z_{l,l+1}   ->  sigma^z_{2l+1} / 2
    ```

    The transformation is exact.  Matter sites and links are placed on one interleaved qubit
    register.  The Jordan-Wigner string runs over the matter sites only, and the coupling joins
    adjacent matter sites, so the string reduces to a sign: ``+1`` for the pair coupling and
    ``-1`` for the hopping coupling under the alternating-sign convention of
    [`qsimod.realise.build`][qsimod.realise.build].  A coupling of longer range would leave a
    string behind.  In the case study this is transformation (e), from ``H_IR2`` to ``H_IR4``.

    Attributes:
        convention: the quantum-link convention of the source.
        retain_staggering_constant: whether the constant ``-m * floor(N/2)`` carried by the
            source is added to the target.

    """

    convention: QuantumLinkConvention = HOMOGENEOUS_CONVENTION
    retain_staggering_constant: bool = True

    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The image on the qubit register."""
        return k_local_qubit_model(
            matter_sites,
            self.target_namespace,
            self.convention,
            name=self.target_name or "H_qubit",
            additive_constant=(
                staggering_constant(matter_sites, self.target_namespace.symbol(names.MASS))
                if self.retain_staggering_constant
                else None
            ),
        )

    def image(self, source: HamiltonianModel) -> OperatorSum:
        """The Jordan-Wigner image of the source on the qubit register."""
        return fermions_to_qubits_image(source)


def jordan_wigner_to_qubits(
    source: Namespace,
    target: Namespace,
    *,
    convention: QuantumLinkConvention = HOMOGENEOUS_CONVENTION,
    retain_staggering_constant: bool = True,
    target_name: str = "H_qubit",
    name: str = "Jordan-Wigner to qubits",
) -> JordanWignerToQubits:
    """Build a Jordan-Wigner transformation to an interleaved qubit register.

    Args:
        source: the namespace of the fermion-spin model.
        target: the namespace of the qubit model.
        convention: the quantum-link convention of the source.
        retain_staggering_constant: whether the source carries the staggering constant.
        target_name: the name of the model produced.
        name: the name of the transformation.

    Returns:
        The transformation.

    """
    return JordanWignerToQubits(
        name=name,
        source_pattern=FERMION_SPIN_CHAIN_PATTERN,
        target_pattern=QUBIT_CHAIN_PATTERN,
        exactness=Exactness.EXACT,
        relation=carried_over(source, target, names.MASS, names.COUPLING),
        source_parameters=(source(names.MASS), source(names.COUPLING)),
        target_parameters=(target(names.MASS), target(names.COUPLING)),
        source_level=AbstractionLevel.INTERMEDIATE,
        target_level=AbstractionLevel.INTERMEDIATE,
        description=(
            "the matter fermions map to the even register positions and the link spins to the "
            "odd ones, on 2N-1 qubits; the Jordan-Wigner string is empty because the coupling "
            "is between adjacent matter sites"
        ),
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
        convention=convention,
        retain_staggering_constant=retain_staggering_constant,
    )


@dataclass(frozen=True)
class JordanWignerToFermions(ModelTransformation):
    """The Jordan-Wigner transformation of a spin-1/2 chain to spinless fermions.

    ```
    S^+_j -> c^dag_j prod_{k<j} (1 - 2 n_k)        S^z_j -> n_j - 1/2
    ```

    The transformation is exact.  The transverse term becomes a hopping term, and the ``ZZ``
    term becomes a nearest-neighbour interaction, a chemical potential and a constant; the
    strings cancel because the couplings join adjacent sites.
    """

    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The interacting spinless-fermion chain."""
        return interacting_fermion_chain(
            matter_sites, self.target_namespace, name=self.target_name or "H_tV"
        )

    def image(self, source: HamiltonianModel) -> OperatorSum:
        """The source spin chain rewritten in fermions, on the same sites.

        ```
        S^+_j -> (-1)**j prod_{k<j} (1 - 2 n_k) c^dag_j      S^z_j -> n_j - 1/2
        ```

        The map is the inverse of the Jordan-Wigner convention of
        [`qsimod.realise.build`][qsimod.realise.build], with every site fermionic.
        """
        sites = tuple(source.structure.sites)
        position_of = {site: index for index, site in enumerate(sites)}

        def image(operator: SiteOperator) -> OperatorSum:
            site, symbol = operator.site, operator.symbol
            if symbol is OpSymbol.SPIN_Z:
                return word(1, number(site)) + constant_word(-0.5)
            if symbol in {OpSymbol.SPIN_PLUS, OpSymbol.SPIN_MINUS}:
                ladder = create(site) if symbol is OpSymbol.SPIN_PLUS else annihilate(site)
                return jordan_wigner_ladder(position_of[site], sites[: position_of[site]], ladder)
            return word(1, operator)

        return rewrite(source.hamiltonian, image)


def jordan_wigner_to_fermions(
    source: Namespace,
    target: Namespace,
    *,
    target_name: str = "H_tV",
    name: str = "Jordan-Wigner to spinless fermions",
) -> JordanWignerToFermions:
    """Build a Jordan-Wigner transformation from a spin-1/2 chain to spinless fermions.

    Args:
        source: the namespace of the spin chain, which carries ``Jxy`` and ``Jz``.
        target: the namespace of the fermion chain, which carries the same two parameters.
        target_name: the name of the model produced.
        name: the name of the transformation.

    Returns:
        The transformation.

    """
    return JordanWignerToFermions(
        name=name,
        source_pattern=SPIN_HALF_CHAIN_PATTERN,
        target_pattern=FERMION_CHAIN_PATTERN,
        exactness=Exactness.EXACT,
        relation=carried_over(
            source, target, names.TRANSVERSE_COUPLING, names.LONGITUDINAL_COUPLING
        ),
        source_parameters=(
            source(names.TRANSVERSE_COUPLING),
            source(names.LONGITUDINAL_COUPLING),
        ),
        target_parameters=(
            target(names.TRANSVERSE_COUPLING),
            target(names.LONGITUDINAL_COUPLING),
        ),
        source_level=AbstractionLevel.INTERMEDIATE,
        target_level=AbstractionLevel.INTERMEDIATE,
        description=(
            "the spins become spinless fermions on the same N positions; the transverse term "
            "becomes a hopping, the ZZ term a nearest-neighbour interaction, and the "
            "Jordan-Wigner string cancels because the couplings join adjacent sites"
        ),
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
    )
