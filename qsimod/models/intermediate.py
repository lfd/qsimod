"""Models of the intermediate representations: theory- and hardware-influenced Hamiltonians.

Level 2 of [`AbstractionLevel`][qsimod.levels.AbstractionLevel].  A quantum-link model (QLM)
exists in four self-consistent conventions, given by the coupling form, the mass pattern and
the link operator; [`quantum_link_model`][qsimod.models.intermediate.quantum_link_model] takes
the convention as an argument, and
[`QuantumLinkConvention`][qsimod.models.intermediate.QuantumLinkConvention] rejects a mixed
convention, whose Hamiltonian would not commute with its own Gauss operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from qsimod.artifact import HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models import names
from qsimod.models.gauge import (
    ElectricField,
    GaussForm,
    gauss_operators,
    gauss_sector,
    link_triples,
    local_occupation_subspace,
    matter_gauge_pattern,
    matter_gauge_structure,
)
from qsimod.models.magnetism import (
    coordination_numbers,
    ising_chain_terms,
    spin_chain_pattern,
    spin_chain_structure,
    xxz_chain_terms,
)
from qsimod.parameters import Namespace, ParameterSet
from qsimod.scalar import Scalar, sqrt
from qsimod.structure import (
    Algebra,
    StructurePattern,
    interleaved_matter_index,
)
from qsimod.symbolic import (
    OperatorSum,
    SiteOperator,
    annihilate,
    create,
    number,
    sigma_minus,
    sigma_plus,
    spin_minus,
    spin_plus,
    word,
)
from qsimod.units import Dimension

__all__ = [
    "BOSONIC_CHAIN_PATTERN",
    "FERMION_CHAIN_PATTERN",
    "FERMION_SPIN_CHAIN_PATTERN",
    "HOMOGENEOUS_CONVENTION",
    "ISING_CHAIN_PATTERN",
    "QUBIT_CHAIN_PATTERN",
    "SPIN_HALF_CHAIN_PATTERN",
    "STAGGERED_CONVENTION",
    "CouplingForm",
    "LinkPattern",
    "MassPattern",
    "QuantumLinkConvention",
    "bosonic_pair_coupling_model",
    "interacting_fermion_chain",
    "ising_spin_chain",
    "k_local_qubit_model",
    "quantum_link_model",
    "xxz_spin_chain",
]

#: Fermionic matter with spin-1/2 gauge links on an open interleaved chain.
FERMION_SPIN_CHAIN_PATTERN: StructurePattern = matter_gauge_pattern(
    "spinless fermionic matter on an open 1D chain with spin-1/2 gauge links",
    matter_algebras=[Algebra.FERMION],
    gauge_algebras=[Algebra.SPIN_HALF],
)

#: Bosonic modes on every position of an open interleaved chain.
BOSONIC_CHAIN_PATTERN: StructurePattern = matter_gauge_pattern(
    "bosonic modes on every position of an open 1D interleaved chain",
    matter_algebras=[Algebra.BOSON],
    gauge_algebras=[Algebra.BOSON],
)

#: Qubits on every position of an open interleaved chain.
QUBIT_CHAIN_PATTERN: StructurePattern = matter_gauge_pattern(
    "qubits on every position of an open 1D interleaved chain",
    matter_algebras=[Algebra.QUBIT],
    gauge_algebras=[Algebra.QUBIT],
)

#: A spin-1/2 chain with conserved total magnetisation.
SPIN_HALF_CHAIN_PATTERN: StructurePattern = spin_chain_pattern(
    "a spin-1/2 chain with conserved total magnetisation",
    [Algebra.SPIN_HALF],
)

#: A chain of spinless fermions with conserved particle number.
FERMION_CHAIN_PATTERN: StructurePattern = spin_chain_pattern(
    "a chain of spinless fermions with conserved particle number",
    [Algebra.FERMION],
)

#: A spin-1/2 chain without conserved magnetisation.
ISING_CHAIN_PATTERN: StructurePattern = spin_chain_pattern(
    "a spin-1/2 chain with no conserved magnetisation",
    [Algebra.SPIN_HALF],
    require_magnetisation=False,
    forbid_magnetisation=True,
)


class CouplingForm(Enum):
    """The matter-gauge coupling carried by a quantum-link model."""

    PAIR = "pair"
    """The pair coupling ``psi_l S^+ psi_{l+1}`` of two annihilation operators; the total
    matter charge is not conserved."""

    HOPPING = "hopping"
    """The hopping coupling ``psi^dag_l S^+ psi_{l+1}`` of one creation and one annihilation
    operator; the charge is conserved."""

    def __str__(self) -> str:
        return self.value


class MassPattern(Enum):
    """Whether the mass term alternates in sign along the chain."""

    UNIFORM = "uniform"
    STAGGERED = "staggered"

    def __str__(self) -> str:
        return self.value


class LinkPattern(Enum):
    """Whether the link operator alternates along the chain."""

    UNIFORM = "uniform"
    """The operator ``S^+`` on every link."""

    ALTERNATING = "alternating"
    """The operators ``S^+`` and ``S^-`` on alternate links.  The pattern is spectrally
    invisible; it changes which ``G_l`` is conserved."""

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class QuantumLinkConvention:
    """One of the four self-consistent quantum-link conventions.

    The mass pattern is determined by the coupling form: a pair coupling is combined with a
    uniform mass and a hopping coupling with a staggered mass, since the particle-hole
    transformation changes both at once.  The link pattern is unconstrained.

    Attributes:
        coupling: the coupling form.
        mass: the mass pattern.
        link: the link operator pattern.

    """

    coupling: CouplingForm
    mass: MassPattern
    link: LinkPattern = LinkPattern.UNIFORM

    def __post_init__(self) -> None:
        expected = (
            MassPattern.UNIFORM if self.coupling is CouplingForm.PAIR else MassPattern.STAGGERED
        )
        if self.mass is not expected:
            msg = (
                f"a {self.coupling} coupling goes with a {expected} mass, not a {self.mass} "
                "one: the particle-hole transformation flips the coupling and cancels the "
                "mass staggering in one step, so mixing them gives a model that does not "
                "commute with its own symmetry generators"
            )
            raise ValueError(msg)

    @property
    def gauss_form(self) -> GaussForm:
        """The generator form conserved by this convention."""
        return GaussForm.HOMOGENEOUS if self.coupling is CouplingForm.PAIR else GaussForm.STAGGERED

    def __str__(self) -> str:
        return f"{self.coupling} coupling, {self.mass} mass, {self.link} link"


STAGGERED_CONVENTION = QuantumLinkConvention(CouplingForm.HOPPING, MassPattern.STAGGERED)
"""The convention before a particle-hole transformation: hopping coupling and staggered mass.

This is the convention of the quantum-link model ``H_IR1`` of the article, the artifact ``L2a``
of the case study.
"""

HOMOGENEOUS_CONVENTION = QuantumLinkConvention(CouplingForm.PAIR, MassPattern.UNIFORM)
"""The convention after a particle-hole transformation: pair coupling and uniform mass.

This is the convention of the quantum-link model ``H_IR2`` of the article, the artifact ``L2b``
of the case study and its branch point.
"""


def _matter_operator(site: int, form: CouplingForm) -> SiteOperator:
    """The first matter factor of the coupling term."""
    return annihilate(site) if form is CouplingForm.PAIR else create(site)


def _link_operator(link: int, site: int, pattern: LinkPattern) -> SiteOperator:
    """The link factor of the coupling term."""
    if pattern is LinkPattern.ALTERNATING and link % 2 == 1:
        return spin_minus(site)
    return spin_plus(site)


def _mass_sign(site: int, pattern: MassPattern) -> int:
    return (-1) ** site if pattern is MassPattern.STAGGERED else 1


def _mass_and_constant(
    matter_sites: int,
    mass: Scalar,
    pattern: MassPattern,
    additive_constant: Scalar | float | None,
) -> OperatorSum:
    """The mass term over every matter position, together with any retained additive constant."""
    total = OperatorSum()
    for site in range(matter_sites):
        total = total + word(
            mass * _mass_sign(site, pattern),
            number(interleaved_matter_index(site)),
        )
    if additive_constant is not None:
        total = total + word(additive_constant)
    return total


def _theory_parameters(namespace: Namespace, mass_note: str, coupling_note: str) -> ParameterSet:
    """The parameter set ``(m, kappa)`` of a theory-side model."""
    return namespace.parameter_set(
        (names.MASS, Dimension.ENERGY, mass_note),
        (names.COUPLING, Dimension.ENERGY, coupling_note),
    )


def quantum_link_model(
    matter_sites: int,
    namespace: Namespace,
    convention: QuantumLinkConvention = HOMOGENEOUS_CONVENTION,
    *,
    name: str = "H_QLM",
    hamiltonian_name: str = "",
    additive_constant: Scalar | float | None = None,
) -> HamiltonianModel:
    """A spin-1/2 U(1) quantum-link model, in any of the four conventions.

    ```
    H = sum_l [ (kappa/2)( <coupling> + h.c. ) + m <mass sign>_l psi^dag_l psi_l ] + constant
    ```

    In the case study of the article the staggered convention yields the quantum-link model
    ``H_IR1`` at the artifact ``L2a``, and the homogeneous convention yields ``H_IR2`` at the
    artifact ``L2b``, the branch point; both carry the parameter set
    ``Theta_IR1 = Theta_IR2 = {m, kappa}``.

    Args:
        matter_sites: the chain length ``N``.
        namespace: the parameter namespace owned by this model; it names ``m`` and ``kappa``.
        convention: the convention of the model; defaults to the homogeneous one.
        name: the name of the model.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.
        additive_constant: a constant retained as an explicit identity term, for instance the
            ``-m * floor(N/2)`` left behind by a particle-hole transformation.

    Returns:
        The model, as an intermediate representation.

    """
    mass = namespace.symbol(names.MASS)
    coupling = namespace.symbol(names.COUPLING)

    hamiltonian = OperatorSum()
    for link, left, middle, right in link_triples(matter_sites):
        hamiltonian = (
            hamiltonian
            + word(
                coupling / 2,
                _matter_operator(left, convention.coupling),
                _link_operator(link, middle, convention.link),
                annihilate(right),
            ).plus_adjoint()
        )
    hamiltonian = hamiltonian + _mass_and_constant(
        matter_sites, mass, convention.mass, additive_constant
    )

    constraints = gauss_operators(
        matter_sites, field=ElectricField.SPIN, form=convention.gauss_form
    )
    return HamiltonianModel(
        name=name,
        structure=matter_gauge_structure(
            matter_sites, Algebra.FERMION, Algebra.SPIN_HALF, name=hamiltonian_name or name
        ),
        level=AbstractionLevel.INTERMEDIATE,
        parameters=_theory_parameters(
            namespace,
            f"{convention.mass} fermion mass",
            "gauge-invariant coupling kappa",
        ),
        hamiltonian=hamiltonian.renamed(hamiltonian_name or name),
        constraint_operators=constraints,
        sector=gauss_sector(constraints),
        origin=f"spin-1/2 U(1) quantum-link model ({convention})",
    )


def bosonic_pair_coupling_model(
    matter_sites: int,
    namespace: Namespace,
    *,
    name: str = "H_eff",
    hamiltonian_name: str = "",
    additive_constant: Scalar | float | None = None,
) -> HamiltonianModel:
    """The boson encoding of a pair-coupling quantum-link model.

    ```
    H = sum_{j even, 0 <= j <= 2N-4} [ (kappa / 2 sqrt(2)) b_j b_{j+2} (b^dag_{j+1})^2 + h.c. ]
      + sum_{j even, 0 <= j <= 2N-2}   m n_j
      + constant
    ```

    In the case study of the article this is the effective bosonic model ``H_IR3`` at the
    artifact ``L2c``, with the parameter set ``Theta_IR3 = {m, kappa}``.  The operator
    identities of the encoding hold as ``P H P`` on the declared local occupation subspace,
    ``{0, 1}`` on matter positions and ``{0, 2}`` on links, so the encoding is exact on the
    encoded subspace; [`sandwich`][qsimod.realise.build.sandwich] builds the projected operator.

    Args:
        matter_sites: the chain length ``N``.
        namespace: the parameter namespace; it names ``m`` and ``kappa``.
        name: the name of the model.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.
        additive_constant: a constant retained as an explicit identity term.

    Returns:
        The model, as an intermediate representation.

    """
    mass = namespace.symbol(names.MASS)
    coupling = namespace.symbol(names.COUPLING)

    hamiltonian = OperatorSum()
    for _, left, middle, right in link_triples(matter_sites):
        hamiltonian = (
            hamiltonian
            + word(
                coupling / (2 * sqrt(2)),
                annihilate(left),
                create(middle),
                create(middle),
                annihilate(right),
            ).plus_adjoint()
        )
    hamiltonian = hamiltonian + _mass_and_constant(
        matter_sites, mass, MassPattern.UNIFORM, additive_constant
    )

    constraints = gauss_operators(matter_sites, field=ElectricField.BOSON)
    return HamiltonianModel(
        name=name,
        structure=matter_gauge_structure(
            matter_sites, Algebra.BOSON, Algebra.BOSON, name=hamiltonian_name or name
        ),
        level=AbstractionLevel.INTERMEDIATE,
        parameters=_theory_parameters(
            namespace, "effective rest mass m", "effective pair coupling kappa"
        ),
        hamiltonian=hamiltonian.renamed(hamiltonian_name or name),
        constraint_operators=constraints,
        sector=gauss_sector(constraints),
        local_subspace=local_occupation_subspace(matter_sites),
        origin="bosonic doublon encoding of a pair-coupling quantum-link model",
    )


def k_local_qubit_model(
    matter_sites: int,
    namespace: Namespace,
    convention: QuantumLinkConvention = HOMOGENEOUS_CONVENTION,
    *,
    name: str = "H_qubit",
    hamiltonian_name: str = "",
    additive_constant: Scalar | float | None = None,
) -> HamiltonianModel:
    """The qubit-register form of a quantum-link model, a 3-local Pauli Hamiltonian.

    ```
    H = sum_l (kappa/2) ( sigma^-_{2l} sigma^+_{2l+1} sigma^-_{2l+2} + h.c. )
      + sum_l m n_{2l} + constant                          (pair convention)
    ```

    In the case study of the article this is the qubit Hamiltonian ``H_IR4`` at the artifact
    ``L2d``, the Jordan-Wigner image of ``H_IR2``, with the parameter set
    ``Theta_IR4 = {m, kappa}``; the second digital branch reaches the artifact ``L2d_st`` from
    ``L2a`` in the hopping convention.  In the pair convention the Pauli expansion of a coupling
    term is ``(1/4)( X_aX_bX_c + X_aY_bY_c - Y_aX_bY_c + Y_aY_bX_c )``; the four strings commute
    mutually, and each commutes with every Gauss operator.  In the hopping convention the
    Jordan-Wigner string contributes a factor ``-1`` to the coupling.

    Args:
        matter_sites: the chain length ``N``.
        namespace: the parameter namespace; it names ``m`` and ``kappa``.
        convention: the convention of which this model is the image.
        name: the name of the model.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.
        additive_constant: a constant retained as an explicit identity term.

    Returns:
        The model, as an intermediate representation.

    """
    mass = namespace.symbol(names.MASS)
    coupling = namespace.symbol(names.COUPLING)
    pair = convention.coupling is CouplingForm.PAIR

    hamiltonian = OperatorSum()
    for link, left, middle, right in link_triples(matter_sites):
        first = sigma_minus(left) if pair else sigma_plus(left)
        centre = (
            sigma_minus(middle)
            if convention.link is LinkPattern.ALTERNATING and link % 2 == 1
            else sigma_plus(middle)
        )
        # Jordan-Wigner sign under the alternating-sign convention of qsimod.realise.build.
        sign = 1 if pair else -1
        hamiltonian = (
            hamiltonian
            + word(sign * coupling / 2, first, centre, sigma_minus(right)).plus_adjoint()
        )
    hamiltonian = hamiltonian + _mass_and_constant(
        matter_sites, mass, convention.mass, additive_constant
    )

    constraints = gauss_operators(
        matter_sites, field=ElectricField.QUBIT, form=convention.gauss_form
    )
    return HamiltonianModel(
        name=name,
        structure=matter_gauge_structure(
            matter_sites, Algebra.QUBIT, Algebra.QUBIT, name=hamiltonian_name or name
        ),
        level=AbstractionLevel.INTERMEDIATE,
        parameters=_theory_parameters(
            namespace, f"{convention.mass} mass on the qubit register", "3-local coupling kappa"
        ),
        hamiltonian=hamiltonian.renamed(hamiltonian_name or name),
        constraint_operators=constraints,
        sector=gauss_sector(constraints),
        origin=(
            "Jordan-Wigner image of a quantum-link model on an interleaved qubit register "
            f"({convention})"
        ),
    )


# ---------------------------------------------------------------------------
# The magnetism branch: a spin chain and its Jordan-Wigner image
# ---------------------------------------------------------------------------


def _magnet_parameters(namespace: Namespace, transverse: str, longitudinal: str) -> ParameterSet:
    """The parameter set ``(Jxy, Jz)`` of a magnetism model."""
    return namespace.parameter_set(
        (names.TRANSVERSE_COUPLING, Dimension.ENERGY, transverse),
        (names.LONGITUDINAL_COUPLING, Dimension.ENERGY, longitudinal),
    )


def xxz_spin_chain(
    sites: int,
    namespace: Namespace,
    *,
    name: str = "H_XXZ",
    hamiltonian_name: str = "",
) -> HamiltonianModel:
    """The XXZ chain, stated by two coupling energies.

    ```
    H = sum_{j=0}^{N-2} [ (Jxy/2)( S^+_j S^-_{j+1} + h.c. ) + Jz S^z_j S^z_{j+1} ]
    ```

    Args:
        sites: the chain length ``N``.
        namespace: the parameter namespace owned by this model; it names ``Jxy`` and ``Jz``.
        name: the name of the model.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.

    Returns:
        The model, as an intermediate representation.

    """
    transverse = namespace.symbol(names.TRANSVERSE_COUPLING)
    longitudinal = namespace.symbol(names.LONGITUDINAL_COUPLING)
    return HamiltonianModel(
        name=name,
        structure=spin_chain_structure(sites, Algebra.SPIN_HALF, name=hamiltonian_name or name),
        level=AbstractionLevel.INTERMEDIATE,
        parameters=_magnet_parameters(namespace, "spin-exchange coupling Jxy", "ZZ coupling Jz"),
        hamiltonian=xxz_chain_terms(sites, transverse, longitudinal).renamed(
            hamiltonian_name or name
        ),
        origin="spin-1/2 XXZ chain with independent transverse and longitudinal couplings",
    )


def interacting_fermion_chain(
    sites: int,
    namespace: Namespace,
    *,
    name: str = "H_tV",
    hamiltonian_name: str = "",
) -> HamiltonianModel:
    """The Jordan-Wigner image of an XXZ chain: spinless fermions with a nearest-neighbour ``V``.

    ```
    H = - (Jxy/2) sum_j ( c^dag_j c_{j+1} + h.c. )
      + Jz sum_j n_j n_{j+1}
      - (Jz/2) sum_j z_j n_j
      + (N-1) Jz/4
    ```

    The symbol ``z_j`` denotes the coordination number of site ``j``; the last two terms are
    the expansion of ``Jz sum_j (n_j - 1/2)(n_{j+1} - 1/2)``, constant included.  The sign of
    the hopping is the alternating-sign Jordan-Wigner gauge of
    [`qsimod.realise.build`][qsimod.realise.build], under which this model and the spin chain
    realise to the same matrix; at ``Jz = 0`` the single-particle band is
    ``E(q) = -Jxy cos(qa)``.

    Args:
        sites: the chain length ``N``.
        namespace: the parameter namespace; it names ``Jxy`` and ``Jz``.
        name: the name of the model.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.

    Returns:
        The model, as an intermediate representation.

    """
    transverse = namespace.symbol(names.TRANSVERSE_COUPLING)
    longitudinal = namespace.symbol(names.LONGITUDINAL_COUPLING)

    hamiltonian = OperatorSum()
    for site in range(sites - 1):
        hamiltonian = (
            hamiltonian + word(-transverse / 2, create(site), annihilate(site + 1)).plus_adjoint()
        )
        hamiltonian = hamiltonian + word(longitudinal, number(site), number(site + 1))
    for site, coordination in enumerate(coordination_numbers(sites)):
        hamiltonian = hamiltonian + word(-longitudinal * coordination / 2, number(site))
    hamiltonian = hamiltonian + word(longitudinal * (sites - 1) / 4)

    return HamiltonianModel(
        name=name,
        structure=spin_chain_structure(sites, Algebra.FERMION, name=hamiltonian_name or name),
        level=AbstractionLevel.INTERMEDIATE,
        parameters=_magnet_parameters(
            namespace,
            "hopping amplitude, as Jxy/2 in the spin language",
            "nearest-neighbour interaction V, equal to Jz",
        ),
        hamiltonian=hamiltonian.renamed(hamiltonian_name or name),
        origin="Jordan-Wigner image of an XXZ chain: spinless fermions with nearest-neighbour V",
    )


def ising_spin_chain(
    sites: int,
    namespace: Namespace,
    *,
    name: str = "H_Ising",
    hamiltonian_name: str = "",
) -> HamiltonianModel:
    """The Ising chain, stated by three energies.

    ```
    H = Jz sum_{j=0}^{N-2} S^z_j S^z_{j+1} - Gamma sum_j S^x_j - B sum_j S^z_j
    ```

    Args:
        sites: the chain length ``N``.
        namespace: the parameter namespace; it names ``Jz``, ``Gamma`` and ``B``.
        name: the name of the model.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.

    Returns:
        The model, as an intermediate representation.

    """
    coupling = namespace.symbol(names.LONGITUDINAL_COUPLING)
    transverse = namespace.symbol(names.TRANSVERSE_AMPLITUDE)
    longitudinal = namespace.symbol(names.LONGITUDINAL_BIAS)
    return HamiltonianModel(
        name=name,
        structure=spin_chain_structure(
            sites, Algebra.SPIN_HALF, name=hamiltonian_name or name, symmetries=()
        ),
        level=AbstractionLevel.INTERMEDIATE,
        parameters=namespace.parameter_set(
            (names.LONGITUDINAL_COUPLING, Dimension.ENERGY, "Ising coupling Jz"),
            (names.TRANSVERSE_AMPLITUDE, Dimension.ENERGY, "transverse field Gamma"),
            (names.LONGITUDINAL_BIAS, Dimension.ENERGY, "longitudinal field B"),
        ),
        hamiltonian=ising_chain_terms(sites, coupling, transverse, longitudinal).renamed(
            hamiltonian_name or name
        ),
        origin="antiferromagnetic Ising chain with independent field energies",
    )
