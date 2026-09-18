"""Models of the application layer: physical system models, stated in their own terms.

Level 1 of [`AbstractionLevel`][qsimod.levels.AbstractionLevel].  A physical system model may
have no finite-dimensional representation; for instance, a compact U(1) link carries an
unbounded electric field.  Its structural type declares this property, and a numerical
realisation is refused until a truncation has been applied.
"""

from __future__ import annotations

from qsimod.artifact import HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models import names
from qsimod.models.gauge import (
    ElectricField,
    GaussForm,
    gauss_operators,
    gauss_sector,
    link_triples,
    matter_gauge_pattern,
    matter_gauge_structure,
)
from qsimod.models.magnetism import (
    ising_chain_terms,
    spin_chain_pattern,
    spin_chain_structure,
    xxz_chain_terms,
)
from qsimod.parameters import Namespace
from qsimod.structure import (
    Algebra,
    StructurePattern,
    interleaved_link_index,
    interleaved_matter_index,
)
from qsimod.symbolic import (
    OperatorSum,
    annihilate,
    create,
    electric_field,
    link_u,
    number,
    word,
)
from qsimod.units import Dimension

__all__ = [
    "HEISENBERG_MAGNET_PATTERN",
    "ISING_MAGNET_PATTERN",
    "LATTICE_GAUGE_THEORY_PATTERN",
    "heisenberg_magnet",
    "ising_magnet",
    "kogut_susskind_gauge_theory",
]

#: A spin-1/2 chain with conserved total magnetisation.
HEISENBERG_MAGNET_PATTERN: StructurePattern = spin_chain_pattern(
    "a spin-1/2 chain with conserved total magnetisation",
    [Algebra.SPIN_HALF],
)

#: A spin-1/2 chain without conserved magnetisation; a transverse field breaks the symmetry.
ISING_MAGNET_PATTERN: StructurePattern = spin_chain_pattern(
    "a spin-1/2 chain with no conserved magnetisation",
    [Algebra.SPIN_HALF],
    require_magnetisation=False,
    forbid_magnetisation=True,
)

#: Fermionic matter on an open chain with untruncated U(1) gauge links.
LATTICE_GAUGE_THEORY_PATTERN: StructurePattern = matter_gauge_pattern(
    "staggered fermionic matter on an open 1D chain with untruncated U(1) gauge links",
    matter_algebras=[Algebra.FERMION],
    gauge_algebras=[Algebra.GAUGE_LINK_U1],
)


def kogut_susskind_gauge_theory(
    matter_sites: int,
    namespace: Namespace,
    *,
    name: str = "H_QED",
    hamiltonian_name: str = "H_QED",
) -> HamiltonianModel:
    """The Kogut-Susskind lattice Schwinger model with staggered fermions in 1+1 dimensions.

    The model is one-dimensional lattice quantum electrodynamics (QED), the application model
    ``H_sys`` of the case study of the article, at the artifact ``lattice_qed``:

    ```
    H = (a/2) sum_l E^2_{l,l+1}
      + m sum_l (-1)**l psi^dag_l psi_l
      - (i/2a) sum_l ( psi^dag_l U_{l,l+1} psi_{l+1} - h.c. )
    ```

    The Gauss operators take the staggered form of [`GaussForm`][qsimod.models.gauge.GaussForm]
    with the charge coefficient ``e``.  The parameter set is ``Theta_sys = {m, a, e}``: the
    rest mass ``m``, the lattice spacing ``a`` and the gauge coupling ``e``.  The additional
    parameter ``electric_gap = a e**2 / 2``, the energy of one unit of electric flux, enters no
    term of the Hamiltonian and is read by the validity condition of the quantum-link
    truncation.

    Args:
        matter_sites: the chain length ``N``; the register has ``2N - 1`` positions.
        namespace: the parameter namespace owned by this model.
        name: the name of the model, used as the name of its artifact in a model graph.
        hamiltonian_name: the name under which the Hamiltonian prints.

    Returns:
        The model, in the application layer.

    """
    mass = namespace.symbol(names.MASS)
    spacing = namespace.symbol(names.LATTICE_SPACING)

    hamiltonian = OperatorSum()
    for link in range(matter_sites - 1):
        site = interleaved_link_index(link)
        hamiltonian = hamiltonian + word(spacing / 2, electric_field(site), electric_field(site))
    for site in range(matter_sites):
        hamiltonian = hamiltonian + word(
            mass * (-1) ** site, number(interleaved_matter_index(site))
        )
    for _, left, middle, right in link_triples(matter_sites):
        hamiltonian = (
            hamiltonian
            + word(
                -1j / (2 * spacing), create(left), link_u(middle), annihilate(right)
            ).plus_adjoint()
        )

    constraints = gauss_operators(
        matter_sites,
        field=ElectricField.OPERATOR,
        form=GaussForm.STAGGERED,
        charge=namespace.symbol(names.GAUGE_COUPLING),
    )
    return HamiltonianModel(
        name=name,
        structure=matter_gauge_structure(
            matter_sites, Algebra.FERMION, Algebra.GAUGE_LINK_U1, name=hamiltonian_name
        ),
        level=AbstractionLevel.APPLICATION,
        parameters=namespace.parameter_set(
            (names.MASS, Dimension.ENERGY, "fermion rest mass"),
            (names.LATTICE_SPACING, Dimension.DIMENSIONLESS, "lattice spacing a"),
            (names.GAUGE_COUPLING, Dimension.DIMENSIONLESS, "gauge coupling e"),
            (
                names.ELECTRIC_GAP,
                Dimension.ENERGY,
                "energy cost of one extra unit of electric flux, a*e**2/2",
            ),
        ),
        hamiltonian=hamiltonian.renamed(hamiltonian_name),
        constraint_operators=constraints,
        sector=gauss_sector(constraints),
        origin="Kogut-Susskind lattice gauge theory with staggered fermions, 1+1D",
    )


def heisenberg_magnet(
    sites: int,
    namespace: Namespace,
    *,
    name: str = "H_XXZ",
    hamiltonian_name: str = "",
) -> HamiltonianModel:
    """The anisotropic Heisenberg (XXZ) chain, stated by an energy and a dimensionless anisotropy.

    ```
    H = sum_j [ Jxy ( S^x_j S^x_{j+1} + S^y_j S^y_{j+1} ) + Delta Jxy S^z_j S^z_{j+1} ]
    ```

    The transverse term is written as ``(Jxy/2)( S^+_j S^-_{j+1} + h.c. )``.  The anisotropy
    ``Delta = Jz / Jxy`` is dimensionless: ``Delta = 0`` is the XX point and ``Delta = 1`` the
    isotropic magnet.

    Args:
        sites: the chain length ``N``; the register has ``N`` positions.
        namespace: the parameter namespace owned by this model; it names ``Jxy`` and ``Delta``.
        name: the name of the model, used as the name of its artifact in a model graph.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.

    Returns:
        The model, in the application layer.

    """
    transverse = namespace.symbol(names.TRANSVERSE_COUPLING)
    anisotropy = namespace.symbol(names.ANISOTROPY)
    return HamiltonianModel(
        name=name,
        structure=spin_chain_structure(sites, Algebra.SPIN_HALF, name=hamiltonian_name or name),
        level=AbstractionLevel.APPLICATION,
        parameters=namespace.parameter_set(
            (
                names.TRANSVERSE_COUPLING,
                Dimension.ENERGY,
                "spin-exchange coupling Jxy",
            ),
            (
                names.ANISOTROPY,
                Dimension.DIMENSIONLESS,
                "anisotropy Delta = Jz / Jxy",
            ),
        ),
        hamiltonian=xxz_chain_terms(sites, transverse, anisotropy * transverse).renamed(
            hamiltonian_name or name
        ),
        origin="anisotropic Heisenberg (XXZ) chain, stated by its anisotropy",
    )


def ising_magnet(
    sites: int,
    namespace: Namespace,
    *,
    name: str = "H_Ising",
    hamiltonian_name: str = "",
) -> HamiltonianModel:
    """The antiferromagnetic Ising chain in longitudinal and transverse fields.

    ```
    H = Jz sum_j ( S^z_j S^z_{j+1} - hz S^z_j - hx S^x_j )
    ```

    The parameters are one energy ``Jz`` and two dimensionless fields ``(hz, hx)``, as in Simon
    et al., Nature **472**, 307 (2011).  The transverse field breaks the conservation of the
    magnetisation, so the structural type declares no symmetry.

    Args:
        sites: the chain length ``N``.
        namespace: the parameter namespace owned by this model; it names ``Jz``, ``hz`` and
            ``hx``.
        name: the name of the model, used as the name of its artifact in a model graph.
        hamiltonian_name: the name under which the Hamiltonian prints; defaults to ``name``.

    Returns:
        The model, in the application layer.

    """
    coupling = namespace.symbol(names.LONGITUDINAL_COUPLING)
    longitudinal = namespace.symbol(names.LONGITUDINAL_FIELD)
    transverse = namespace.symbol(names.TRANSVERSE_FIELD)
    return HamiltonianModel(
        name=name,
        structure=spin_chain_structure(
            sites, Algebra.SPIN_HALF, name=hamiltonian_name or name, symmetries=()
        ),
        level=AbstractionLevel.APPLICATION,
        parameters=namespace.parameter_set(
            (names.LONGITUDINAL_COUPLING, Dimension.ENERGY, "Ising coupling Jz"),
            (
                names.LONGITUDINAL_FIELD,
                Dimension.DIMENSIONLESS,
                "longitudinal field hz, in units of Jz",
            ),
            (
                names.TRANSVERSE_FIELD,
                Dimension.DIMENSIONLESS,
                "transverse field hx, in units of Jz",
            ),
        ),
        hamiltonian=ising_chain_terms(
            sites, coupling, coupling * transverse, coupling * longitudinal
        ).renamed(hamiltonian_name or name),
        origin="antiferromagnetic Ising chain, stated by its dimensionless fields",
    )
