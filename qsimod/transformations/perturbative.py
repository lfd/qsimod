"""Perturbative reductions: the knobs of a hardware model related to an effective theory.

Each relation carries the forward direction of the derivation, from the hardware parameters
to the effective parameters, as closed forms, and is ``UNDER_DETERMINED`` in the inverse
direction, which is therefore posed as a solve for the knob settings.  Every transformation
is ``APPROXIMATE`` and valid within a declared regime.  Three reductions are provided: an
optical superlattice realising a gauge theory, a two-component lattice realising a Heisenberg
magnet, and a tilted lattice realising an Ising chain.
"""

from __future__ import annotations

from dataclasses import dataclass

from qsimod.artifact import Artifact, HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models import names
from qsimod.models.hardware import (
    TWO_COMPONENT_CHAIN_PATTERN,
    tilted_bose_hubbard_chain,
    two_component_bose_hubbard_chain,
)
from qsimod.models.intermediate import (
    BOSONIC_CHAIN_PATTERN,
    ISING_CHAIN_PATTERN,
    SPIN_HALF_CHAIN_PATTERN,
)
from qsimod.parameters import AdmissibleSet, Namespace
from qsimod.relations import Definition, ParameterRelation, equation
from qsimod.scalar import Scalar, sqrt
from qsimod.transform import Approximation, ApproximationKind, Exactness
from qsimod.transformations.base import ModelTransformation
from qsimod.validity import (
    DEFAULT_MUCH_LESS_THAN_THRESHOLD,
    DEFAULT_NONZERO_TOLERANCE,
    Conjunction,
    ValidityCondition,
    conjunction,
    much_less_than,
    non_zero,
)

__all__ = [
    "DIPOLE_ENHANCEMENT",
    "DipoleReduction",
    "Reduction",
    "SecondOrderPerturbationTheory",
    "SuperexchangeReduction",
    "dipole_boundary_field",
    "dipole_coupling",
    "dipole_detuning",
    "dipole_longitudinal",
    "dipole_reduction",
    "dipole_transverse",
    "dipole_validity",
    "pole_sum",
    "second_order_coupling",
    "second_order_mass",
    "second_order_perturbation_theory",
    "superexchange_field",
    "superexchange_longitudinal",
    "superexchange_reduction",
    "superexchange_transverse",
    "superexchange_validity",
    "superlattice_validity",
]


@dataclass(frozen=True)
class Reduction(ModelTransformation):
    """A perturbative transformation from an effective theory to a hardware model.

    Applying the transformation returns the hardware model unbound: the relation is
    under-determined in this direction, and the knob settings are determined by a solve.

    Attributes:
        admissible_set: the admissible knob set of the hardware, attached to the hardware model
            produced.

    """

    admissible_set: AdmissibleSet | None = None

    def _transform(self, artifact: Artifact) -> Artifact:
        """Build the hardware model with its parameters left free."""
        return self.build_target(self.target_sites(self.matter_sites(artifact)))


def pole_sum(namespace: Namespace) -> Scalar:
    """The pole sum of the coupling map.

    ``1/(delta+Delta) + 1/(delta-Delta) + 1/(U-delta+Delta) + 1/(U-delta-Delta)``, whose four
    denominators are the energy offsets of the intermediate configurations.
    """
    interaction = namespace.symbol(names.INTERACTION)
    superlattice = namespace.symbol(names.SUPERLATTICE)
    tilt = namespace.symbol(names.TILT)
    return (
        1 / (superlattice + tilt)
        + 1 / (superlattice - tilt)
        + 1 / (interaction - superlattice + tilt)
        + 1 / (interaction - superlattice - tilt)
    )


def second_order_coupling(namespace: Namespace) -> Scalar:
    """The coupling in the forward direction of the derivation.

    ``kappa = sqrt(2) J**2 * <pole sum>``, quadratic in the tunnelling rate ``J``.
    """
    return sqrt(2) * namespace.symbol(names.TUNNELLING) ** 2.0 * pole_sum(namespace)


def second_order_mass(namespace: Namespace) -> Scalar:
    """The mass in the forward direction of the derivation, ``m = delta - U/2``.

    The mass is the detuning from the resonance ``U = 2 delta``.
    """
    return namespace.symbol(names.SUPERLATTICE) - namespace.symbol(names.INTERACTION) / 2


def superlattice_validity(
    effective: Namespace,
    device: Namespace,
    *,
    much_less_threshold: float = DEFAULT_MUCH_LESS_THAN_THRESHOLD,
    pole_tolerance: float = DEFAULT_NONZERO_TOLERANCE,
) -> Conjunction:
    """The validity conditions of the second-order derivation on the superlattice.

    Four domain conditions exclude the poles of the coupling map; six regime conditions are
    ``J << delta``, ``J << U``, ``|m| << U``, ``Delta << delta``, ``Delta << U`` and
    ``kappa << Delta``.

    Args:
        effective: the namespace of the effective theory, with ``m`` and ``kappa``.
        device: the namespace of the hardware model, with ``J``, ``U``, ``delta`` and
            ``Delta``.
        much_less_threshold: the threshold ``theta`` at which a condition ``a << b`` is
            evaluated as ``a/b <= theta``.
        pole_tolerance: the relative distance from a pole below which a domain condition fails.

    Returns:
        The conjunction.

    """
    mass = effective.symbol(names.MASS)
    coupling = effective.symbol(names.COUPLING)
    tunnelling = device.symbol(names.TUNNELLING)
    interaction = device.symbol(names.INTERACTION)
    superlattice = device.symbol(names.SUPERLATTICE)
    tilt = device.symbol(names.TILT)
    perturbative = "perturbative expansion parameter of the second-order derivation"
    manifold = "the tilt must not disturb the manifold structure"

    return conjunction(
        [
            non_zero(
                superlattice - tilt,
                superlattice,
                name="pole: delta != +Delta",
                label="|delta-Delta|/delta",
                tolerance=pole_tolerance,
                rationale="first pole of the coupling map",
            ),
            non_zero(
                superlattice + tilt,
                superlattice,
                name="pole: delta != -Delta",
                label="|delta+Delta|/delta",
                tolerance=pole_tolerance,
                rationale="second pole of the coupling map",
            ),
            non_zero(
                interaction - superlattice - tilt,
                interaction,
                name="pole: U-delta != +Delta",
                label="|U-delta-Delta|/U",
                tolerance=pole_tolerance,
                rationale="third pole of the coupling map",
            ),
            non_zero(
                interaction - superlattice + tilt,
                interaction,
                name="pole: U-delta != -Delta",
                label="|U-delta+Delta|/U",
                tolerance=pole_tolerance,
                rationale="fourth pole of the coupling map",
            ),
            much_less_than(
                tunnelling,
                superlattice,
                name="J << delta",
                label="J/delta",
                threshold=much_less_threshold,
                rationale=perturbative,
            ),
            much_less_than(
                tunnelling,
                interaction,
                name="J << U",
                label="J/U",
                threshold=much_less_threshold,
                rationale=perturbative,
            ),
            much_less_than(
                mass,
                interaction,
                name="|m| << U  (U ~= 2 delta)",
                label="|m|/U",
                threshold=much_less_threshold,
                rationale="near-degeneracy of the manifold: m = delta - U/2 is the detuning",
            ),
            much_less_than(
                tilt,
                superlattice,
                name="Delta << delta",
                label="Delta/delta",
                threshold=much_less_threshold,
                rationale=manifold,
            ),
            much_less_than(
                tilt,
                interaction,
                name="Delta << U",
                label="Delta/U",
                threshold=much_less_threshold,
                rationale=manifold,
            ),
            much_less_than(
                coupling,
                tilt,
                name="kappa << Delta",
                label="kappa/Delta",
                threshold=much_less_threshold,
                rationale=(
                    "the tilt must suppress second-order tunnelling over two sites, which "
                    "would break gauge invariance"
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class SecondOrderPerturbationTheory(Reduction):
    """Second-order degenerate perturbation theory on an optical superlattice.

    ```
    m = delta - U/2
    kappa = sqrt(2) J^2 [ 1/(delta+Delta) + 1/(delta-Delta)
                        + 1/(U-delta+Delta) + 1/(U-delta-Delta) ]
    ```

    The relations are derived in the ``|101> <-> |020>`` manifold of each three-site block,
    which is not the ground-state manifold.  In the case study this is the step from
    ``H_IR3`` to ``H_sim``, solved for the knob settings ``Theta_sim = {J, U, delta,
    Delta}``.
    """

    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The tilted Bose-Hubbard chain, with the admissible knob set of the hardware."""
        return tilted_bose_hubbard_chain(
            matter_sites,
            self.target_namespace,
            admissible_set=self.admissible_set,
            name=self.target_name or "H_BHM",
        )


def second_order_perturbation_theory(
    source: Namespace,
    target: Namespace,
    *,
    admissible_set: AdmissibleSet | None = None,
    validity: Conjunction | None = None,
    target_name: str = "H_BHM",
    name: str = "second-order degenerate perturbation theory, inverted",
) -> SecondOrderPerturbationTheory:
    """Build the perturbative transformation between an effective theory and a superlattice.

    The relation carries the two equations of the forward direction, both solved forms of the
    mass equation, and the positive branch of the coupling equation solved for ``J``.

    Args:
        source: the namespace of the effective theory, with ``m`` and ``kappa``.
        target: the namespace of the hardware model, with ``J``, ``U``, ``delta`` and ``Delta``.
        admissible_set: the admissible knob set of the hardware; defaults to that of the
            hardware model.
        validity: a replacement for the default validity conditions.
        target_name: the name of the model produced.
        name: the name of the transformation.

    Returns:
        The transformation.

    """
    mass = source.symbol(names.MASS)
    coupling = source.symbol(names.COUPLING)
    interaction = target.symbol(names.INTERACTION)
    superlattice = target.symbol(names.SUPERLATTICE)
    coupling_map = second_order_coupling(target)
    mass_map = second_order_mass(target)
    poles = pole_sum(target)

    relation = ParameterRelation(
        name="second-order degenerate perturbation theory",
        equations=(
            equation(
                "mass from detuning",
                mass,
                mass_map,
                "m = delta - U/2: the mass is the detuning from resonance",
            ),
            equation(
                "coupling from second order",
                coupling,
                coupling_map,
                "kappa = sqrt(2) J^2 times the sum of the four pole terms",
            ),
        ),
        definitions=(
            Definition(source(names.MASS), mass_map, "mass from detuning"),
            Definition(source(names.COUPLING), coupling_map, "coupling from second order"),
            Definition(
                target(names.SUPERLATTICE),
                mass + interaction / 2,
                "mass from detuning",
                note="delta = m + U/2, at fixed U",
            ),
            Definition(
                target(names.INTERACTION),
                2 * (superlattice - mass),
                "mass from detuning",
                note="U = 2(delta - m), at fixed delta",
            ),
            Definition(
                target(names.TUNNELLING),
                sqrt(coupling / (sqrt(2) * poles)),
                "coupling from second order",
                note=(
                    "positive branch of J = sqrt(kappa / (sqrt(2) * pole sum)), at fixed "
                    "(U, delta, Delta)"
                ),
            ),
        ),
    )
    return SecondOrderPerturbationTheory(
        name=name,
        source_pattern=BOSONIC_CHAIN_PATTERN,
        target_pattern=BOSONIC_CHAIN_PATTERN,
        exactness=Exactness.APPROXIMATE,
        approximation=Approximation(
            kind=ApproximationKind.REGIME_LIMITED,
            leading_error_order=(
                "fourth order in J (the third-order terms cancel); relative to kappa ~ J^2/U "
                "a relative error of order (J/U)^2"
            ),
            note="reducing J/U reduces kappa with it",
        ),
        relation=relation,
        validity=validity or superlattice_validity(source, target),
        source_parameters=(source(names.MASS), source(names.COUPLING)),
        target_parameters=(
            target(names.TUNNELLING),
            target(names.INTERACTION),
            target(names.SUPERLATTICE),
            target(names.TILT),
        ),
        source_level=AbstractionLevel.INTERMEDIATE,
        target_level=AbstractionLevel.HARDWARE,
        description=(
            "degenerate perturbation theory in the |101> <-> |020> manifold of each three-site "
            "block, at second order in J; derived hardware-to-theory, solved theory-to-hardware"
        ),
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
        admissible_set=admissible_set,
    )


# ---------------------------------------------------------------------------
# Superexchange: a two-component optical lattice to a Heisenberg magnet
# ---------------------------------------------------------------------------


def _virtual_energy(namespace: Namespace, channel: str) -> Scalar:
    """The second-order energy ``4 t**2 / U_channel`` of one virtual-tunnelling channel."""
    return 4 * namespace.symbol(names.HOPPING) ** 2.0 / namespace.symbol(channel)


def superexchange_transverse(namespace: Namespace) -> Scalar:
    """The transverse coupling ``Jxy = -4 t**2 / U_ud``."""
    return -_virtual_energy(namespace, names.INTERACTION_MIXED)


def superexchange_longitudinal(namespace: Namespace) -> Scalar:
    """The longitudinal coupling ``Jz = 4 t**2 / U_ud - (4 t**2 / U_uu + 4 t**2 / U_dd)``."""
    return (
        _virtual_energy(namespace, names.INTERACTION_MIXED)
        - _virtual_energy(namespace, names.INTERACTION_UP)
        - _virtual_energy(namespace, names.INTERACTION_DOWN)
    )


def superexchange_field(namespace: Namespace) -> Scalar:
    """The longitudinal field the derivation also produces, ``2 t**2 (1/U_dd - 1/U_uu)``.

    Per bond, second-order perturbation theory yields ``-(A - B)/2 (S^z_j + S^z_{j+1})`` with
    ``A = 4t**2/U_uu`` and ``B = 4t**2/U_dd``.  Summed over the bonds, this is the field
    strength times ``sum_j z_j S^z_j`` of
    [`weighted_magnetisation_term`][qsimod.models.magnetism.weighted_magnetisation_term],
    which is a constant within a magnetisation sector and a field on the two end spins.  The
    field vanishes at ``U_uu = U_dd`` and is not part of the XXZ Hamiltonian of the target.
    """
    hopping = namespace.symbol(names.HOPPING)
    return (
        2
        * hopping**2.0
        * (
            1 / namespace.symbol(names.INTERACTION_DOWN)
            - 1 / namespace.symbol(names.INTERACTION_UP)
        )
    )


def superexchange_validity(
    magnet: Namespace,
    device: Namespace,
    *,
    much_less_threshold: float = DEFAULT_MUCH_LESS_THAN_THRESHOLD,
    pole_tolerance: float = DEFAULT_NONZERO_TOLERANCE,
) -> Conjunction:
    """The validity conditions of the superexchange derivation.

    Three domain conditions, ``U != 0`` per channel, and six regime conditions: ``t << |U|``
    per channel, ``|Jxy| << |U_ud|``, ``|Jz| << |U_uu|`` and ``|Jz| << |U_dd|``.

    Args:
        magnet: the namespace of the magnet, with ``Jxy`` and ``Jz``.
        device: the namespace of the hardware model, with ``t``, ``U_uu``, ``U_ud`` and
            ``U_dd``.
        much_less_threshold: the threshold ``theta`` at which a condition ``a << b`` is
            evaluated as ``a/b <= theta``.
        pole_tolerance: the relative distance from a pole below which a domain condition fails.

    Returns:
        The conjunction.

    """
    transverse = magnet.symbol(names.TRANSVERSE_COUPLING)
    longitudinal = magnet.symbol(names.LONGITUDINAL_COUPLING)
    hopping = device.symbol(names.HOPPING)
    channels = {
        names.INTERACTION_UP: device.symbol(names.INTERACTION_UP),
        names.INTERACTION_MIXED: device.symbol(names.INTERACTION_MIXED),
        names.INTERACTION_DOWN: device.symbol(names.INTERACTION_DOWN),
    }
    perturbative = "perturbative expansion parameter of the superexchange derivation"
    separated = (
        "the derived coupling must be small on the scale of the doubly occupied states it "
        "is generated by"
    )

    conditions: list[ValidityCondition] = []
    conditions += [
        non_zero(
            interaction,
            hopping,
            name=f"pole: {channel} != 0",
            label=f"|{channel}|/t",
            tolerance=pole_tolerance,
            rationale=f"the {channel} virtual energy 4t^2/{channel} has a pole here",
        )
        for channel, interaction in channels.items()
    ]
    conditions += [
        much_less_than(
            hopping,
            interaction,
            name=f"t << {channel}",
            label=f"t/{channel}",
            threshold=much_less_threshold,
            rationale=perturbative,
        )
        for channel, interaction in channels.items()
    ]
    conditions.append(
        much_less_than(
            transverse,
            channels[names.INTERACTION_MIXED],
            name="|Jxy| << U_ud",
            label="|Jxy/U_ud|",
            threshold=much_less_threshold,
            rationale=separated,
        )
    )
    conditions += [
        much_less_than(
            longitudinal,
            channels[channel],
            name=f"|Jz| << {channel}",
            label=f"|Jz/{channel}|",
            threshold=much_less_threshold,
            rationale=separated,
        )
        for channel in (names.INTERACTION_UP, names.INTERACTION_DOWN)
    ]
    return conjunction(conditions)


@dataclass(frozen=True)
class SuperexchangeReduction(Reduction):
    """Second-order superexchange on a two-component optical lattice.

    ```
    Jxy = -4 t^2 / U_ud
    Jz  =  4 t^2 / U_ud - ( 4 t^2 / U_uu + 4 t^2 / U_dd )
    ```

    The relations are derived in the one-atom-per-site manifold, which on the attractive branch
    lies above the doubly occupied states.
    """

    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The two-component Bose-Hubbard chain, with the admissible knob set of the hardware."""
        return two_component_bose_hubbard_chain(
            matter_sites,
            self.target_namespace,
            admissible_set=self.admissible_set,
            name=self.target_name or "H_2BHM",
        )


def superexchange_reduction(
    source: Namespace,
    target: Namespace,
    *,
    admissible_set: AdmissibleSet | None = None,
    validity: Conjunction | None = None,
    target_name: str = "H_2BHM",
    name: str = "second-order superexchange, inverted",
) -> SuperexchangeReduction:
    """Build the superexchange transformation between a magnet and a two-component lattice.

    The relation carries the two forward equations and their solved forms; with only
    ``(Jxy, Jz)`` known it is ``UNDER_DETERMINED`` by two.

    Args:
        source: the namespace of the magnet, with ``Jxy`` and ``Jz``.
        target: the namespace of the hardware model, with ``t``, ``U_uu``, ``U_ud`` and
            ``U_dd``.
        admissible_set: the admissible knob set of the hardware; defaults to that of the
            hardware model.
        validity: a replacement for the default validity conditions.
        target_name: the name of the model produced.
        name: the name of the transformation.

    Returns:
        The transformation.

    """
    transverse = source.symbol(names.TRANSVERSE_COUPLING)
    longitudinal = source.symbol(names.LONGITUDINAL_COUPLING)
    hopping = target.symbol(names.HOPPING)
    transverse_map = superexchange_transverse(target)
    longitudinal_map = superexchange_longitudinal(target)
    mixed = _virtual_energy(target, names.INTERACTION_MIXED)
    up = _virtual_energy(target, names.INTERACTION_UP)
    down = _virtual_energy(target, names.INTERACTION_DOWN)

    relation = ParameterRelation(
        name="second-order superexchange",
        equations=(
            equation(
                "transverse from superexchange",
                transverse,
                transverse_map,
                "Jxy = -4 t^2 / U_ud: only the interspecies channel exchanges two spins",
            ),
            equation(
                "longitudinal from superexchange",
                longitudinal,
                longitudinal_map,
                "Jz = 4t^2/U_ud - (4t^2/U_uu + 4t^2/U_dd): aligned against anti-aligned",
            ),
        ),
        definitions=(
            Definition(
                source(names.TRANSVERSE_COUPLING),
                transverse_map,
                "transverse from superexchange",
            ),
            Definition(
                source(names.LONGITUDINAL_COUPLING),
                longitudinal_map,
                "longitudinal from superexchange",
            ),
            Definition(
                target(names.INTERACTION_MIXED),
                -4 * hopping**2.0 / transverse,
                "transverse from superexchange",
                note="U_ud = -4 t^2 / Jxy, at fixed t",
            ),
            Definition(
                target(names.HOPPING),
                sqrt(-transverse * target.symbol(names.INTERACTION_MIXED) / 4),
                "transverse from superexchange",
                note="positive branch of t = sqrt(-Jxy U_ud / 4), at fixed U_ud",
            ),
            Definition(
                target(names.INTERACTION_UP),
                4 * hopping**2.0 / (mixed - down - longitudinal),
                "longitudinal from superexchange",
                note="U_uu from the longitudinal equation, at fixed (t, U_ud, U_dd)",
            ),
            Definition(
                target(names.INTERACTION_DOWN),
                4 * hopping**2.0 / (mixed - up - longitudinal),
                "longitudinal from superexchange",
                note="U_dd from the longitudinal equation, at fixed (t, U_ud, U_uu)",
            ),
        ),
    )
    return SuperexchangeReduction(
        name=name,
        source_pattern=SPIN_HALF_CHAIN_PATTERN,
        target_pattern=TWO_COMPONENT_CHAIN_PATTERN,
        exactness=Exactness.APPROXIMATE,
        approximation=Approximation(
            kind=ApproximationKind.REGIME_LIMITED,
            leading_error_order=(
                "fourth order in t; relative to Jxy ~ 4 t^2 / U a relative error of order (t/U)^2"
            ),
            note=(
                "the derivation also produces the longitudinal field superexchange_field(device) "
                "times sum_j z_j S^z_j, which the XXZ Hamiltonian does not carry; it vanishes "
                "at U_uu = U_dd"
            ),
        ),
        relation=relation,
        validity=validity or superexchange_validity(source, target),
        source_parameters=(
            source(names.TRANSVERSE_COUPLING),
            source(names.LONGITUDINAL_COUPLING),
        ),
        target_parameters=(
            target(names.HOPPING),
            target(names.INTERACTION_UP),
            target(names.INTERACTION_MIXED),
            target(names.INTERACTION_DOWN),
        ),
        source_level=AbstractionLevel.INTERMEDIATE,
        target_level=AbstractionLevel.HARDWARE,
        description=(
            "degenerate perturbation theory in the one-atom-per-site manifold, at second "
            "order in the tunnelling; derived hardware-to-theory, solved theory-to-hardware"
        ),
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
        admissible_set=admissible_set,
    )


# ---------------------------------------------------------------------------
# Resonant dipoles: a tilted optical lattice to an Ising magnet
# ---------------------------------------------------------------------------

#: The bosonic enhancement ``2 sqrt(M(M+1))`` at unit filling ``M = 1`` of the resonant
#: tunnelling, which becomes the transverse field.
DIPOLE_ENHANCEMENT = 2.0 * 2.0**0.5


def dipole_detuning(namespace: Namespace) -> Scalar:
    """The detuning ``Delta - U`` of the tilt from the dipole resonance ``Delta = U``."""
    return namespace.symbol(names.TILT) - namespace.symbol(names.INTERACTION)


def dipole_coupling(namespace: Namespace) -> Scalar:
    """The Ising coupling ``Jz = U``.

    The constraint that forbids adjacent dipoles is a penalty of order ``U`` (Simon et al.
    2011) and is taken equal to ``U``.  Both fields are divided by this penalty, so the
    dimensionless fields ``(hz, hx)`` do not depend on this choice.
    """
    return namespace.symbol(names.INTERACTION)


def dipole_transverse(namespace: Namespace) -> Scalar:
    """The transverse field ``Gamma = 2 sqrt(2) J``, of first order in the tunnelling."""
    return DIPOLE_ENHANCEMENT * namespace.symbol(names.TUNNELLING)


def dipole_longitudinal(magnet: Namespace, device: Namespace) -> Scalar:
    """The longitudinal field ``B = Jz - (Delta - U)``, that is ``hz = 1 - (Delta - U)/Jz``."""
    return magnet.symbol(names.LONGITUDINAL_COUPLING) - dipole_detuning(device)


def dipole_boundary_field(namespace: Namespace) -> Scalar:
    """The end-spin field ``Jz / 2`` that the Ising Hamiltonian of the target omits.

    The constraint is a sum over bonds, ``Jz sum_j (1/2 - S^z_j)(1/2 - S^z_{j+1})``.  Written as
    a coupling and the uniform field ``-Jz sum_j S^z_j``, it lacks the term ``+(Jz/2) S^z`` on
    each end spin; see
    [`end_magnetisation_term`][qsimod.models.magnetism.end_magnetisation_term].
    """
    return dipole_coupling(namespace) / 2


def dipole_validity(
    magnet: Namespace,
    device: Namespace,
    *,
    much_less_threshold: float = DEFAULT_MUCH_LESS_THAN_THRESHOLD,
    pole_tolerance: float = DEFAULT_NONZERO_TOLERANCE,
) -> Conjunction:
    """The validity conditions of the resonant-dipole derivation.

    One domain condition, ``U != 0``, and four regime conditions: ``J << U`` (a Mott
    insulator), ``|Delta - U| << Jz`` and ``Gamma << Jz`` (near the multicritical point) and
    ``delta << Gamma`` (the superlattice is a small staggered field on the spins).

    Args:
        magnet: the namespace of the Ising chain, with ``Jz``, ``Gamma`` and ``B``.
        device: the namespace of the hardware model, with ``J``, ``U``, ``delta`` and
            ``Delta``.
        much_less_threshold: the threshold ``theta`` at which a condition ``a << b`` is
            evaluated as ``a/b <= theta``.
        pole_tolerance: the relative distance from a pole below which a domain condition fails.

    Returns:
        The conjunction.

    """
    coupling = magnet.symbol(names.LONGITUDINAL_COUPLING)
    transverse = magnet.symbol(names.TRANSVERSE_AMPLITUDE)
    tunnelling = device.symbol(names.TUNNELLING)
    interaction = device.symbol(names.INTERACTION)
    superlattice = device.symbol(names.SUPERLATTICE)
    neighbourhood = "the mapping holds near the multicritical point (hz, hx) = (1, 0)"
    return conjunction(
        [
            non_zero(
                interaction,
                tunnelling,
                name="pole: U != 0",
                label="|U|/J",
                tolerance=pole_tolerance,
                rationale="the coupling is the interaction, and both fields are measured by it",
            ),
            much_less_than(
                tunnelling,
                interaction,
                name="J << U",
                label="J/U",
                threshold=much_less_threshold,
                rationale="the sample is a Mott insulator, so an atom moves only on resonance",
            ),
            much_less_than(
                dipole_detuning(device),
                coupling,
                name="|Delta - U| << Jz",
                label="|(Delta-U)/Jz|",
                threshold=much_less_threshold,
                rationale=neighbourhood,
            ),
            much_less_than(
                transverse,
                coupling,
                name="Gamma << Jz",
                label="|Gamma/Jz|",
                threshold=much_less_threshold,
                rationale=neighbourhood,
            ),
            much_less_than(
                superlattice,
                transverse,
                name="delta << Gamma",
                label="delta/Gamma",
                threshold=much_less_threshold,
                rationale=(
                    "the superlattice shifts the dipole energy by +-delta from bond to bond, a "
                    "staggered longitudinal field that must be small against the level spacing"
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class DipoleReduction(Reduction):
    """The resonant dipole reduction: a tilted Mott insulator realising an Ising chain.

    ```
    Jz    = U
    Gamma = 2 sqrt(2) J
    B     = Jz - (Delta - U)
    ```

    A dipole on bond ``j``, that is an atom moved down the tilt onto its neighbour, is a spin
    down; the nearest-neighbour exclusion is the antiferromagnetic coupling, and the tunnelling
    is the transverse field.  There is one spin per bond, so a register of ``2N-1`` sites
    carries ``2N-2`` spins.  The relation has three equations against four knobs, and the
    superlattice is left free.
    """

    @staticmethod
    def lattice_sites(spins: int) -> int:
        """The chain length ``N`` whose register of ``2N-1`` sites carries ``spins`` bonds.

        Raises:
            ValueError: if ``spins`` is odd.

        """
        if spins % 2:
            msg = (
                f"an Ising chain of {spins} spins does not sit on the bonds of an "
                "interleaved register: a register of 2N-1 sites has 2N-2 bonds, which is "
                "even"
            )
            raise ValueError(msg)
        return (spins + 2) // 2

    def target_sites(self, source_sites: int) -> int:
        """One spin per bond: ``2N-2`` spins correspond to the register of ``2N-1`` sites."""
        return self.lattice_sites(source_sites)

    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The tilted Bose-Hubbard chain at the chain length given by ``target_sites``."""
        return tilted_bose_hubbard_chain(
            matter_sites,
            self.target_namespace,
            admissible_set=self.admissible_set,
            name=self.target_name or "H_BHM",
        )


def dipole_reduction(
    source: Namespace,
    target: Namespace,
    *,
    admissible_set: AdmissibleSet | None = None,
    validity: Conjunction | None = None,
    target_name: str = "H_BHM",
    name: str = "resonant dipole reduction, inverted",
) -> DipoleReduction:
    """Build the dipole transformation between an Ising chain and a tilted optical lattice.

    Args:
        source: the namespace of the magnet, with ``Jz``, ``Gamma`` and ``B``.
        target: the namespace of the hardware model, with ``J``, ``U``, ``delta`` and
            ``Delta``.  Passing the namespace that another use case assigned to the hardware
            model makes the artifact shared.
        admissible_set: the admissible knob set of the hardware; defaults to that of the
            hardware model.
        validity: a replacement for the default validity conditions.
        target_name: the name of the model produced.
        name: the name of the transformation.

    Returns:
        The transformation.

    """
    coupling = source.symbol(names.LONGITUDINAL_COUPLING)
    transverse = source.symbol(names.TRANSVERSE_AMPLITUDE)
    longitudinal = source.symbol(names.LONGITUDINAL_BIAS)
    interaction = target.symbol(names.INTERACTION)
    tilt = target.symbol(names.TILT)

    relation = ParameterRelation(
        name="resonant dipole reduction",
        equations=(
            equation(
                "coupling from the constraint",
                coupling,
                dipole_coupling(target),
                "Jz = U: the constraint forbidding adjacent dipoles is of order U",
            ),
            equation(
                "transverse from tunnelling",
                transverse,
                dipole_transverse(target),
                "Gamma = 2 sqrt(2) J: one resonant tunnelling event flips one spin",
            ),
            equation(
                "longitudinal from detuning",
                longitudinal,
                dipole_longitudinal(source, target),
                "B = Jz - (Delta - U): the constraint against the detuning from resonance",
            ),
        ),
        definitions=(
            Definition(
                source(names.LONGITUDINAL_COUPLING),
                dipole_coupling(target),
                "coupling from the constraint",
            ),
            Definition(
                target(names.INTERACTION),
                coupling,
                "coupling from the constraint",
                note="U = Jz, the convention this step takes for a constraint of order U",
            ),
            Definition(
                source(names.TRANSVERSE_AMPLITUDE),
                dipole_transverse(target),
                "transverse from tunnelling",
            ),
            Definition(
                target(names.TUNNELLING),
                transverse / DIPOLE_ENHANCEMENT,
                "transverse from tunnelling",
                note="J = Gamma / (2 sqrt(2))",
            ),
            Definition(
                source(names.LONGITUDINAL_BIAS),
                dipole_longitudinal(source, target),
                "longitudinal from detuning",
            ),
            Definition(
                target(names.TILT),
                coupling - longitudinal + interaction,
                "longitudinal from detuning",
                note="Delta = Jz - B + U, at fixed U",
            ),
            Definition(
                target(names.INTERACTION),
                longitudinal - coupling + tilt,
                "longitudinal from detuning",
                note="U = B - Jz + Delta, at fixed Delta",
            ),
        ),
    )
    return DipoleReduction(
        name=name,
        source_pattern=ISING_CHAIN_PATTERN,
        target_pattern=BOSONIC_CHAIN_PATTERN,
        exactness=Exactness.APPROXIMATE,
        approximation=Approximation(
            kind=ApproximationKind.REGIME_LIMITED,
            leading_error_order=(
                "second order in J/U, from the off-resonant states outside the manifold "
                "(three atoms on a site, dipoles on adjacent bonds)"
            ),
            note=(
                "the constraint is a finite penalty of order U taken equal to U; written as a "
                "coupling plus a uniform field it is short of dipole_boundary_field(device) on "
                "each end spin"
            ),
        ),
        relation=relation,
        validity=validity or dipole_validity(source, target),
        source_parameters=(
            source(names.LONGITUDINAL_COUPLING),
            source(names.TRANSVERSE_AMPLITUDE),
            source(names.LONGITUDINAL_BIAS),
        ),
        target_parameters=(
            target(names.TUNNELLING),
            target(names.INTERACTION),
            target(names.SUPERLATTICE),
            target(names.TILT),
        ),
        source_level=AbstractionLevel.INTERMEDIATE,
        target_level=AbstractionLevel.HARDWARE,
        description=(
            "resonant degenerate perturbation theory in the manifold of non-adjacent dipoles "
            "of a tilted unit-filling Mott insulator; derived hardware-to-theory, solved "
            "theory-to-hardware with the superlattice left free"
        ),
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
        admissible_set=admissible_set,
    )
