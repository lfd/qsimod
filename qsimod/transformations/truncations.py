"""Truncations: an infinite-dimensional degree of freedom is made finite.

A truncation is approximate and valid within a declared regime: the discarded states are
removed from the model, and no resource parameter reduces the error.
"""

from __future__ import annotations

from dataclasses import dataclass

from qsimod.artifact import Artifact, DroppedConstant, HamiltonianModel
from qsimod.levels import AbstractionLevel
from qsimod.models import names
from qsimod.models.application import LATTICE_GAUGE_THEORY_PATTERN
from qsimod.models.intermediate import (
    FERMION_SPIN_CHAIN_PATTERN,
    STAGGERED_CONVENTION,
    QuantumLinkConvention,
    quantum_link_model,
)
from qsimod.parameters import Namespace
from qsimod.relations import Definition, ParameterRelation, equation
from qsimod.transform import Approximation, ApproximationKind, Exactness
from qsimod.transformations.base import ModelTransformation
from qsimod.validity import conjunction, much_less_than

__all__ = ["QuantumLinkTruncation", "quantum_link_truncation"]

#: The default threshold of the validity condition of the truncation: the coupling is at most
#: the electric-flux gap.  The condition is a statement about the field sector, not a
#: perturbative one.
DEFAULT_FIELD_SECTOR_THRESHOLD = 1.0


@dataclass(frozen=True)
class QuantumLinkTruncation(ModelTransformation):
    """The quantum-link truncation of a compact U(1) gauge field to a spin-1/2 on every link.

    ```
    U_{l,l+1} -> -i (2/sqrt(3)) S^+_{l,l+1}       E_{l,l+1} -> e S^z_{l,l+1}
    ```

    The phase ``-i`` absorbs the ``-i`` of the Kogut-Susskind coupling and yields the real form
    ``(kappa/2)(psi^dag S^+ psi + h.c.)``.  Since ``S^z**2 = 1/4``, the electric term becomes
    the constant ``(N-1) a e^2 / 8``, which is dropped and recorded on the target artifact;
    subsequently ``e -> 1`` and ``a -> 1``.  The literal substitution fixes
    ``kappa = 2/sqrt(3)``; the relation carries only the mass and releases ``kappa`` as a free
    parameter from this level downwards.  In the case study this is transformation (a), from
    ``H_sys`` to ``H_IR1``.

    Attributes:
        convention: the quantum-link convention produced; the uniform substitution yields the
            staggered convention (hopping coupling, staggered mass).

    """

    convention: QuantumLinkConvention = STAGGERED_CONVENTION

    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The quantum-link model with spin-1/2 links."""
        return quantum_link_model(
            matter_sites,
            self.target_namespace,
            self.convention,
            name=self.target_name or "H_QLM",
        )

    def dropped_constant_at(self, source: Artifact) -> DroppedConstant:
        """The constant ``(a/2) sum_l E^2 -> (N-1) a e^2 / 8``, evaluated at the source binding."""
        links = self.matter_sites(source) - 1
        spacing = self.source_namespace.symbol(names.LATTICE_SPACING)
        coupling = self.source_namespace.symbol(names.GAUGE_COUPLING)
        value = (links * spacing * coupling**2.0 / 8).substitute(source.binding.as_dict())
        return DroppedConstant(self.dropped_constant, value.simplified())


def quantum_link_truncation(
    source: Namespace,
    target: Namespace,
    *,
    convention: QuantumLinkConvention = STAGGERED_CONVENTION,
    target_name: str = "H_QLM",
    name: str = "spin-1/2 quantum-link truncation",
    field_sector_threshold: float = DEFAULT_FIELD_SECTOR_THRESHOLD,
) -> QuantumLinkTruncation:
    """Build a spin-1/2 quantum-link truncation between two namespaces.

    Args:
        source: the namespace of the untruncated theory, which supplies ``m`` and
            ``electric_gap``.
        target: the namespace of the truncated model, which owns ``m`` and ``kappa``.
        convention: the quantum-link convention to produce.
        target_name: the name of the model produced.
        name: the name of the transformation.
        field_sector_threshold: the threshold of the regime condition
            ``kappa << electric_gap``.

    Returns:
        The transformation.

    """
    mass_in = source.symbol(names.MASS)
    mass_out = target.symbol(names.MASS)
    coupling = target.symbol(names.COUPLING)
    gap = source.symbol(names.ELECTRIC_GAP)

    relation = ParameterRelation(
        name="quantum-link truncation",
        equations=(
            equation(
                "mass carried over",
                mass_out,
                mass_in,
                "the rest mass is untouched by the gauge-field truncation",
            ),
        ),
        definitions=(
            Definition(target(names.MASS), mass_in, "mass carried over"),
            Definition(source(names.MASS), mass_out, "mass carried over"),
        ),
    )
    return QuantumLinkTruncation(
        name=name,
        source_pattern=LATTICE_GAUGE_THEORY_PATTERN,
        target_pattern=FERMION_SPIN_CHAIN_PATTERN,
        exactness=Exactness.APPROXIMATE,
        approximation=Approximation(
            kind=ApproximationKind.REGIME_LIMITED,
            leading_error_order="leading correction from the discarded |E| >= 3/2 link states",
            note=(
                "kappa is released as a free parameter here; the E**2 term becomes an "
                "additive constant and is dropped"
            ),
        ),
        relation=relation,
        validity=conjunction(
            [
                much_less_than(
                    coupling,
                    gap,
                    name="field sector within truncation",
                    label="kappa/(a e^2/2)",
                    threshold=field_sector_threshold,
                    rationale=(
                        "the coupling must not pay the cost of one extra unit of electric "
                        "flux, or the dynamics leaves the two-state link sector"
                    ),
                )
            ]
        ),
        source_parameters=(source(names.MASS),),
        target_parameters=(target(names.MASS), target(names.COUPLING)),
        source_level=AbstractionLevel.APPLICATION,
        target_level=AbstractionLevel.INTERMEDIATE,
        description=(
            "U -> -i (2/sqrt(3)) S^+ and E -> e S^z, uniformly in l; then e -> 1, a -> 1.  "
            "S^z squared is a constant, so the electric term drops out."
        ),
        dropped_constant="(a/2) sum_l E^2 -> (N-1) a e^2 / 8",
        source_namespace=source,
        target_namespace=target,
        target_name=target_name,
        convention=convention,
    )
