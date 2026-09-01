"""Base classes of the transformations in the library.

[`NamespacedTransformation`][qsimod.transformations.base.NamespacedTransformation] adds the
two parameter namespaces a library transformation relates.
[`ModelTransformation`][qsimod.transformations.base.ModelTransformation] adds the common
application: the target model is built at the mapped chain length, the dropped constants are
forwarded, and the parameters determined by the solved forms of the relation are bound.  A
transformation defined outside the library need not subclass either class;
[`Transformation`][qsimod.transform.Transformation] is the contract.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field, replace

from qsimod.artifact import Artifact, DroppedConstant, HamiltonianModel
from qsimod.parameters import Namespace
from qsimod.structure import StructureType
from qsimod.transform import Exactness, ExactnessError, Transformation

__all__ = ["ModelTransformation", "NamespacedTransformation"]


@dataclass(frozen=True)
class NamespacedTransformation(Transformation):
    """A transformation between two models, each of which owns one parameter namespace.

    Attributes:
        source_namespace: the namespace of the model consumed.
        target_namespace: the namespace of the model produced.
        target_name: the name of the model produced, which is the name of its artifact in the
            model graph.

    """

    source_namespace: Namespace = field(default_factory=lambda: Namespace("source"))
    target_namespace: Namespace = field(default_factory=lambda: Namespace("target"))
    target_name: str = ""

    @staticmethod
    def matter_sites(artifact: Artifact) -> int:
        """The chain length of the structural type of an artifact."""
        return artifact.structure.lattice.matter_sites


@dataclass(frozen=True)
class ModelTransformation(NamespacedTransformation):
    """A transformation whose target is a Hamiltonian model built by the model library.

    Subclasses implement
    [`build_target`][qsimod.transformations.base.ModelTransformation.build_target]; the
    structural type of the target and the application follow from it.
    """

    @abstractmethod
    def build_target(self, matter_sites: int) -> HamiltonianModel:
        """The target model at the given target chain length, with its parameters unbound."""

    def target_structure(self, source: StructureType) -> StructureType:
        """The structural type of the built target; a basis change returns ``source`` instead."""
        return self.build_target(self.target_sites(source.lattice.matter_sites)).structure

    def _transform(self, artifact: Artifact) -> Artifact:
        """Build the target, check an exactness claim, forward constants and bind parameters.

        For an ``EXACT`` transformation with an [`image`][qsimod.transform.Transformation.image],
        the target built by the library must equal that image up to an additive constant; a
        constant difference is recorded on the target as a dropped constant, and any other
        difference raises.  Only the parameters that the solved forms of the relation determine
        from the bound values of the source are bound; the remaining parameters are left free.

        Raises:
            ExactnessError: if the target is not the image of the source.

        """
        target = self.build_target(self.target_sites(self.matter_sites(artifact)))
        if isinstance(artifact, HamiltonianModel):
            dropped = artifact.dropped_constants
            own = self.dropped_constant_at(artifact)
            if own is not None:
                dropped = (*dropped, own)
            if self.exactness is Exactness.EXACT:
                discarded = self._checked_exact(artifact, target)
                if discarded is not None:
                    dropped = (*dropped, discarded)
            target = replace(target, dropped_constants=dropped)
        return target.bind_all(self.carry_parameters(artifact, target))

    def _checked_exact(
        self, source: HamiltonianModel, target: HamiltonianModel
    ) -> DroppedConstant | None:
        """Check the exactness claim and return the constant the target omits, if any."""
        defect = self.exactness_defect(source, target)
        if defect is None or not defect.terms:
            return None
        if any(not term.is_constant for term in defect.terms):
            raise ExactnessError(self.name, defect)
        # `defect = target - image`, so the image carries `-defect` more than the target.
        omitted = (-defect.constant_part()).substitute(source.binding.as_dict()).simplified()
        return DroppedConstant(
            self.dropped_constant or f"additive constant omitted by {self.name}", omitted
        )
