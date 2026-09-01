"""Q-SiMod: a typed metamodel for the model-driven engineering (MDE) of quantum simulation.

The package is organised in layers.  The declaration layer comprises
[`qsimod.scalar`][qsimod.scalar], [`qsimod.affine`][qsimod.affine],
[`qsimod.units`][qsimod.units], [`qsimod.structure`][qsimod.structure],
[`qsimod.symbolic`][qsimod.symbolic], [`qsimod.normal_form`][qsimod.normal_form],
[`qsimod.parameters`][qsimod.parameters], [`qsimod.artifact`][qsimod.artifact],
[`qsimod.relations`][qsimod.relations], [`qsimod.validity`][qsimod.validity],
[`qsimod.transform`][qsimod.transform] and [`qsimod.pipeline`][qsimod.pipeline].  The
structural layer comprises [`qsimod.pauli`][qsimod.pauli] and [`qsimod.trotter`][qsimod.trotter].
The solving layer is [`qsimod.solving`][qsimod.solving], and the numerical realisation is
provided by [`qsimod.realise`][qsimod.realise].  The libraries of models and transformations are
[`qsimod.models`][qsimod.models] and [`qsimod.transformations`][qsimod.transformations]; the use
cases are collected in [`qsimod.usecases`][qsimod.usecases].

Importing [`qsimod.jax_setup`][qsimod.jax_setup] enables the 64-bit mode of JAX for the whole
process.
"""

from qsimod.artifact import (
    Artifact,
    ArtifactKind,
    ArtifactKindError,
    ConstraintOperator,
    HamiltonianModel,
    LocalSubspace,
    Sector,
    as_hamiltonian,
)
from qsimod.jax_setup import enable_x64, require_x64, x64_enabled
from qsimod.levels import AbstractionLevel
from qsimod.parameters import (
    AdmissibleSet,
    Binding,
    Bound,
    InequalityConstraint,
    Namespace,
    Parameter,
    ParameterSet,
)
from qsimod.pipeline import (
    BranchSummary,
    CompositionError,
    Edge,
    ErrorReport,
    ModelGraph,
    Pipeline,
    ResourceCost,
    StructuralApproximation,
)
from qsimod.relations import (
    Definition,
    Equation,
    ParameterRelation,
    RelationClassification,
    RelationKind,
)
from qsimod.scalar import Abs, DomainError, Interval, Scalar, Symbol, absolute, sqrt
from qsimod.structure import (
    Algebra,
    DegreeOfFreedom,
    DofRequirement,
    Lattice,
    LatticeGeometry,
    StructureMismatch,
    StructurePattern,
    StructureType,
    StructureTypeError,
    SymmetryDeclaration,
)
from qsimod.symbolic import OperatorSum, OpSymbol, SiteOperator, Term
from qsimod.transform import (
    Approximation,
    ApproximationKind,
    Exactness,
    ExactnessError,
    Transformation,
)
from qsimod.units import Dimension, from_hertz, from_kilohertz, to_hertz, to_kilohertz
from qsimod.validity import (
    ConditionOutcome,
    ConditionStatus,
    Conjunction,
    MuchLessThan,
    NonZero,
    Severity,
    ValidityCondition,
    ValidityReport,
)

__version__ = "0.1.0"

__all__ = [
    "Abs",
    "AbstractionLevel",
    "AdmissibleSet",
    "Algebra",
    "Approximation",
    "ApproximationKind",
    "Artifact",
    "ArtifactKind",
    "ArtifactKindError",
    "Binding",
    "Bound",
    "BranchSummary",
    "CompositionError",
    "ConditionOutcome",
    "ConditionStatus",
    "Conjunction",
    "ConstraintOperator",
    "Definition",
    "DegreeOfFreedom",
    "Dimension",
    "DofRequirement",
    "DomainError",
    "Edge",
    "Equation",
    "ErrorReport",
    "Exactness",
    "ExactnessError",
    "HamiltonianModel",
    "InequalityConstraint",
    "Interval",
    "Lattice",
    "LatticeGeometry",
    "LocalSubspace",
    "ModelGraph",
    "MuchLessThan",
    "Namespace",
    "NonZero",
    "OpSymbol",
    "OperatorSum",
    "Parameter",
    "ParameterRelation",
    "ParameterSet",
    "Pipeline",
    "RelationClassification",
    "RelationKind",
    "ResourceCost",
    "Scalar",
    "Sector",
    "Severity",
    "SiteOperator",
    "StructuralApproximation",
    "StructureMismatch",
    "StructurePattern",
    "StructureType",
    "StructureTypeError",
    "Symbol",
    "SymmetryDeclaration",
    "Term",
    "Transformation",
    "ValidityCondition",
    "ValidityReport",
    "__version__",
    "absolute",
    "as_hamiltonian",
    "enable_x64",
    "from_hertz",
    "from_kilohertz",
    "require_x64",
    "sqrt",
    "to_hertz",
    "to_kilohertz",
    "x64_enabled",
]
