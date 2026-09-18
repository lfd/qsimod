"""Q-SiMod: a typed metamodel for the model-driven engineering (MDE) of quantum simulation.

The package is organised in layers: the declaration layer (from [`qsimod.scalar`][qsimod.scalar]
to [`qsimod.pipeline`][qsimod.pipeline]), the structural layer ([`qsimod.pauli`][qsimod.pauli],
[`qsimod.trotter`][qsimod.trotter]), the solving layer ([`qsimod.solving`][qsimod.solving]) and
the numerical realisation ([`qsimod.realise`][qsimod.realise]), with the libraries
[`qsimod.models`][qsimod.models] and [`qsimod.transformations`][qsimod.transformations] and the
use cases [`qsimod.usecases`][qsimod.usecases] on top.  Importing the package enables the 64-bit
mode of JAX for the whole process ([`qsimod.jax_setup`][qsimod.jax_setup]).
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
