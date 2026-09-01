"""The transformation library, indexed by the effect of a transformation.

| Module | Effect | Exactness |
|---|---|---|
| `truncations` | an infinite degree of freedom is made finite | approximate, declared regime |
| `basis_changes` | the Hamiltonian is rewritten in different operators | exact |
| `encodings` | the Hamiltonian is rewritten in another algebra | exact on the encoded subspace |
| `reparametrisations` | the model is restated by different parameters | exact |
| `perturbative` | hardware knobs related to an effective theory | approximate, declared regime |
| `trotterisation` | a Hamiltonian to a product of unitaries | approximate, resource-controlled |

Every builder takes the two parameter namespaces it relates.  None of these modules imports a
solver: each declares equations, solved forms, validity conditions and, for a hardware model as
target, an admissible knob set.
"""

from qsimod.transformations.base import ModelTransformation, NamespacedTransformation
from qsimod.transformations.basis_changes import (
    JordanWignerToFermions,
    JordanWignerToQubits,
    ParticleHoleTransformation,
    carried_over,
    jordan_wigner_to_fermions,
    jordan_wigner_to_qubits,
    particle_hole_transformation,
    staggering_constant,
)
from qsimod.transformations.encodings import HardcoreBosonEncoding, hardcore_boson_encoding
from qsimod.transformations.perturbative import (
    DIPOLE_ENHANCEMENT,
    DipoleReduction,
    Reduction,
    SecondOrderPerturbationTheory,
    SuperexchangeReduction,
    dipole_boundary_field,
    dipole_coupling,
    dipole_detuning,
    dipole_longitudinal,
    dipole_reduction,
    dipole_transverse,
    dipole_validity,
    pole_sum,
    second_order_coupling,
    second_order_mass,
    second_order_perturbation_theory,
    superexchange_field,
    superexchange_longitudinal,
    superexchange_reduction,
    superexchange_transverse,
    superexchange_validity,
    superlattice_validity,
)
from qsimod.transformations.reparametrisations import (
    AnisotropyResolution,
    FieldResolution,
    Reparametrisation,
    anisotropy_resolution,
    field_resolution,
)
from qsimod.transformations.trotterisation import (
    INTERLEAVED_LAYER_NAMES,
    SuzukiTrotter,
    derived_layers_of,
    interleaved_layers_of,
    product_formula_from,
    suzuki_trotter,
)
from qsimod.transformations.truncations import QuantumLinkTruncation, quantum_link_truncation

__all__ = [
    "DIPOLE_ENHANCEMENT",
    "INTERLEAVED_LAYER_NAMES",
    "AnisotropyResolution",
    "DipoleReduction",
    "FieldResolution",
    "HardcoreBosonEncoding",
    "JordanWignerToFermions",
    "JordanWignerToQubits",
    "ModelTransformation",
    "NamespacedTransformation",
    "ParticleHoleTransformation",
    "QuantumLinkTruncation",
    "Reduction",
    "Reparametrisation",
    "SecondOrderPerturbationTheory",
    "SuperexchangeReduction",
    "SuzukiTrotter",
    "anisotropy_resolution",
    "carried_over",
    "derived_layers_of",
    "dipole_boundary_field",
    "dipole_coupling",
    "dipole_detuning",
    "dipole_longitudinal",
    "dipole_reduction",
    "dipole_transverse",
    "dipole_validity",
    "field_resolution",
    "hardcore_boson_encoding",
    "interleaved_layers_of",
    "jordan_wigner_to_fermions",
    "jordan_wigner_to_qubits",
    "particle_hole_transformation",
    "pole_sum",
    "product_formula_from",
    "quantum_link_truncation",
    "second_order_coupling",
    "second_order_mass",
    "second_order_perturbation_theory",
    "staggering_constant",
    "superexchange_field",
    "superexchange_longitudinal",
    "superexchange_reduction",
    "superexchange_transverse",
    "superexchange_validity",
    "superlattice_validity",
    "suzuki_trotter",
]
