# The layers

The package is organised in layers.  The declaration layer holds the symbolic artifacts and
transformations, the structural layer reasons about operators without constructing them, the
solving layer finds parameter settings, and the realisation layer constructs dense operators for
numerical validation.  On top of these sit the libraries of models and transformations and the
use cases that assemble them.  For the physics, see the guide to each application model: [the
lattice Schwinger model](schwinger.md), [the Heisenberg magnet](heisenberg.md) and [the Ising
chain](ising.md).

| Layer | Modules | Constructs an operator | Imports a solver |
|---|---|---|---|
| Declaration | [`levels`][qsimod.levels], [`scalar`][qsimod.scalar], [`affine`][qsimod.affine], [`units`][qsimod.units], [`structure`][qsimod.structure], [`symbolic`][qsimod.symbolic], [`normal_form`][qsimod.normal_form], [`parameters`][qsimod.parameters], [`artifact`][qsimod.artifact], [`relations`][qsimod.relations], [`validity`][qsimod.validity], [`transform`][qsimod.transform], [`pipeline`][qsimod.pipeline] | no | no |
| Structural | [`pauli`][qsimod.pauli], [`trotter`][qsimod.trotter] | no | no |
| Solving | [`solving`][qsimod.solving] | no | yes, and only here |
| Realisation | [`realise`][qsimod.realise], [`jax_setup`][qsimod.jax_setup] | yes | no |
| Model library | [`models`][qsimod.models]: [`application`][qsimod.models.application], [`intermediate`][qsimod.models.intermediate], [`hardware`][qsimod.models.hardware], with [`names`][qsimod.models.names] and the shared parts of a family of models, [`gauge`][qsimod.models.gauge] and [`magnetism`][qsimod.models.magnetism] | no | no |
| Transformation library | [`transformations`][qsimod.transformations] | no | no |
| Use cases | [`usecases`][qsimod.usecases]: [`schwinger`][qsimod.usecases.schwinger], [`heisenberg`][qsimod.usecases.heisenberg], [`ising`][qsimod.usecases.ising], over [`base`][qsimod.usecases.base] | no | no |

Importing any module enables the 64-bit mode of JAX for the whole process, in
[`qsimod.jax_setup`][qsimod.jax_setup].
