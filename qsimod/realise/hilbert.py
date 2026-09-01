"""Hilbert spaces, basis ordering and single-site operator matrices.

The following conventions are fixed in this module:

* **Basis ordering.**  The local basis index is the occupation number on every algebra; a boson
  with cutoff ``n_max`` has ``n_max + 1`` levels.
* **Tensor-factor order.**  Register site 0 is the leftmost, most significant factor,
  ``index = sum_j config[j] * prod_{k>j} dim[k]``.
* **Pauli matrices.**  ``Z = 2n - 1 = diag(-1, +1)``, ``X = [[0,1],[1,0]]``,
  ``Y = [[0,i],[-i,0]]``; hence ``n = (Z+1)/2``, ``sigma^+ = |1><0| = (X+iY)/2`` and
  ``XY = iZ``.  [`qsimod.pauli`][qsimod.pauli] uses the same conventions.
* **Spin-1/2.**  ``S^z = Z/2``; ``S^+-`` coincide with ``sigma^+-``.
* **Bosons.**  The occupation cutoff is set by a
  [`RealisationRequest`][qsimod.realise.hilbert.RealisationRequest], with default ``n_max = 2``.

Every operator is realised as a dense array.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from qsimod.jax_setup import COMPLEX_DTYPE, require_x64
from qsimod.structure import Algebra, StructureType
from qsimod.symbolic import OpSymbol

__all__ = [
    "HilbertSpace",
    "LocalSpace",
    "RealisationRequest",
    "local_matrix",
]


@dataclass(frozen=True)
class RealisationRequest:
    """The choices a numerical realisation requires in addition to the model.

    Attributes:
        boson_cutoff: the maximum occupation ``n_max`` kept on a bosonic site, so that a
            bosonic local space has ``n_max + 1`` levels.  The cutoff is ignored by algebras
            that are finite-dimensional.
        fermion_ordering: the roles whose sites carry a Jordan-Wigner string, in the order
            in which the string runs.  Defaults to every fermionic role, ascending by site.

    """

    boson_cutoff: int = 2
    fermion_ordering: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.boson_cutoff < 1:
            msg = f"boson_cutoff must be at least 1, got {self.boson_cutoff}"
            raise ValueError(msg)

    def __str__(self) -> str:
        return f"n_max={self.boson_cutoff}"


@dataclass(frozen=True)
class LocalSpace:
    """The local Hilbert space of one register position.

    Attributes:
        site: the register index.
        algebra: the algebra of the site.
        dimension: the number of levels kept.
        jordan_wigner_position: for a fermionic site, its position in the Jordan-Wigner
            ordering; ``None`` for every other algebra.  The string of a fermionic
            operator runs over the sites with a strictly smaller position.

    """

    site: int
    algebra: Algebra
    dimension: int
    jordan_wigner_position: int | None = None

    def __str__(self) -> str:
        return f"site {self.site}: {self.algebra.value}, dim {self.dimension}"


def _boson_matrix(symbol: OpSymbol, dimension: int) -> Array:
    """The bosonic number and ladder operators truncated at ``dimension - 1`` quanta."""
    occupations = jnp.arange(dimension, dtype=COMPLEX_DTYPE)
    if symbol is OpSymbol.NUMBER:
        return jnp.diag(occupations)
    if symbol is OpSymbol.ANNIHILATE:
        # a|n> = sqrt(n)|n-1>: the entries sit one step above the diagonal.
        return jnp.diag(jnp.sqrt(occupations[1:]), k=1)
    if symbol is OpSymbol.CREATE:
        return jnp.diag(jnp.sqrt(occupations[1:]), k=-1)
    msg = f"{symbol.value} is not a bosonic operator"
    raise ValueError(msg)


def _fermion_matrix(symbol: OpSymbol) -> Array:
    """The local factor of a fermionic operator; the parity string is added by the builder."""
    if symbol is OpSymbol.NUMBER:
        return jnp.diag(jnp.asarray([0.0, 1.0], dtype=COMPLEX_DTYPE))
    if symbol is OpSymbol.PARITY:
        return jnp.diag(jnp.asarray([1.0, -1.0], dtype=COMPLEX_DTYPE))
    if symbol is OpSymbol.ANNIHILATE:
        return jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=COMPLEX_DTYPE)
    if symbol is OpSymbol.CREATE:
        return jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE)
    msg = f"{symbol.value} is not a fermionic operator"
    raise ValueError(msg)


def _spin_half_matrix(symbol: OpSymbol) -> Array:
    """The spin-1/2 operators, with ``S^z = Z/2``, in the occupation-ordered basis."""
    if symbol is OpSymbol.SPIN_Z:
        return jnp.diag(jnp.asarray([-0.5, 0.5], dtype=COMPLEX_DTYPE))
    if symbol is OpSymbol.SPIN_PLUS:
        return jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE)
    if symbol is OpSymbol.SPIN_MINUS:
        return jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=COMPLEX_DTYPE)
    msg = f"{symbol.value} is not a spin-1/2 operator"
    raise ValueError(msg)


def _qubit_matrix(symbol: OpSymbol) -> Array:
    """The Pauli and occupation operators of a qubit, in the occupation-ordered basis."""
    if symbol is OpSymbol.PAULI_X:
        return jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE)
    if symbol is OpSymbol.PAULI_Y:
        return jnp.asarray([[0.0, 1.0j], [-1.0j, 0.0]], dtype=COMPLEX_DTYPE)
    if symbol is OpSymbol.PAULI_Z:
        return jnp.diag(jnp.asarray([-1.0, 1.0], dtype=COMPLEX_DTYPE))
    if symbol is OpSymbol.SIGMA_PLUS:
        return jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE)
    if symbol is OpSymbol.SIGMA_MINUS:
        return jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=COMPLEX_DTYPE)
    if symbol is OpSymbol.NUMBER:
        return jnp.diag(jnp.asarray([0.0, 1.0], dtype=COMPLEX_DTYPE))
    if symbol is OpSymbol.PARITY:
        return jnp.diag(jnp.asarray([1.0, -1.0], dtype=COMPLEX_DTYPE))
    msg = f"{symbol.value} is not a qubit operator"
    raise ValueError(msg)


#: The matrix builders of the two-level algebras.  Bosons, whose local dimension is set by the
#: realisation request, are handled separately.
_TWO_LEVEL_BUILDERS: dict[Algebra, Callable[[OpSymbol], Array]] = {
    Algebra.FERMION: _fermion_matrix,
    Algebra.SPIN_HALF: _spin_half_matrix,
    Algebra.QUBIT: _qubit_matrix,
}


def local_matrix(symbol: OpSymbol, algebra: Algebra, dimension: int) -> Array:
    """The matrix of one primitive operator on the local space of one site.

    Args:
        symbol: the primitive operator.
        algebra: the algebra of the site.
        dimension: the local dimension of the site.

    Returns:
        A ``(dimension, dimension)`` complex array.

    Raises:
        ValueError: if the operator is not defined on that algebra, or the algebra has
            no finite-dimensional representation.

    """
    require_x64()
    if algebra is Algebra.GAUGE_LINK_U1:
        msg = (
            "an untruncated U(1) gauge link has no finite-dimensional representation; "
            "apply the quantum-link truncation before realising"
        )
        raise ValueError(msg)
    if algebra is Algebra.BOSON:
        return _boson_matrix(symbol, dimension)
    builder = _TWO_LEVEL_BUILDERS.get(algebra)
    if builder is None:
        msg = f"no realisation rule for algebra {algebra}"
        raise ValueError(msg)
    return builder(symbol)


@dataclass(frozen=True)
class HilbertSpace:
    """A tensor product of local spaces, with an explicit basis ordering.

    Attributes:
        spaces: the local spaces, ascending by register site.  Site 0 is the leftmost
            tensor factor.
        request: the realisation request from which the space was constructed, retained for
            reports.

    """

    spaces: tuple[LocalSpace, ...]
    request: RealisationRequest = field(default_factory=RealisationRequest)

    @classmethod
    def of(
        cls,
        structure: StructureType,
        request: RealisationRequest | None = None,
    ) -> HilbertSpace:
        """Construct the Hilbert space of a structural type.

        Args:
            structure: the structural type of the model.
            request: the cutoffs and orderings; defaults are used if omitted.

        Returns:
            The Hilbert space.

        Raises:
            ValueError: if any degree of freedom has no finite-dimensional representation.

        """
        request = request or RealisationRequest()
        offenders = [
            dof.role
            for dof in structure.degrees_of_freedom
            if not dof.algebra.is_finite_dimensional
        ]
        if offenders:
            msg = (
                f"cannot realise {structure.name or 'this structure'}: degree(s) of freedom "
                f"{offenders} have no finite-dimensional representation"
            )
            raise ValueError(msg)

        fermionic_roles = request.fermion_ordering or tuple(
            dof.role for dof in structure.degrees_of_freedom if dof.algebra is Algebra.FERMION
        )
        fermionic_sites = sorted(
            site
            for dof in structure.degrees_of_freedom
            if dof.role in fermionic_roles
            for site in dof.sites
        )
        jordan_wigner = {site: index for index, site in enumerate(fermionic_sites)}

        spaces: list[LocalSpace] = []
        for site in structure.sites:
            algebra = structure.algebra_at(site)
            dimension = (
                request.boson_cutoff + 1 if algebra is Algebra.BOSON else algebra.local_dimension
            )
            if dimension is None:  # pragma: no cover -- guarded above
                msg = f"site {site} has no finite dimension"
                raise ValueError(msg)
            spaces.append(
                LocalSpace(
                    site=site,
                    algebra=algebra,
                    dimension=dimension,
                    jordan_wigner_position=jordan_wigner.get(site),
                )
            )
        return cls(tuple(spaces), request)

    # -- basic properties --------------------------------------------------------

    @property
    def sites(self) -> tuple[int, ...]:
        """The register sites, ascending."""
        return tuple(space.site for space in self.spaces)

    @property
    def dimensions(self) -> tuple[int, ...]:
        """The local dimension of each site, in site order."""
        return tuple(space.dimension for space in self.spaces)

    @property
    def dimension(self) -> int:
        """The total Hilbert-space dimension."""
        total = 1
        for space in self.spaces:
            total *= space.dimension
        return total

    def space_at(self, site: int) -> LocalSpace:
        """The local space at a register site.

        Raises:
            KeyError: if the site is not part of the space.

        """
        for space in self.spaces:
            if space.site == site:
                return space
        msg = f"register site {site} is not part of this Hilbert space ({self.sites})"
        raise KeyError(msg)

    def position_of(self, site: int) -> int:
        """The tensor-factor position of a register site."""
        return self.sites.index(site)

    # -- basis ordering ----------------------------------------------------------

    @property
    def strides(self) -> tuple[int, ...]:
        """The mixed-radix stride of each tensor factor; site 0 is the most significant."""
        strides: list[int] = []
        running = 1
        for space in reversed(self.spaces):
            strides.append(running)
            running *= space.dimension
        return tuple(reversed(strides))

    def configuration(self, index: int) -> tuple[int, ...]:
        """The occupation configuration of a basis index, in site order.

        Raises:
            IndexError: if the index is out of range.

        """
        if not 0 <= index < self.dimension:
            msg = f"basis index {index} out of range for dimension {self.dimension}"
            raise IndexError(msg)
        configuration: list[int] = []
        remainder = index
        for stride in self.strides:
            configuration.append(remainder // stride)
            remainder %= stride
        return tuple(configuration)

    def index_of(self, configuration: Sequence[int]) -> int:
        """The basis index of an occupation configuration given in site order.

        Raises:
            ValueError: if the configuration has the wrong length or an out-of-range
                occupation.

        """
        if len(configuration) != len(self.spaces):
            msg = (
                f"configuration has {len(configuration)} entries but the space has "
                f"{len(self.spaces)} sites"
            )
            raise ValueError(msg)
        index = 0
        for occupation, stride, space in zip(configuration, self.strides, self.spaces, strict=True):
            if not 0 <= occupation < space.dimension:
                msg = (
                    f"occupation {occupation} at site {space.site} is outside "
                    f"[0, {space.dimension - 1}]"
                )
                raise ValueError(msg)
            index += occupation * stride
        return index

    def basis_state(self, configuration: Sequence[int]) -> Array:
        """The computational basis state of an occupation configuration."""
        vector = jnp.zeros((self.dimension,), dtype=COMPLEX_DTYPE)
        return vector.at[self.index_of(configuration)].set(1.0)

    def configurations(self) -> tuple[tuple[int, ...], ...]:
        """Every occupation configuration, in basis order; the whole basis is materialised."""
        return tuple(self.configuration(index) for index in range(self.dimension))

    def label(self, index: int) -> str:
        """A textual label of a basis index, for instance ``"|1 0 1 0 1>"``."""
        return "|" + " ".join(str(n) for n in self.configuration(index)) + ">"

    def identity(self) -> Array:
        """The identity operator on the whole space."""
        return jnp.eye(self.dimension, dtype=COMPLEX_DTYPE)

    def embed(self, site: int, matrix: Array) -> Array:
        """Embed a local matrix at ``site`` into the full space as a tensor factor."""
        return self.kron_of({site: matrix})

    def kron_of(self, factors: Mapping[int, Array]) -> Array:
        """The tensor product with ``factors`` at the given sites and the identity elsewhere."""
        result = jnp.asarray([[1.0 + 0.0j]], dtype=COMPLEX_DTYPE)
        for space in self.spaces:
            factor = factors.get(space.site)
            if factor is None:
                factor = jnp.eye(space.dimension, dtype=COMPLEX_DTYPE)
            result = jnp.kron(result, factor)
        return result

    @property
    def fermionic_sites(self) -> tuple[int, ...]:
        """The sites carrying a Jordan-Wigner position, in string order."""
        return tuple(
            space.site
            for space in sorted(
                (s for s in self.spaces if s.jordan_wigner_position is not None),
                key=lambda s: s.jordan_wigner_position or 0,
            )
        )

    def jordan_wigner_string(self, site: int) -> tuple[int, ...]:
        """The sites whose parity operator precedes a fermionic operator at ``site``.

        The result is empty for a non-fermionic site; otherwise it lists the fermionic sites
        with a strictly smaller Jordan-Wigner position.
        """
        space = self.space_at(site)
        if space.jordan_wigner_position is None:
            return ()
        return tuple(
            other.site
            for other in self.spaces
            if other.jordan_wigner_position is not None
            and other.jordan_wigner_position < space.jordan_wigner_position
        )

    def __str__(self) -> str:
        return (
            f"HilbertSpace: {len(self.spaces)} site(s), dimension {self.dimension} "
            f"({'x'.join(str(d) for d in self.dimensions)}), {self.request}"
        )

    def __repr__(self) -> str:
        return f"HilbertSpace(sites={self.sites}, dimension={self.dimension})"
