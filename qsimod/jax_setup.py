"""Process-global JAX configuration.

Importing this module enables the 64-bit mode of JAX.  The setting is process-global and also
affects the JAX code of the importing program.  Every operator is built in ``complex128``; real
spectra, times and observables use ``float64``.
"""

from __future__ import annotations

import jax

__all__ = ["COMPLEX_DTYPE", "REAL_DTYPE", "enable_x64", "require_x64", "x64_enabled"]


def enable_x64() -> None:
    """Enable the 64-bit mode of JAX; the call is idempotent and process-global."""
    jax.config.update("jax_enable_x64", True)


def x64_enabled() -> bool:
    """Whether JAX is currently in 64-bit mode."""
    return bool(jax.config.jax_enable_x64)


def require_x64() -> None:
    """Raise if JAX is not in 64-bit mode.

    Raises:
        RuntimeError: if the 64-bit mode has been disabled after the import of the package.

    """
    if not x64_enabled():
        msg = (
            "qsimod requires JAX in 64-bit mode; something has called "
            "jax.config.update('jax_enable_x64', False) since qsimod was imported."
        )
        raise RuntimeError(msg)


enable_x64()

#: The dtype of every operator and propagator.
COMPLEX_DTYPE = "complex128"
#: The dtype of real spectra, times and observables.
REAL_DTYPE = "float64"
