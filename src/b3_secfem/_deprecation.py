"""One warning category for names removed in 0.3.0."""

from __future__ import annotations

import warnings


class B3SecfemDeprecationWarning(DeprecationWarning):
    """Deprecation emitted by b3_secfem shims. Removed in 0.3.0."""


def warn_deprecated(
    old: str,
    new: str,
    *,
    removal: str = "0.3.0",
    stacklevel: int = 3,
) -> None:
    """Warn that ``old`` is replaced by ``new``."""
    warnings.warn(
        f"{old} is deprecated and will be removed in {removal}; use {new}",
        B3SecfemDeprecationWarning,
        stacklevel=stacklevel,
    )
