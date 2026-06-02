"""Backend registry for pluggable FEM engines (fenicsx, mfem, ...)."""

from __future__ import annotations

from typing import Any

_BACKENDS: dict[str, Any] = {}


def register_backend(name: str, module: Any) -> None:
    """Register a backend module under a short name."""
    _BACKENDS[name] = module


def get_backend(name: str):
    """Return the backend module for name ("fenicsx" or "mfem").

    The module must provide a callable ``solve(inp: SectionInput) -> SectionResult``.
    """
    if name in _BACKENDS:
        return _BACKENDS[name]

    if name == "fenicsx":
        from . import fenicsx as mod  # type: ignore
        register_backend(name, mod)
        return mod
    if name == "mfem":
        from . import mfem as mod  # type: ignore
        register_backend(name, mod)
        return mod

    msg = f"unknown backend {name!r}; supported: fenicsx, mfem"
    raise ValueError(msg)


__all__ = ["get_backend", "register_backend"]
