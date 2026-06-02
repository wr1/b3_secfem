"""Shared constants and helpers for all backends.

These are pure-Python (numpy) and independent of any FEM library.
"""

from __future__ import annotations

from typing import Any

STAGE1_MODES = (2, 3, 4, 5)  # Fz, Mx, My, Mz
STAGE2_MODES = (0, 1)  # Vx, Vy
ALL_MODES = (0, 1, 2, 3, 4, 5)


def get_backend_name_from_inp(inp: Any) -> str:
    """Return backend string from a SectionInput (or object with .backend)."""
    return getattr(inp, "backend", "fenicsx")
