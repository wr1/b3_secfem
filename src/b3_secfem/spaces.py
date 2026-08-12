"""Function-space helpers for the fenicsx backend.

Implementation lives under ``backends.fenicsx.spaces``. Re-exported here for
historical imports.
"""

from __future__ import annotations

from .backends.fenicsx.spaces import (
    fill_per_cell_density,
    fill_per_cell_stiffness,
    make_density_space,
    make_displacement_space,
    make_stiffness_space,
)

__all__ = [
    "fill_per_cell_density",
    "fill_per_cell_stiffness",
    "make_density_space",
    "make_displacement_space",
    "make_stiffness_space",
]
