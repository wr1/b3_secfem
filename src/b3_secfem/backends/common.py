"""Shared constants and helpers for all backends.

These are pure-Python (numpy) and independent of any FEM library.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..rotation3d import rotate_stiffness_6x6

STAGE1_MODES = (2, 3, 4, 5)  # Fz, Mx, My, Mz
STAGE2_MODES = (0, 1)  # Vx, Vy
ALL_MODES = (0, 1, 2, 3, 4, 5)


def get_backend_name_from_inp(inp: Any) -> str:
    """Return backend string from a SectionInput (or object with .backend)."""
    return getattr(inp, "backend", "fenicsx")


def per_cell_arrays(
    inp: Any, n_cells: int, cell_tags: Any | None
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-cell rotated C (n,6,6) and rho (n,) from SectionInput.

    ``cell_tags`` is optional backend payload with ``.indices`` / ``.values``
    (dolfinx MeshTags) or a SimpleNamespace with the same fields (mfem).
    """
    C = np.zeros((n_cells, 6, 6))
    rho = np.zeros(n_cells)

    if inp.per_cell_material is not None:
        beta = (
            inp.per_cell_beta_deg
            if inp.per_cell_beta_deg is not None
            else np.zeros(n_cells)
        )
        alpha = (
            inp.per_cell_alpha_deg
            if inp.per_cell_alpha_deg is not None
            else np.zeros(n_cells)
        )
        for k, mat in enumerate(inp.per_cell_material):
            C[k] = rotate_stiffness_6x6(mat.C_local(), beta[k], alpha[k])
            rho[k] = mat.rho
        return C, rho

    if inp.region_materials is None:
        msg = "no per-cell or region material specification"
        raise ValueError(msg)

    default_tag = next(iter(inp.region_materials))
    tags_per_cell = np.full(n_cells, default_tag, dtype=np.int64)
    if cell_tags is not None:
        tags_per_cell[cell_tags.indices] = cell_tags.values
    elif len(inp.region_materials) > 1:
        msg = (
            "region_materials specifies multiple regions but mesh has no "
            "cell tags; either tag the mesh or supply per_cell_material"
        )
        raise ValueError(msg)

    # One rotate per unique region tag (not per cell).
    for tag, rm in inp.region_materials.items():
        mask = tags_per_cell == int(tag)
        if not mask.any():
            continue
        C_rot = rotate_stiffness_6x6(rm.material.C_local(), rm.beta_deg, rm.alpha_deg)
        C[mask] = C_rot
        rho[mask] = rm.material.rho

    missing = ~np.isin(tags_per_cell, list(inp.region_materials.keys()))
    if missing.any():
        bad = int(tags_per_cell[missing][0])
        msg = f"cell tag {bad} has no region material"
        raise KeyError(msg)
    return C, rho
