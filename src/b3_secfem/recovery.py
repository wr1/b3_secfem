"""Per-cell strain and stress field recovery for unit-load cases.

For each mode i, the total Voigt strain at z = 0 is

    eps_total^(i) = eps_z(d_1) + eps_xy(d_2)         [shear: i = 0, 1]
                  = eps_z(d_0) + eps_xy(d_1)         [other: i = 2..5]

For shear modes the d_1 field is the bending warping (i=4 -> Vx,
i=3 -> Vy). The "primary warping" Function returned by ``solve`` is
``d_2`` for shear modes and ``d_1`` for the others.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from .forms import d0_kinematic, total_strain_voigt
from .result import SectionResult
from .solver import ALL_MODES, STAGE2_MODES


class StrainField(BaseModel):
    """Per-cell Voigt strain and stress for each of the 6 unit-load cases."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    epsilon: np.ndarray  # (6, n_cells, 6) Voigt strain (engineering shears)
    sigma: np.ndarray    # (6, n_cells, 6) Voigt stress
    cell_areas: np.ndarray  # (n_cells,)


def recover_strains(result: SectionResult) -> StrainField:
    """Per-cell Voigt strain / stress for each unit-load case."""
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_vector

    if result.u_solutions is None or result.C_func is None or result.mesh is None:
        msg = "result lacks dolfinx state; pass keep_solutions=True to solve()"
        raise ValueError(msg)

    mesh = result.mesh
    Q_func = result.C_func
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cell_areas = _cell_areas(mesh)

    eps = np.zeros((6, n_cells, 6))
    sig = np.zeros((6, n_cells, 6))

    x = ufl.SpatialCoordinate(mesh)
    DG0 = fem.functionspace(mesh, ("DG", 0))
    v = ufl.TestFunction(DG0)
    C_arr = Q_func.x.array.reshape(n_cells, 6, 6)

    primary = result.u_solutions
    for i in ALL_MODES:
        if primary[i] is None:
            continue
        d0 = d0_kinematic(i, x)
        if i in STAGE2_MODES:
            # shear: d_1 is the corresponding bending warping
            bending_idx = 4 if i == 0 else 3
            d1 = primary[bending_idx]
            d2 = primary[i]
        else:
            d1 = primary[i]
            d2 = None
        eps_total = total_strain_voigt(i, d0, d1, d2)

        for k_voigt in range(6):
            f = fem.form(v * eps_total[k_voigt] * ufl.dx)
            b = assemble_vector(f)
            b.assemble()
            eps[i, :, k_voigt] = b.array.copy() / cell_areas

        for k in range(n_cells):
            sig[i, k, :] = C_arr[k] @ eps[i, k, :]

    return StrainField(epsilon=eps, sigma=sig, cell_areas=cell_areas)


def _cell_areas(mesh: Any) -> np.ndarray:
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_vector

    DG0 = fem.functionspace(mesh, ("DG", 0))
    v = ufl.TestFunction(DG0)
    f = fem.form(v * ufl.dx)
    b = assemble_vector(f)
    b.assemble()
    return b.array.copy()
