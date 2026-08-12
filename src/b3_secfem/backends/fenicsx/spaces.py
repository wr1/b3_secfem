"""Function-space construction for the cross-section problem.

We solve for a 3D displacement field u: Omega -> R^3 on a 2D mesh in
(x, y). Use a continuous Lagrange vector space (degree configurable,
default 2). The rotated 6x6 stiffness lives on a DG-0 tensor space so
each cell carries its own constitutive matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def make_displacement_space(mesh: Any, degree: int = 2) -> Any:
    """CG-degree vector function space for 3D displacement on a 2D mesh."""
    import basix.ufl
    from dolfinx import fem

    cell = mesh.basix_cell()
    e = basix.ufl.element("Lagrange", cell, degree, shape=(3,))
    return fem.functionspace(mesh, e)


def make_stiffness_space(mesh: Any) -> Any:
    """DG-0 tensor (6, 6) function space for the per-cell rotated stiffness."""
    import basix.ufl
    from dolfinx import fem

    cell = mesh.basix_cell()
    e = basix.ufl.element("DG", cell, 0, shape=(6, 6))
    return fem.functionspace(mesh, e)


def make_density_space(mesh: Any) -> Any:
    """DG-0 scalar space for per-cell mass density."""
    import basix.ufl
    from dolfinx import fem

    cell = mesh.basix_cell()
    e = basix.ufl.element("DG", cell, 0)
    return fem.functionspace(mesh, e)


def fill_per_cell_stiffness(Q_func: Any, C_per_cell: np.ndarray) -> None:
    """Populate a DG-0 (6, 6) Function from a per-cell (n_cells, 6, 6) array."""
    arr = Q_func.x.array
    # Layout: dolfinx flattens DG-0 tensor with last index fastest. For shape
    # (6, 6) on n_cells DG-0 cells the array length is 36 * n_cells.
    n_cells = C_per_cell.shape[0]
    if arr.size != 36 * n_cells:
        msg = (
            f"stiffness function has {arr.size} dofs but expected "
            f"{36 * n_cells} for {n_cells} cells"
        )
        raise ValueError(msg)
    arr[:] = C_per_cell.reshape(-1)
    Q_func.x.scatter_forward()


def fill_per_cell_density(rho_func: Any, rho_per_cell: np.ndarray) -> None:
    """Populate a DG-0 scalar density Function."""
    arr = rho_func.x.array
    n_cells = rho_per_cell.size
    if arr.size != n_cells:
        msg = f"rho function has {arr.size} dofs but expected {n_cells}"
        raise ValueError(msg)
    arr[:] = rho_per_cell
    rho_func.x.scatter_forward()
