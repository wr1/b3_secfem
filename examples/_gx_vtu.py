"""Write a gxbeam-format VTU (with per-cell ``mat_props`` and ``theta``).

Shared by `examples/cross_check_b3_gx.py` and `tests/test_cross_check.py` so
both engines (gxbeam_section and b3_secfem) ingest the exact same mesh and
per-cell material data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_gx_vtu(
    path: str | Path,
    coords: np.ndarray,
    quads: np.ndarray,
    mat_props: np.ndarray,
    theta: np.ndarray,
) -> Path:
    """Write a quad mesh as a gxbeam_section-ingestible VTU.

    Parameters
    ----------
    path : str or Path
        Output VTU path.
    coords : (n_nodes, 2) or (n_nodes, 3) float
        Node coordinates. A 2D array is padded with z=0.
    quads : (n_cells, 4) int
        Quad connectivity, CCW winding (viewed from +z).
    mat_props : (n_cells, 10) float
        Per-element material property row, gxbeam layout
        ``[E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho]``.
    theta : (n_cells,) float
        Per-element ply / fibre angle [rad].
    """
    import meshio

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        msg = f"coords must be 2D, got shape {coords.shape}"
        raise ValueError(msg)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])
    elif coords.shape[1] != 3:
        msg = f"coords must have 2 or 3 columns, got {coords.shape[1]}"
        raise ValueError(msg)

    quads = np.asarray(quads, dtype=np.int64)
    mat_props = np.asarray(mat_props, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    n_cells = quads.shape[0]
    if mat_props.shape != (n_cells, 10):
        msg = (
            f"mat_props shape {mat_props.shape} does not match "
            f"(n_cells, 10) = ({n_cells}, 10)"
        )
        raise ValueError(msg)
    if theta.shape != (n_cells,):
        msg = f"theta shape {theta.shape} does not match (n_cells,) = ({n_cells},)"
        raise ValueError(msg)

    m = meshio.Mesh(
        coords,
        cells=[("quad", quads)],
        cell_data={"mat_props": [mat_props], "theta": [theta]},
    )
    p = Path(path)
    meshio.write(p, m)
    return p
