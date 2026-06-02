"""Mesh I/O for b3_secfem.

Two ingestion paths:

1. ``read_xdmf(path)``: native FEniCSx XDMF reader. Returns the dolfinx
   mesh plus the cell-tags MeshTags that map to ``region_materials``.
2. ``from_gxbeam_vtu(path)``: read a quad mesh produced by gxbeam_section
   (with per-element ``mat_props`` and ``theta`` arrays). Returns a
   dolfinx mesh plus per-cell numpy arrays suitable for the per-cell
   ``SectionInput`` path.

dolfinx is imported lazily so the rest of the package can be tested
without a working FEniCSx install.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


def read_xdmf(path: str | Path, comm=None) -> tuple[Any, Any]:
    """Read an XDMF mesh file produced by meshio or another dolfinx run.

    Returns (mesh, cell_tags). cell_tags may be None if the file does
    not carry tags.
    """
    from dolfinx import io
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD
    with io.XDMFFile(comm, str(path), "r") as xf:
        mesh = xf.read_mesh()
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
        cell_tags = None
        for name in ("cell_tags", "mesh_tags", "Cell tags"):
            try:
                cell_tags = xf.read_meshtags(mesh, name=name)
                break
            except (RuntimeError, KeyError):
                continue
    return mesh, cell_tags


def write_xdmf(
    path: str | Path,
    mesh: Any,
    cell_tags: Any | None = None,
) -> None:
    """Write a dolfinx mesh (and optional cell tags) to XDMF."""
    from dolfinx import io

    with io.XDMFFile(mesh.comm, str(path), "w") as xf:
        xf.write_mesh(mesh)
        if cell_tags is not None:
            xf.write_meshtags(cell_tags, mesh.geometry)


def from_gxbeam_vtu(path: str | Path) -> dict[str, Any]:
    """Ingest a gxbeam_section quad mesh from a VTU file.

    Returns a dict::

        {
            "mesh":          dolfinx.mesh.Mesh    # 2D quad mesh
            "n_cells":       int
            "mat_props":     np.ndarray (n_cells, 10) or None
            "theta":         np.ndarray (n_cells,)   or None     [rad]
        }

    gxbeam stores per-element material properties (10 floats) and ply angle
    (theta, radians). Quad winding must be CCW when viewed from +z; we
    validate and flip if necessary.
    """
    import meshio
    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    m = meshio.read(str(path))
    quads = None
    for cb in m.cells:
        if cb.type == "quad":
            quads = cb.data.astype(np.int64)
            break
    if quads is None:
        msg = f"no quad cells in {path}"
        raise ValueError(msg)

    nodes = m.points[:, :2].astype(np.float64)
    n_cells = quads.shape[0]

    # Validate CCW winding (signed shoelace area > 0); flip non-CCW quads.
    p = nodes[quads]
    sa = (
        p[:, 0, 0] * p[:, 1, 1] - p[:, 1, 0] * p[:, 0, 1]
        + p[:, 1, 0] * p[:, 2, 1] - p[:, 2, 0] * p[:, 1, 1]
        + p[:, 2, 0] * p[:, 3, 1] - p[:, 3, 0] * p[:, 2, 1]
        + p[:, 3, 0] * p[:, 0, 1] - p[:, 0, 0] * p[:, 3, 1]
    )
    flip = sa < 0
    if flip.any():
        quads[flip] = quads[flip][:, [0, 3, 2, 1]]

    # Convert CCW (BL, BR, TR, TL) -> dolfinx 0.10 tensor-product ordering
    # (BL, TL, BR, TR) by the permutation [0, 3, 1, 2].
    quads_dolfinx = quads[:, [0, 3, 1, 2]]

    domain = _ufl_quad_domain()
    dolfinx_mesh = dmesh.create_mesh(MPI.COMM_WORLD, quads_dolfinx, domain, nodes)

    cell_data = m.cell_data_dict if hasattr(m, "cell_data_dict") else {}
    mat_props = _extract_cell_array(cell_data, "mat_props", "quad", n_cells, 10)
    theta = _extract_cell_array(cell_data, "theta", "quad", n_cells, None)

    return {
        "mesh": dolfinx_mesh,
        "n_cells": n_cells,
        "mat_props": mat_props,
        "theta": theta,
    }


def _ufl_quad_domain():
    """UFL domain spec for a 2D quadrilateral mesh, geometry dim 2."""
    import basix.ufl
    import ufl

    e = basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,))
    return ufl.Mesh(e)


def _extract_cell_array(
    cell_data: dict, key: str, ctype: str, n_cells: int, expected_cols: int | None
) -> np.ndarray | None:
    """Pull a per-cell array out of meshio cell_data_dict; None if missing."""
    if key not in cell_data:
        return None
    arr = cell_data[key].get(ctype)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.shape[0] != n_cells:
        msg = f"{key}: expected {n_cells} rows, got {arr.shape[0]}"
        raise ValueError(msg)
    if expected_cols is not None and arr.ndim == 2 and arr.shape[1] != expected_cols:
        msg = f"{key}: expected {expected_cols} cols, got {arr.shape[1]}"
        raise ValueError(msg)
    return arr
