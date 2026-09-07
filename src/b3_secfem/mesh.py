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
        # dolfinx names its grid "mesh"; meshio names it "Grid". The docstring
        # has always claimed both are readable, but only the dolfinx name was
        # ever tried, so a meshio-written file failed with a bare
        # "<Grid> with name 'mesh' not found".
        mesh = None
        for grid in ("mesh", "Grid"):
            try:
                mesh = xf.read_mesh(name=grid)
                break
            except (RuntimeError, KeyError):
                continue
        if mesh is None:
            msg = f"no readable <Grid> (tried 'mesh', 'Grid') in {path}"
            raise RuntimeError(msg)
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
        cell_tags = None
        for name in ("cell_tags", "mesh_tags", "Cell tags"):
            try:
                cell_tags = xf.read_meshtags(mesh, name=name)
                break
            except (RuntimeError, KeyError):
                continue
    if cell_tags is None:
        cell_tags = _cell_tags_via_meshio(path, mesh)
    return mesh, cell_tags


def _cell_tags_via_meshio(path: str | Path, mesh: Any):
    """Recover cell tags from a meshio-written XDMF.

    dolfinx stores tags as a second ``<Grid>``; meshio stores them as an
    ``<Attribute>`` inside the single grid, which ``read_meshtags`` cannot see.
    So the geometry is read by dolfinx and the tags by meshio.

    The two disagree on cell order — ``read_mesh`` permutes — so the file-order
    tags are gathered through ``topology.original_cell_index``. Skipping that
    step attaches materials to the wrong cells, silently.
    """
    try:
        import meshio
        import numpy as _np
        from dolfinx.mesh import meshtags
    except ImportError:
        return None

    try:
        m = meshio.read(str(path))
    except Exception:
        return None

    values = None
    for name, blocks in (m.cell_data or {}).items():
        if "tag" in name.lower():
            values = _np.concatenate([_np.asarray(b).ravel() for b in blocks])
            break
    if values is None:
        return None

    dim = mesh.topology.dim
    n = mesh.topology.index_map(dim).size_local
    perm = mesh.topology.original_cell_index[:n]
    ordered = _np.asarray(values, dtype=_np.int32)[perm]
    mesh.topology.create_connectivity(dim, dim)
    mt = meshtags(mesh, dim, _np.arange(n, dtype=_np.int32), ordered)
    mt.name = "cell_tags"
    return mt


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


def write_quad_xdmf(
    path: str | Path,
    coords: np.ndarray,
    quads: np.ndarray,
    cell_tags: np.ndarray | None = None,
) -> np.ndarray:
    """Write a quad mesh straight from arrays — no FEM backend required.

    ``write_xdmf`` needs a live dolfinx mesh, which makes producing an input
    file require the very backend the file is meant to feed. This writes the
    same XDMF through meshio, so a caller can mesh a section without dolfinx
    installed and hand the result to either backend: the fenicsx path reads it
    with ``read_xdmf`` (which already accepts meshio-written files) and the
    mfem path reads it with meshio directly.

    Cells are written in **input order**. That matters: ``dmesh.create_mesh``
    permutes cells internally, so a dolfinx round-trip has to gather tags
    through ``topology.original_cell_index`` or materials land on the wrong
    cells. Writing in input order removes that failure mode rather than
    handling it.

    Winding is corrected in place (shoelace, same convention as the rest of the
    package) and the corrected connectivity is returned.
    """
    import meshio

    coords = np.asarray(coords, dtype=np.float64)
    quads = np.asarray(quads).copy()

    p = coords[quads]
    shoelace = (
        p[:, 0, 0] * p[:, 1, 1]
        - p[:, 1, 0] * p[:, 0, 1]
        + p[:, 1, 0] * p[:, 2, 1]
        - p[:, 2, 0] * p[:, 1, 1]
        + p[:, 2, 0] * p[:, 3, 1]
        - p[:, 3, 0] * p[:, 2, 1]
        + p[:, 3, 0] * p[:, 0, 1]
        - p[:, 0, 0] * p[:, 3, 1]
    )
    flip = shoelace < 0
    if flip.any():
        quads[flip] = quads[flip][:, [0, 3, 2, 1]]

    # meshio wants 3D points; the section lives in z = 0.
    pts = np.zeros((coords.shape[0], 3), dtype=np.float64)
    pts[:, :2] = coords

    cell_data = None
    if cell_tags is not None:
        cell_data = {"cell_tags": [np.asarray(cell_tags, dtype=np.int32)]}

    meshio.write(
        str(path),
        meshio.Mesh(points=pts, cells=[("quad", quads)], cell_data=cell_data),
        file_format="xdmf",
    )
    return quads


def solver_cell_tags(result: Any, mesh_path: str | Path) -> np.ndarray:
    """Per-cell tags in the SOLVER's cell order, for either backend.

    Callers index recovered strain/stress arrays with these, so file order is
    not good enough: the fenicsx reader permutes cells, the mfem loader does
    not. Getting this wrong pairs each cell's strain with another cell's
    material and allowables, with nothing to show for it in the output.
    """
    backend = getattr(result, "backend", "fenicsx")

    if backend == "mfem":
        # _load_mfem_mesh preserves input cell order, so file order IS solver
        # order and meshio can read the tags without dolfinx.
        import meshio

        m = meshio.read(str(mesh_path))
        for name, blocks in (m.cell_data or {}).items():
            if "tag" in name.lower():
                return np.concatenate([np.asarray(b).ravel() for b in blocks]).astype(
                    int
                )
        msg = f"no cell tags found in {mesh_path}"
        raise ValueError(msg)

    _, cell_tags = read_xdmf(mesh_path)
    if cell_tags is None:
        msg = f"no cell tags found in {mesh_path}"
        raise ValueError(msg)
    return np.asarray(cell_tags.values).astype(int)


def cell_centroids(mesh: Any) -> np.ndarray:
    """Per-cell centroids, in the mesh's own cell order, for either backend.

    Returns ``(n_cells, 2)``. Callers pair these with per-cell result arrays
    (strains, stresses, tags), so the ordering must be the solver's own — not
    the file order. Both branches below read the solver's mesh directly, which
    is what makes that true.
    """
    kind = f"{type(mesh).__module__}.{type(mesh).__name__}"

    if "dolfinx" in kind:
        import numpy as _np
        from dolfinx.mesh import compute_midpoints

        dim = mesh.topology.dim
        n = mesh.topology.index_map(dim).size_local
        return _np.asarray(
            compute_midpoints(mesh, dim, _np.arange(n, dtype=_np.int32))
        )[:, :2]

    if "mfem" in kind:
        import numpy as _np

        n = mesh.GetNE()
        verts = _np.asarray(mesh.GetVertexArray())
        out = _np.zeros((n, 2), dtype=float)
        for e in range(n):
            vids = mesh.GetElementVertices(e)
            out[e] = verts[list(vids), :2].mean(axis=0)
        return out

    msg = f"cannot compute centroids for mesh of type {kind!r}"
    raise TypeError(msg)


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
        p[:, 0, 0] * p[:, 1, 1]
        - p[:, 1, 0] * p[:, 0, 1]
        + p[:, 1, 0] * p[:, 2, 1]
        - p[:, 2, 0] * p[:, 1, 1]
        + p[:, 2, 0] * p[:, 3, 1]
        - p[:, 3, 0] * p[:, 2, 1]
        + p[:, 3, 0] * p[:, 0, 1]
        - p[:, 0, 0] * p[:, 3, 1]
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
