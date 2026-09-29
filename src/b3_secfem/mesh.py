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

    Deprecated. Returns (mesh, cell_tags). cell_tags may be None.
    """
    from ._deprecation import warn_deprecated
    from .backends.fenicsx.io import read_xdmf as _read

    warn_deprecated(
        "mesh.read_xdmf",
        "b3_secfem.backends.fenicsx.io.read_xdmf",
        stacklevel=2,
    )
    return _read(path, comm=comm)


def write_xdmf(
    path: str | Path,
    mesh: Any,
    cell_tags: Any | None = None,
) -> None:
    """Write a dolfinx mesh (and optional cell tags) to XDMF.

    Deprecated. Prefer :func:`b3_secfem.mesh_io.write_section_mesh` for a
    file both backends read.
    """
    from ._deprecation import warn_deprecated
    from .backends.fenicsx.io import write_xdmf as _write

    warn_deprecated(
        "mesh.write_xdmf",
        "b3_secfem.mesh_io.write_section_mesh",
        stacklevel=2,
    )
    _write(path, mesh, cell_tags=cell_tags)


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

    Winding is corrected (see :func:`b3_secfem.mesh_io.orient_ccw`) and the
    corrected connectivity is returned.
    """
    from .mesh_io import SectionMesh, orient_ccw, write_section_mesh

    coords = np.asarray(coords, dtype=np.float64)
    cells, _flip = orient_ccw(coords, np.asarray(quads))
    tags = None if cell_tags is None else np.asarray(cell_tags, dtype=np.int32)
    write_section_mesh(
        path,
        SectionMesh(
            points=coords[:, :2],
            cells=cells,
            cell_type="quad",
            cell_tags=tags,
            cell_data={},
            source=None,
        ),
    )
    return cells


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

    from .backends.fenicsx.io import read_xdmf as _read_xdmf

    _, cell_tags = _read_xdmf(mesh_path)
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


def from_gxbeam_vtu(path: str | Path, *, swap23: bool = False) -> dict[str, Any]:
    """Ingest a gxbeam_section quad mesh from a VTU file.

    Deprecated: returns a dict with a dolfinx mesh. Prefer
    :func:`b3_secfem.mesh_io.read_section_mesh`, which keeps ``mat_props`` and
    ``theta`` on :attr:`SectionMesh.cell_data`.

    ``mat_props`` columns are read as
    ``[E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho]``. ``swap23=True``
    relabels principal axes 2 and 3 (including G and the major Poisson ratio).

    Returns a dict::

        {
            "mesh":          dolfinx.mesh.Mesh    # 2D quad mesh
            "n_cells":       int
            "mat_props":     np.ndarray (n_cells, 10) or None
            "theta":         np.ndarray (n_cells,)   or None     [rad]
        }
    """
    from ._deprecation import warn_deprecated
    from .mesh_io import read_section_mesh

    warn_deprecated(
        "mesh.from_gxbeam_vtu",
        "b3_secfem.mesh_io.read_section_mesh",
        stacklevel=2,
    )
    loaded = read_section_mesh(path)
    if loaded.cell_type != "quad":
        msg = f"no quad cells in {path}"
        raise ValueError(msg)

    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    # CCW (BL, BR, TR, TL) -> dolfinx tensor-product (BL, TL, BR, TR).
    quads_dolfinx = loaded.cells[:, [0, 3, 1, 2]]
    domain = _ufl_quad_domain()
    dolfinx_mesh = dmesh.create_mesh(
        MPI.COMM_WORLD, quads_dolfinx, domain, np.asarray(loaded.points)
    )
    n_cells = loaded.cells.shape[0]
    mat_props = _cell_array(loaded.cell_data, "mat_props", n_cells)
    theta = _cell_array(loaded.cell_data, "theta", n_cells)
    if mat_props is not None and swap23:
        mat_props = _swap23_columns(mat_props)
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


def _cell_array(cell_data: dict, key: str, n_cells: int) -> np.ndarray | None:
    """Per-cell array from a :class:`SectionMesh`, or None when the key is absent."""
    if key not in cell_data:
        return None
    arr = np.asarray(cell_data[key])
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.shape[0] != n_cells:
        msg = f"{key}: expected {n_cells} rows, got {arr.shape[0]}"
        raise ValueError(msg)
    return arr


def _swap23_columns(mat_props: np.ndarray) -> np.ndarray:
    """Relabel principal 2 ↔ 3 on gxbeam ``mat_props`` columns.

    Column order is ``[E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho]``.
    ``nu32 = nu23 * E3 / E2`` becomes the new ``nu23``.
    """
    arr = np.array(mat_props, dtype=np.float64, copy=True)
    if arr.ndim != 2 or arr.shape[1] != 10:
        msg = f"mat_props must be (n, 10), got {arr.shape}"
        raise ValueError(msg)
    e2 = arr[:, 1].copy()
    e3 = arr[:, 2].copy()
    g12 = arr[:, 3].copy()
    g13 = arr[:, 4].copy()
    nu12 = arr[:, 6].copy()
    nu13 = arr[:, 7].copy()
    nu23 = arr[:, 8].copy()
    arr[:, 1] = e3
    arr[:, 2] = e2
    arr[:, 3] = g13
    arr[:, 4] = g12
    arr[:, 8] = nu23 * e3 / e2
    arr[:, 6] = nu13
    arr[:, 7] = nu12
    return arr
