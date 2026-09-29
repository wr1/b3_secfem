"""Dolfinx mesh construction and the legacy XDMF file helpers.

``mesh_from_section`` is the path ``solve`` uses. ``read_xdmf`` / ``write_xdmf``
remain for callers that already hold a dolfinx mesh; the names on
``b3_secfem.mesh`` warn and forward here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...mesh_io import SectionMesh


def mesh_from_section(section: SectionMesh) -> tuple[Any, Any]:
    """Build a dolfinx mesh on ``MPI.COMM_SELF`` from a ``SectionMesh``.

    Quad connectivity is the counter-clockwise file order. ``create_mesh``
    wants tensor-product order, so quads are permuted ``[0, 3, 1, 2]``.
    Cell tags are returned in dolfinx cell order via ``original_cell_index``.
    """
    import basix.ufl
    import ufl
    from dolfinx import mesh as dmesh
    from dolfinx.mesh import meshtags
    from mpi4py import MPI

    points = np.asarray(section.points, dtype=np.float64)
    cells = np.asarray(section.cells, dtype=np.int64)
    if section.cell_type == "quad":
        conn = cells[:, [0, 3, 1, 2]]
        element = basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,))
    else:
        conn = cells
        element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    mesh = dmesh.create_mesh(MPI.COMM_SELF, conn, ufl.Mesh(element), points)
    dim = mesh.topology.dim
    mesh.topology.create_connectivity(dim, dim)
    if section.cell_tags is None:
        return mesh, None
    n = mesh.topology.index_map(dim).size_local
    perm = np.asarray(mesh.topology.original_cell_index[:n])
    values = np.asarray(section.cell_tags, dtype=np.int32)[perm]
    tags = meshtags(mesh, dim, np.arange(n, dtype=np.int32), values)
    tags.name = "cell_tags"
    return mesh, tags


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
        # dolfinx names its grid "mesh"; meshio names it "Grid".
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
        from dolfinx.mesh import meshtags
    except ImportError:
        return None

    try:
        loaded = meshio.read(str(path))
    except Exception:
        return None

    values = None
    for name, blocks in (loaded.cell_data or {}).items():
        if "tag" in name.lower():
            values = np.concatenate([np.asarray(b).ravel() for b in blocks])
            break
    if values is None:
        return None

    dim = mesh.topology.dim
    n = mesh.topology.index_map(dim).size_local
    perm = mesh.topology.original_cell_index[:n]
    ordered = np.asarray(values, dtype=np.int32)[perm]
    mesh.topology.create_connectivity(dim, dim)
    tags = meshtags(mesh, dim, np.arange(n, dtype=np.int32), ordered)
    tags.name = "cell_tags"
    return tags


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
