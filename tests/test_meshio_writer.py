"""``write_quad_xdmf`` must give both backends the same mesh.

Meshing used to require dolfinx, because the only way to produce the section
XDMF was to build a dolfinx mesh and write it out — a FEM backend for a file
write. Worse, the file it produced could not be read by meshio at all ("Only
supports one grid"), so the mfem backend could never consume it.

These tests pin the replacement: geometry and, crucially, the tag-to-geometry
association, which is where a cell-ordering mistake hides. Cell ORDER is an
internal detail and deliberately not asserted; centroid -> tag is the contract.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAVE_DOLFINX = importlib.util.find_spec("dolfinx") is not None


def _mixed_winding_mesh():
    """A 3x2 quad grid with half the cells wound clockwise, each uniquely tagged."""
    nx, ny = 3, 2
    xs = np.linspace(0.0, 3.0, nx + 1)
    ys = np.linspace(0.0, 2.0, ny + 1)
    coords = np.array([[x, y] for y in ys for x in xs], dtype=float)
    quads, tags = [], []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            q = [n0, n1, n2, n3]
            if (i + j) % 2:
                q = [q[0], q[3], q[2], q[1]]  # clockwise
            quads.append(q)
            tags.append(10 + i + j)
    return coords, np.array(quads, dtype=np.int64), np.array(tags, dtype=np.int32)


def test_written_mesh_is_readable_by_meshio(tmp_path):
    """meshio must be able to read it back — the mfem backend loads it that way."""
    import meshio

    from b3_secfem.mesh import write_quad_xdmf

    coords, quads, tags = _mixed_winding_mesh()
    p = tmp_path / "sec.xdmf"
    write_quad_xdmf(p, coords, quads, cell_tags=tags)

    m = meshio.read(str(p))
    cells = next(cb.data for cb in m.cells if cb.type in ("quad", "quadrilateral"))
    assert cells.shape == (6, 4)

    read_tags = None
    for name, blocks in (m.cell_data or {}).items():
        if "tag" in name.lower():
            read_tags = np.concatenate([np.asarray(b).ravel() for b in blocks])
    assert read_tags is not None, "cell tags did not survive the round trip"
    # meshio preserves input order, so tags come back exactly as written.
    assert np.array_equal(read_tags.astype(int), tags.astype(int))


def test_winding_is_corrected(tmp_path):
    """Every written cell must be counter-clockwise, whatever went in."""
    from b3_secfem.mesh import write_quad_xdmf

    coords, quads, tags = _mixed_winding_mesh()
    fixed = write_quad_xdmf(tmp_path / "sec.xdmf", coords, quads, cell_tags=tags)

    p = coords[fixed]
    shoelace = (
        p[:, 0, 0] * p[:, 1, 1] - p[:, 1, 0] * p[:, 0, 1]
        + p[:, 1, 0] * p[:, 2, 1] - p[:, 2, 0] * p[:, 1, 1]
        + p[:, 2, 0] * p[:, 3, 1] - p[:, 3, 0] * p[:, 2, 1]
        + p[:, 3, 0] * p[:, 0, 1] - p[:, 0, 0] * p[:, 3, 1]
    )
    assert (shoelace > 0).all(), "some cells are still clockwise"


@pytest.mark.skipif(not _HAVE_DOLFINX, reason="needs dolfinx to read the file back")
def test_dolfinx_sees_the_same_centroid_to_tag_map(tmp_path):
    """The fenicsx path must agree with the file about which cell has which tag.

    ``read_mesh`` permutes cells, so the tags have to be gathered through
    ``original_cell_index``. Get that wrong and every material lands on the
    wrong geometry — with no error, and a plausible-looking answer.
    """
    from b3_secfem.mesh import cell_centroids, read_xdmf, write_quad_xdmf

    coords, quads, tags = _mixed_winding_mesh()
    p = tmp_path / "sec.xdmf"
    write_quad_xdmf(p, coords, quads, cell_tags=tags)

    mesh, cell_tags = read_xdmf(p)
    assert cell_tags is not None, "tags were lost on the dolfinx read path"

    mids = cell_centroids(mesh)
    got = sorted(
        (round(float(m[0]), 9), round(float(m[1]), 9), int(t))
        for m, t in zip(mids, cell_tags.values, strict=True)
    )

    # What the file says, independent of any solver's cell ordering.
    ref_mid = coords[quads].mean(axis=1)
    want = sorted(
        (round(float(m[0]), 9), round(float(m[1]), 9), int(t))
        for m, t in zip(ref_mid, tags, strict=True)
    )
    assert got == want
