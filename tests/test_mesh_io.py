"""Neutral mesh reader: one winding fix, one tag rule, dolfinx two-grid files."""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from b3_secfem import B3SecfemDeprecationWarning
from b3_secfem.mesh_io import (
    SectionMesh,
    orient_ccw,
    read_section_mesh,
    write_section_mesh,
)


def _square_quads():
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    # second cell is clockwise
    cells = np.array([[0, 1, 2, 3], [1, 4, 5, 2][::-1]], dtype=np.int64)
    # [1, 4, 5, 2] is CCW (BL BR TR TL). reversed is clockwise.
    cells = np.array([[0, 1, 2, 3], [2, 5, 4, 1]], dtype=np.int64)
    return points, cells


def _one_tri():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 2, 1]], dtype=np.int64)  # clockwise
    return points, cells


def test_orient_ccw_flips_clockwise_once():
    points, cells = _square_quads()
    fixed, flip = orient_ccw(points, cells)
    assert flip.tolist() == [False, True]
    again, flip2 = orient_ccw(points, fixed)
    assert not flip2.any()
    assert np.array_equal(again, fixed)

    t_points, t_cells = _one_tri()
    t_fixed, t_flip = orient_ccw(t_points, t_cells)
    assert t_flip.tolist() == [True]
    _, t_flip2 = orient_ccw(t_points, t_fixed)
    assert not t_flip2.any()


def test_quad_and_tri_round_trip(tmp_path: Path):
    points, cells = _square_quads()
    tags = np.array([1, 2], dtype=np.int32)
    dest = write_section_mesh(
        tmp_path / "quad.xdmf",
        SectionMesh(points, cells, "quad", tags, {}, None),
    )
    loaded = read_section_mesh(dest)
    assert loaded.cell_type == "quad"
    assert loaded.points.shape == (6, 2)
    assert np.array_equal(loaded.cell_tags, tags)
    _, flip = orient_ccw(loaded.points, loaded.cells)
    assert not flip.any()

    t_points, t_cells = _one_tri()
    t_dest = write_section_mesh(
        tmp_path / "tri.xdmf",
        SectionMesh(t_points, t_cells, "triangle", None, {}, None),
    )
    t_loaded = read_section_mesh(t_dest)
    assert t_loaded.cell_type == "triangle"
    assert t_loaded.cell_tags is None
    _, t_flip = orient_ccw(t_loaded.points, t_loaded.cells)
    assert not t_flip.any()


def test_legacy_region_key_warns(tmp_path: Path):
    import meshio

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = np.array([[0, 1, 2, 3]])
    path = tmp_path / "region.xdmf"
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [("quad", cells)],
            cell_data={"region": [np.array([7], dtype=np.int32)]},
        ),
    )
    with pytest.warns(B3SecfemDeprecationWarning, match="region"):
        loaded = read_section_mesh(path)
    assert loaded.cell_tags.tolist() == [7]


def test_tag_length_mismatch_raises(tmp_path: Path):
    points, cells = _one_tri()
    bad = np.array([1, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="length"):
        write_section_mesh(
            tmp_path / "bad.xdmf",
            SectionMesh(points, cells, "triangle", bad, {}, None),
        )


def test_gxbeam_vtu_keeps_cell_data_and_winding(tmp_path: Path):
    import meshio

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    # Clockwise quad. Reader must flip it and keep per-cell arrays in cell order.
    cells = np.array([[0, 3, 2, 1]], dtype=np.int64)
    props = np.arange(10, dtype=np.float64).reshape(1, 10)
    theta = np.array([0.25])
    path = tmp_path / "gx.vtu"
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [("quad", cells)],
            cell_data={"mat_props": [props], "theta": [theta]},
        ),
    )
    loaded = read_section_mesh(path)
    assert loaded.cell_type == "quad"
    assert np.allclose(loaded.cell_data["mat_props"], props)
    assert np.allclose(np.ravel(loaded.cell_data["theta"]), theta)
    _, flip = orient_ccw(loaded.points, loaded.cells)
    assert not flip.any()


def test_swap23_columns_relabels_axes():
    from b3_secfem.mesh import _swap23_columns

    row = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.2, 0.3, 0.4, 9.0]])
    out = _swap23_columns(row)
    assert out[0].tolist() == pytest.approx(
        [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 0.3, 0.2, 0.6, 9.0]
    )


def test_two_grid_dolfinx_layout(tmp_path: Path):
    """Hand-written dolfinx two-grid XDMF: mesh grid plus a cell_tags grid."""
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
    tags = np.array([[4]], dtype=np.int32)
    h5_path = tmp_path / "two.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("/Mesh/mesh/geometry", data=points)
        handle.create_dataset("/Mesh/mesh/topology", data=cells)
        handle.create_dataset("/MeshTags/cell_tags/Values", data=tags)
    xdmf = tmp_path / "two.xdmf"
    xdmf.write_text(
        """<?xml version="1.0"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="Quadrilateral" NumberOfElements="1" NodesPerElement="4">
        <DataItem Dimensions="1 4" Format="HDF">two.h5:/Mesh/mesh/topology</DataItem>
      </Topology>
      <Geometry GeometryType="XY">
        <DataItem Dimensions="4 2" Format="HDF">two.h5:/Mesh/mesh/geometry</DataItem>
      </Geometry>
    </Grid>
    <Grid Name="cell_tags" GridType="Uniform">
      <Attribute Name="cell_tags" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="1 1" Format="HDF">two.h5:/MeshTags/cell_tags/Values</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", B3SecfemDeprecationWarning)
        loaded = read_section_mesh(xdmf)
    assert loaded.cell_tags.tolist() == [4]
    assert loaded.points.shape == (4, 2)
    assert loaded.cells.shape == (1, 4)
