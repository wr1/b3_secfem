"""Dolfinx-free geometry builders and the _meshlib file-writing shim."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from b3_secfem import B3SecfemDeprecationWarning
from b3_secfem.geometry import airfoil_with_web, i_beam, rectangle
from b3_secfem.mesh_io import orient_ccw, read_section_mesh


def test_geometry_source_has_no_fenics_import():
    import b3_secfem.geometry as geometry

    src = inspect.getsource(geometry)
    assert "dolfinx" not in src
    assert "mpi4py" not in src


def test_rectangle_round_trip(tmp_path: Path):
    mesh, info = rectangle(0.2, 0.1, 4, 2)
    assert mesh.cells.shape == (8, 4)
    assert info["A"] == pytest.approx(0.02)
    loaded = read_section_mesh(mesh.write(tmp_path / "rect.xdmf"))
    assert loaded.cells.shape == (8, 4)
    _, flip = orient_ccw(loaded.points, loaded.cells)
    assert not flip.any()


def test_i_beam_area_and_round_trip(tmp_path: Path):
    b, h, t_w, t_f = 0.1, 0.2, 0.01, 0.012
    mesh, info = i_beam(b, h, t_w, t_f, n_b=4, n_h=4, n_tw=2, n_tf=1)
    assert info["A"] == pytest.approx(2 * b * t_f + t_w * (h - 2 * t_f))
    loaded = read_section_mesh(mesh.write(tmp_path / "ibeam.xdmf"))
    assert loaded.cells.shape[0] == mesh.cells.shape[0]


def test_airfoil_with_web_tags_both_regions(tmp_path: Path):
    mesh, info = airfoil_with_web(n_chord=8, n_thick=2)
    assert set(mesh.cell_tags.tolist()) == {1, 2}
    assert info["n_cells"] == mesh.cells.shape[0]
    loaded = read_section_mesh(mesh.write(tmp_path / "web.xdmf"))
    assert loaded.cell_tags.tolist() == mesh.cell_tags.tolist()


def test_meshlib_shim_warns_and_writes(tmp_path: Path):
    from b3_secfem._meshlib import hollow_cylinder

    with pytest.warns(B3SecfemDeprecationWarning, match=r"_meshlib\.hollow_cylinder"):
        path, info = hollow_cylinder(tmp_path, 0.05, 0.005, 8, n_rad=2)
    assert path.name == "ring.xdmf"
    assert path.is_file()
    assert info["R_mid"] == pytest.approx(0.05)
    loaded = read_section_mesh(path)
    assert loaded.cells.shape == (16, 4)
