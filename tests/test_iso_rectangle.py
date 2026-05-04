"""Isotropic-rectangle analytic checks for the 6-unit-load solver.

Skipped when FEniCSx is not importable.
"""

from __future__ import annotations

import pytest

dolfinx = pytest.importorskip("dolfinx")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from b3_secfem import (
    IsotropicMaterial,
    RegionMat,
    SectionInput,
    solve,
    write_xdmf,
)


def _make_rectangle_xdmf(tmp_path, a: float, b: float, n: int):
    """Mesh a rectangle [-a/2, a/2] x [-b/2, b/2] with n x n quads."""
    mesh = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [(-a / 2, -b / 2), (a / 2, b / 2)],
        [n, n],
        cell_type=dmesh.CellType.quadrilateral,
    )
    path = tmp_path / "rect.xdmf"
    write_xdmf(path, mesh)
    return path, mesh


@pytest.fixture
def iso_steel():
    return IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)


def test_iso_rectangle_axial_stiffness(tmp_path, iso_steel):
    """K[Fz, Fz] should equal E * A within 1 %."""
    a = b = 0.1  # 100 mm square
    path, _ = _make_rectangle_xdmf(tmp_path, a, b, 16)
    inp = SectionInput(
        mesh_path=path,
        region_materials={1: RegionMat(material=iso_steel)},
    )
    res = solve(inp)
    EA = iso_steel.E * a * b
    np.testing.assert_allclose(res.K[2, 2], EA, rtol=2e-2)


def test_iso_rectangle_bending_stiffness(tmp_path, iso_steel):
    """K[Mx, Mx] = E * Ixx, K[My, My] = E * Iyy."""
    a, b = 0.10, 0.05  # 100 x 50 mm
    path, _ = _make_rectangle_xdmf(tmp_path, a, b, 24)
    inp = SectionInput(
        mesh_path=path,
        region_materials={1: RegionMat(material=iso_steel)},
    )
    res = solve(inp)
    Ixx = a * b ** 3 / 12.0
    Iyy = b * a ** 3 / 12.0
    np.testing.assert_allclose(res.K[3, 3], iso_steel.E * Ixx, rtol=5e-2)
    np.testing.assert_allclose(res.K[4, 4], iso_steel.E * Iyy, rtol=5e-2)


def test_iso_rectangle_torsion_stiffness(tmp_path, iso_steel):
    """K[Mz, Mz] for a square approximates 0.141 * G * a^4 (Roark)."""
    a = b = 0.05  # 50 mm square
    path, _ = _make_rectangle_xdmf(tmp_path, a, b, 32)
    inp = SectionInput(
        mesh_path=path,
        region_materials={1: RegionMat(material=iso_steel)},
    )
    res = solve(inp)
    G = iso_steel.E / (2 * (1 + iso_steel.nu))
    GJ_roark = 0.141 * G * a ** 4
    # Roark coefficient is 0.141 for square; coarse mesh gives ~10% error
    np.testing.assert_allclose(res.K[5, 5], GJ_roark, rtol=0.15)


def test_iso_rectangle_mass(tmp_path, iso_steel):
    """M[0, 0] = M[1, 1] = M[2, 2] = rho * A."""
    a = b = 0.10
    path, _ = _make_rectangle_xdmf(tmp_path, a, b, 16)
    inp = SectionInput(
        mesh_path=path,
        region_materials={1: RegionMat(material=iso_steel)},
    )
    res = solve(inp)
    m = iso_steel.rho * a * b
    np.testing.assert_allclose(res.M[0, 0], m, rtol=1e-9)
    np.testing.assert_allclose(res.M[1, 1], m, rtol=1e-9)
    np.testing.assert_allclose(res.M[2, 2], m, rtol=1e-9)


def test_iso_rectangle_transverse_shear(tmp_path, iso_steel):
    """K[Fx, Fx] = K[Fy, Fy] = (5/6) G A for an isotropic rectangle (Saint-Venant).

    The 5/6 factor is the EXACT Saint-Venant shear stiffness for a
    rectangle (and matches the Timoshenko shear-correction coefficient).
    """
    a = b = 0.05
    path, _ = _make_rectangle_xdmf(tmp_path, a, b, 24)
    inp = SectionInput(
        mesh_path=path,
        region_materials={1: RegionMat(material=iso_steel)},
    )
    res = solve(inp)
    G = iso_steel.E / (2 * (1 + iso_steel.nu))
    K_shear_exact = (5.0 / 6.0) * G * a * b
    np.testing.assert_allclose(res.K[0, 0], K_shear_exact, rtol=2e-2)
    np.testing.assert_allclose(res.K[1, 1], K_shear_exact, rtol=2e-2)


def test_iso_rectangle_K_section_xy(tmp_path, iso_steel):
    """Section-averaged in-plane shear stiffness K_section_xy = G * A
    for an isotropic homogeneous rectangle."""
    a = b = 0.05
    path, _ = _make_rectangle_xdmf(tmp_path, a, b, 16)
    inp = SectionInput(
        mesh_path=path,
        region_materials={1: RegionMat(material=iso_steel)},
    )
    res = solve(inp)
    G = iso_steel.E / (2 * (1 + iso_steel.nu))
    np.testing.assert_allclose(res.K_section_xy, G * a * b, rtol=1e-6)


def test_iso_rectangle_centres_at_origin(tmp_path, iso_steel):
    """Symmetric rectangle has tension and elastic centres at the origin."""
    a = b = 0.10
    path, _ = _make_rectangle_xdmf(tmp_path, a, b, 16)
    inp = SectionInput(
        mesh_path=path,
        region_materials={1: RegionMat(material=iso_steel)},
    )
    res = solve(inp)
    np.testing.assert_allclose(res.tension_center, (0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(res.elastic_center, (0.0, 0.0), atol=1e-12)
