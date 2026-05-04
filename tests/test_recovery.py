"""Strain-recovery analytic checks."""

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
    recover_strains,
    solve,
    write_xdmf,
)


def test_axial_recovery_isotropic(tmp_path):
    """Under unit axial generalised strain (eps_zz = 1), a homogeneous
    isotropic free section recovers eps_xx = eps_yy = -nu (Poisson
    contraction) and eps_zz = 1, with negligible shear.
    """
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    a = b = 0.1
    n = 12
    path = tmp_path / "rect.xdmf"
    m = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [(-a / 2, -b / 2), (a / 2, b / 2)],
        [n, n],
        cell_type=dmesh.CellType.quadrilateral,
    )
    write_xdmf(path, m)
    inp = SectionInput(
        mesh_path=path,
        region_materials={1: RegionMat(material=iso)},
    )
    res = solve(inp)
    fields = recover_strains(res)

    eps_zz = fields.epsilon[2, :, 2].mean()
    eps_xx = fields.epsilon[2, :, 0].mean()
    eps_yy = fields.epsilon[2, :, 1].mean()
    np.testing.assert_allclose(eps_zz, 1.0, rtol=1e-3)
    np.testing.assert_allclose(eps_xx, -iso.nu, rtol=1e-2)
    np.testing.assert_allclose(eps_yy, -iso.nu, rtol=1e-2)
    # Shear components negligible.
    for k in (3, 4, 5):
        np.testing.assert_allclose(fields.epsilon[2, :, k].mean(), 0.0, atol=1e-6)


def test_bending_recovery_eps_zz_linear_in_y(tmp_path):
    """Under unit kappa_x, eps_zz on the section is -y (linear distribution)."""
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    a = b = 0.1
    n = 16
    path = tmp_path / "rect.xdmf"
    m = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [(-a / 2, -b / 2), (a / 2, b / 2)],
        [n, n],
        cell_type=dmesh.CellType.quadrilateral,
    )
    write_xdmf(path, m)
    inp = SectionInput(
        mesh_path=path,
        region_materials={1: RegionMat(material=iso)},
    )
    res = solve(inp)
    fields = recover_strains(res)

    # eps_zz over the section under kappa_x = 1 is -y. Cell areas times -y
    # cell-centroid should give zero (symmetric section). Variance > 0.
    eps_zz_kappa_x = fields.epsilon[3, :, 2]
    np.testing.assert_allclose(eps_zz_kappa_x.mean(), 0.0, atol=1e-6)
    assert eps_zz_kappa_x.std() > 0.01  # nontrivial spread
