"""Unit tests for backends.common (no FEM deps)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from b3_secfem.backends.common import per_cell_arrays
from b3_secfem.config import RegionMat, SectionInput
from b3_secfem.materials import IsotropicMaterial, OrthotropicMaterial


def _iso(E=70e9, nu=0.33, rho=2700.0):
    return IsotropicMaterial(E=E, nu=nu, rho=rho)


def test_per_cell_material_path():
    mats = [_iso(), _iso(E=210e9, rho=7800.0)]
    n = 2
    inp = SectionInput(
        mesh_path="x.xdmf",
        per_cell_material=mats,
        per_cell_beta_deg=np.array([0.0, 15.0]),
        per_cell_alpha_deg=np.zeros(n),
    )
    C, rho = per_cell_arrays(inp, n, None)
    assert C.shape == (2, 6, 6)
    assert rho.tolist() == [2700.0, 7800.0]
    assert C[0, 0, 0] != C[1, 0, 0]


def test_region_materials_with_tags():
    soft = _iso(E=70e9, rho=2700.0)
    hard = _iso(E=210e9, rho=7800.0)
    inp = SectionInput(
        mesh_path="x.xdmf",
        region_materials={
            1: RegionMat(material=soft, beta_deg=0.0),
            2: RegionMat(material=hard, beta_deg=45.0, alpha_deg=0.0),
        },
    )
    tags = SimpleNamespace(indices=np.array([0, 1, 2]), values=np.array([1, 2, 1]))
    C, rho = per_cell_arrays(inp, 3, tags)
    assert rho.tolist() == [2700.0, 7800.0, 2700.0]
    np.testing.assert_allclose(C[0], C[2])
    assert not np.allclose(C[0], C[1])


def test_region_single_material_no_tags():
    inp = SectionInput(
        mesh_path="x.xdmf",
        region_materials={7: RegionMat(material=_iso())},
    )
    C, rho = per_cell_arrays(inp, 4, None)
    assert C.shape == (4, 6, 6)
    assert np.all(rho == 2700.0)
    np.testing.assert_allclose(C[0], C[3])


def test_region_multi_without_tags_errors():
    inp = SectionInput(
        mesh_path="x.xdmf",
        region_materials={
            1: RegionMat(material=_iso()),
            2: RegionMat(material=_iso(E=100e9)),
        },
    )
    with pytest.raises(ValueError, match="no cell tags"):
        per_cell_arrays(inp, 3, None)


def test_unknown_region_tag_errors():
    inp = SectionInput(
        mesh_path="x.xdmf",
        region_materials={1: RegionMat(material=_iso())},
    )
    tags = SimpleNamespace(indices=np.array([0]), values=np.array([99]))
    with pytest.raises(KeyError, match="99"):
        per_cell_arrays(inp, 1, tags)


def test_orthotropic_region_rotates():
    mat = OrthotropicMaterial(
        E1=140e9,
        E2=10e9,
        E3=10e9,
        G12=5e9,
        G13=5e9,
        G23=3.5e9,
        nu12=0.3,
        nu13=0.3,
        nu23=0.4,
        rho=1600.0,
    )
    inp = SectionInput(
        mesh_path="x.xdmf",
        region_materials={
            1: RegionMat(material=mat, beta_deg=0.0, alpha_deg=0.0),
            2: RegionMat(material=mat, beta_deg=0.0, alpha_deg=90.0),
        },
    )
    tags = SimpleNamespace(indices=np.array([0, 1]), values=np.array([1, 2]))
    C, _rho = per_cell_arrays(inp, 2, tags)
    assert not np.allclose(C[0], C[1])
