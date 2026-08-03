"""Pydantic config tests."""

from __future__ import annotations

import numpy as np
import pytest

from b3_secfem.config import RegionMat, SectionInput
from b3_secfem.materials import IsotropicMaterial


def _iso():
    return IsotropicMaterial(E=70e9, nu=0.33, rho=2700.0)


def test_region_materials_path():
    inp = SectionInput(
        mesh_path="x.xdmf",
        region_materials={1: RegionMat(material=_iso(), beta_deg=15.0)},
    )
    assert inp.region_materials[1].beta_deg == 15.0
    assert inp.linear_solver == "gamg"


def test_linear_solver_option():
    inp = SectionInput(
        mesh_path="x.xdmf",
        region_materials={1: RegionMat(material=_iso())},
        linear_solver="lu",
    )
    assert inp.linear_solver == "lu"
    with pytest.raises(Exception):
        SectionInput(
            mesh_path="x.xdmf",
            region_materials={1: RegionMat(material=_iso())},
            linear_solver="not-a-solver",
        )


def test_per_cell_path():
    n = 5
    inp = SectionInput(
        mesh_path="x.xdmf",
        per_cell_material=[_iso()] * n,
        per_cell_beta_deg=np.zeros(n),
        per_cell_alpha_deg=np.zeros(n),
    )
    assert len(inp.per_cell_material) == n


def test_at_least_one_input_required():
    with pytest.raises(ValueError, match="must supply"):
        SectionInput(mesh_path="x.xdmf")


def test_per_cell_size_mismatch_rejected():
    with pytest.raises(ValueError, match="does not match n_cells"):
        SectionInput(
            mesh_path="x.xdmf",
            per_cell_material=[_iso()] * 3,
            per_cell_beta_deg=np.zeros(5),
        )
