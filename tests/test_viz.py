"""Agg smoke for plot_section / plot_warping — no fenicsx solve."""

from __future__ import annotations

import numpy as np

from b3_secfem.config import RegionMat, SectionInput
from b3_secfem.materials import IsotropicMaterial
from b3_secfem.result import SectionResult
from b3_secfem.viz import plot_section, plot_warping


class _IndexMap:
    def __init__(self, n: int) -> None:
        self.size_local = n


class _Topology:
    dim = 2

    def index_map(self, _dim: int) -> _IndexMap:
        return _IndexMap(1)


class _Geometry:
    # One quad; dolfinx tensor-product order is remapped [0, 2, 3, 1] → CCW.
    x = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    dofmap = np.array([[0, 3, 1, 2]], dtype=int)


class _FakeMesh:
    topology = _Topology()
    geometry = _Geometry()


def _result() -> SectionResult:
    eye = np.eye(6)
    return SectionResult(
        K=eye,
        M=eye,
        S=eye,
        R=eye,
        shear_center=(0.0, 0.0),
        tension_center=(0.0, 0.0),
        elastic_center=(0.0, 0.0),
        mass_center=(0.0, 0.0),
        mesh=_FakeMesh(),
    )


def _inp(tmp_path) -> SectionInput:
    steel = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    return SectionInput(
        mesh_path=tmp_path / "missing.xdmf",
        region_materials={1: RegionMat(material=steel)},
    )


def test_plot_section_writes_png(tmp_path):
    out = tmp_path / "sec.png"
    path = plot_section(_inp(tmp_path), _result(), out)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_warping_writes_png(tmp_path, monkeypatch):
    n = _Geometry.x.shape[0]
    disp = np.zeros((n, 3))
    disp[:, 0] = 0.01
    monkeypatch.setattr(
        "b3_secfem.viz._evaluate_mode_at_nodes",
        lambda res, mode, nodes: (disp, "Vx", False),
    )
    out = tmp_path / "warp.png"
    path = plot_warping(_result(), 0, out)
    assert path.is_file()
    assert path.stat().st_size > 0
