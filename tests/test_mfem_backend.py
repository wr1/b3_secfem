"""Tests for the pluggable backend system (fenicsx + mfem).

These tests are designed to run in two modes:
- Without mfem installed (most CI / developer machines): the mfem-specific
  tests are skipped; the dispatcher, config validation, and fenicsx path are
  still exercised.
- With mfem installed: the full chain solve is exercised. Numeric agreement
  with fenicsx lives in ``tests/test_backend_comparison.py``.
"""

from __future__ import annotations

import pytest

from b3_secfem import (
    IsotropicMaterial,
    RegionMat,
    SectionInput,
    solve,
)
import numpy as np

from b3_secfem.backends import get_backend
from b3_secfem.backends.common import ALL_MODES, STAGE1_MODES, STAGE2_MODES


# ─────────────────────────────────────────────────────────────────────────────
# Always-on tests (no FEM backend required)
# ─────────────────────────────────────────────────────────────────────────────


def test_backend_constants_are_consistent():
    """The mode tuples exported for internal use are correct and non-empty."""
    assert set(ALL_MODES) == {0, 1, 2, 3, 4, 5}
    assert set(STAGE1_MODES) == {2, 3, 4, 5}
    assert set(STAGE2_MODES) == {0, 1}
    assert len(ALL_MODES) == 6


def test_sectioninput_accepts_valid_backends():
    mat = IsotropicMaterial(E=1e9, nu=0.3, rho=1000.0)
    for b in ("fenicsx", "mfem"):
        inp = SectionInput(
            mesh_path="dummy.vtu",
            region_materials={1: RegionMat(material=mat)},
            backend=b,
        )
        assert inp.backend == b


def test_sectioninput_rejects_invalid_backend():
    from pydantic import ValidationError

    mat = IsotropicMaterial(E=1e9, nu=0.3, rho=1000.0)
    with pytest.raises(ValidationError):
        SectionInput(
            mesh_path="dummy.vtu",
            region_materials={1: RegionMat(material=mat)},
            backend="nonexistent-backend-xyz",
        )


def test_get_backend_fenicsx_always_available():
    be = get_backend("fenicsx")
    assert be is not None
    assert hasattr(be, "solve")


def test_mfem_reads_two_grid_tags(tmp_path):
    """A dolfinx two-grid file keeps cell tags in file order on the MFEM mesh."""
    pytest.importorskip("mfem")
    import h5py

    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=np.int64)
    tags = np.array([[3], [9]], dtype=np.int32)
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
      <Topology TopologyType="Quadrilateral" NumberOfElements="2" NodesPerElement="4">
        <DataItem Dimensions="2 4" Format="HDF">two.h5:/Mesh/mesh/topology</DataItem>
      </Topology>
      <Geometry GeometryType="XY">
        <DataItem Dimensions="6 2" Format="HDF">two.h5:/Mesh/mesh/geometry</DataItem>
      </Geometry>
    </Grid>
    <Grid Name="cell_tags" GridType="Uniform">
      <Attribute Name="cell_tags" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="2 1" Format="HDF">two.h5:/MeshTags/cell_tags/Values</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
    )
    mat = IsotropicMaterial(E=1.0, nu=0.3, rho=1.0)
    inp = SectionInput(
        mesh_path=xdmf,
        region_materials={3: RegionMat(material=mat), 9: RegionMat(material=mat)},
        backend="mfem",
    )
    from b3_secfem.backends.mfem import _load_mfem_mesh

    _mesh, got, n_cells = _load_mfem_mesh(inp)
    assert n_cells == 2
    assert got.tolist() == [3, 9]


# ─────────────────────────────────────────────────────────────────────────────
# MFEM-specific tests (gated)
# ─────────────────────────────────────────────────────────────────────────────


def test_mfem_backend_is_registered_when_mfem_present():
    pytest.importorskip("mfem")
    be = get_backend("mfem")
    assert be is not None
    assert hasattr(be, "solve")


def test_mfem_backend_full_solve_runs(tmp_path):
    """The full mfem chain solve runs end-to-end and returns a tagged result.

    Numeric agreement with fenicsx + analytic values is covered in
    test_backend_comparison.py; here we just guard that routing to the mfem
    backend produces a complete SectionResult on a real mesh.
    """
    pytest.importorskip("mfem")

    from b3_secfem.geometry import rectangle

    section, info = rectangle(0.1, 0.1, 4, 4)
    p = section.write(tmp_path / "tiny.xdmf")

    mat = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    inp = SectionInput(
        mesh_path=p,
        region_materials={1: RegionMat(material=mat)},
        backend="mfem",
    )
    res = solve(inp)
    assert res.backend == "mfem"
    assert res.K.shape == (6, 6)
    assert res.M.shape == (6, 6)
    assert np.all(np.isfinite(res.K))
    # axial stiffness must be positive and ~ E*A
    assert res.K[2, 2] > 0
    assert np.isclose(res.K[2, 2], 210e9 * info["A"], rtol=1e-6)


def test_mfem_module_can_be_imported_without_side_effects():
    """Importing the mfem backend module should not require mfem at import time
    (only when get_backend("mfem") or solve(backend="mfem") is actually used)."""
    # We deliberately do NOT call importorskip here. If mfem is absent the
    # import of the module must still succeed (it only does the try: import
    # inside the functions that need it).
    import importlib

    mod = importlib.import_module("b3_secfem.backends.mfem")
    assert hasattr(mod, "solve")
    # The _MFEM_AVAILABLE flag must exist and be a bool
    assert isinstance(getattr(mod, "_MFEM_AVAILABLE", None), bool)
