"""Tests for the pluggable backend system (fenicsx + mfem).

These tests are designed to run in two modes:
- Without mfem installed (most CI / developer machines): the mfem-specific
  tests are skipped; the dispatcher, config validation, and fenicsx path are
  still exercised.
- With mfem installed: the NotImplementedError for the incomplete kernel is
  asserted, proving the routing works.

The goal is to protect the refactoring done for the MFEM backend addition
while the numeric kernel (custom integrators) is still being implemented.
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


def test_solve_explicit_backend_fenicsx_kwarg(tmp_path):
    """solve(..., backend="fenicsx") must behave identically to the default."""
    # We only need the module to be importable; the actual heavy test that
    # exercises a full solve lives in test_iso_rectangle.py (which we can
    # also call with the kwarg).
    pytest.importorskip("dolfinx")
    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    from b3_secfem import write_xdmf

    # tiny mesh
    m = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [(-0.05, -0.05), (0.05, 0.05)],
        [4, 4],
        cell_type=dmesh.CellType.quadrilateral,
    )
    p = tmp_path / "tiny_rect.xdmf"
    write_xdmf(p, m)

    mat = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    inp = SectionInput(
        mesh_path=p,
        region_materials={1: RegionMat(material=mat)},
    )

    res_default = solve(inp)
    res_explicit = solve(inp, backend="fenicsx")

    assert res_default.backend == "fenicsx"
    assert res_explicit.backend == "fenicsx"
    np.testing.assert_allclose(res_default.K, res_explicit.K, rtol=1e-12)
    np.testing.assert_allclose(res_default.M, res_explicit.M, rtol=1e-12)


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
    pytest.importorskip("dolfinx")  # used only to build the mesh file

    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    from b3_secfem import write_xdmf

    m = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [(-0.05, -0.05), (0.05, 0.05)],
        [4, 4],
        cell_type=dmesh.CellType.quadrilateral,
    )
    p = tmp_path / "tiny.xdmf"
    write_xdmf(p, m)

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
    assert np.isclose(res.K[2, 2], 210e9 * 0.01, rtol=1e-6)


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
