"""Bulk (numba/numpy) Voigt assembly matches the Python MFEM integrator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mfem")
pytest.importorskip("scipy")

from b3_secfem.backends.bulk_assemble import (  # noqa: E402
    _HAS_NUMBA,
    assemble_voigt_bulk,
    tabulate_fes,
)
from b3_secfem.backends import mfem as mfem_be  # noqa: E402
from b3_secfem.materials import IsotropicMaterial  # noqa: E402


def _rect_mesh_and_C(nx=8, ny=6, degree=2):
    import tempfile
    from pathlib import Path

    import mfem.ser as mfem

    from b3_secfem.config import RegionMat, SectionInput
    from b3_secfem.geometry import rectangle

    mat = IsotropicMaterial(E=100e9, nu=0.3, rho=2000.0)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "m.xdmf"
        section, _info = rectangle(0.2, 0.1, nx, ny)
        section.write(path)
        inp = SectionInput(
            mesh_path=path,
            region_materials={1: RegionMat(material=mat)},
            degree=degree,
            backend="mfem",
        )
        mesh, _tags, n_cells = mfem_be._load_mfem_mesh(inp)

    for e in range(n_cells):
        mesh.GetElement(e).SetAttribute(e + 1)
    mesh.Finalize()
    fec = mfem.H1_FECollection(degree, mesh.Dimension())
    fes = mfem.FiniteElementSpace(mesh, fec, 3)
    C = np.tile(mat.C_local(), (n_cells, 1, 1))
    return mesh, fes, C


@pytest.mark.parametrize(
    "test_kind,trial_kind", [("xy", "xy"), ("xy", "z"), ("z", "z")]
)
def test_bulk_matches_python_integrator(test_kind, trial_kind):
    mesh, fes, C = _rect_mesh_and_C()
    A_py = mfem_be._assemble_voigt_form(
        C, fes, test_kind, trial_kind, mesh=mesh, mode="python"
    )
    tabs = tabulate_fes(mesh, fes)
    engines = ["numpy"]
    if _HAS_NUMBA:
        engines.append("numba")
    for eng in engines:
        A_b = assemble_voigt_bulk(
            C, mesh, fes, test_kind, trial_kind, engine=eng, tables=tabs
        )
        # Frobenius relative (permutation-free: same DOF numbering)
        scale = max(np.linalg.norm(A_py.data), 1e-300)
        err = np.linalg.norm((A_py - A_b).data) / scale if (A_py - A_b).nnz else 0.0
        # denser check
        err = np.linalg.norm((A_py - A_b).toarray()) / max(
            np.linalg.norm(A_py.toarray()), 1e-300
        )
        assert err < 1e-10, f"{eng} {test_kind}/{trial_kind} rel err {err:.3e}"
