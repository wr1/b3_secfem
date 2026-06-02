"""Cross-backend numeric comparison: fenicsx (dolfinx) vs mfem.

These tests assemble the core in-plane stiffness operator ``E`` with both
backends on the *same* mesh and assert they agree. Because dolfinx and mfem
number their global DOFs differently, the operators are equal only up to a
symmetric permutation, so the assertions use permutation-invariant quantities
(sorted eigenvalue spectrum, trace, rigid-body null-space dimension).

Each backend is assembled in its own subprocess (see ``b3_secfem.bench``):
importing dolfinx (PETSc/MPI) and mfem into one interpreter assembles fine but
segfaults at teardown, and the fenicsx CSR can alias freed PETSc memory. The
subprocess boundary serialises each operator to disk, sidestepping both.

The whole module is skipped unless BOTH backends are importable.
"""

from __future__ import annotations

import importlib.util

import pytest

# dolfinx is needed in-process to build the shared mesh; mfem is needed for the
# mfem worker subprocess. Skip the whole module unless both are present.
_HAVE_DOLFINX = importlib.util.find_spec("dolfinx") is not None
_HAVE_MFEM = importlib.util.find_spec("mfem") is not None

pytestmark = pytest.mark.skipif(
    not (_HAVE_DOLFINX and _HAVE_MFEM),
    reason="cross-backend comparison needs both dolfinx and mfem installed",
)


@pytest.mark.parametrize("mat_key", ["iso", "ortho"])
def test_stiffness_operator_matches_fenicsx(mat_key):
    """mfem's E operator equals fenicsx's to ~1e-12 on the sorted spectrum."""
    from b3_secfem.bench import run_comparison

    res = run_comparison(mat_key, nx=12, ny=8)
    c = res["compare"]

    # Both engines must expose the same problem size.
    assert res["fenicsx"]["shape"] == res["mfem"]["shape"]

    # Permutation-invariant agreement.
    assert c["spectrum_max_reldiff"] < 1e-10, c
    assert c["trace_reldiff"] < 1e-10, c

    # Both operators must have the physical 4-D rigid-body null space
    # (translations x, y, z + rotation about z).
    assert c["nulldim_a"] == 4, c
    assert c["nulldim_b"] == 4, c


def test_subprocess_metadata_is_sane():
    """The timing/shape metadata returned by the worker is well-formed."""
    from b3_secfem.bench import run_comparison

    res = run_comparison("iso", nx=8, ny=6)
    for be in ("fenicsx", "mfem"):
        meta = res[be]
        assert meta["backend"] == be
        assert meta["shape"][0] == meta["shape"][1] > 0
        assert meta["nnz"] > 0
        assert meta["assemble_s"] > 0.0
        assert meta["n_cells"] == 8 * 6


@pytest.mark.parametrize("mat_key", ["iso", "ortho"])
def test_full_solve_matches_fenicsx(mat_key):
    """The complete mfem chain solve (K, M, centres, K_xy) equals fenicsx.

    Both run on the same mesh; K/M/centres are physical quantities (not just
    operators) so they compare directly with no permutation ambiguity.
    """
    from b3_secfem.bench import run_full_comparison

    res = run_full_comparison(mat_key, nx=12, ny=8)
    c = res["compare"]
    assert c["K_rel"] < 1e-8, c
    assert c["M_rel"] < 1e-10, c
    assert c["K_section_xy_rel"] < 1e-8, c
    assert c["tension_center_absdiff"] < 1e-9, c
    assert c["shear_center_absdiff"] < 1e-8, c


def test_mfem_rectangle_matches_closed_form():
    """mfem K/M on a centred rectangle match the analytic beam values.

    Guards the full chain end-to-end independently of fenicsx: K[Fz,Fz]=E*A,
    K[Mx,Mx]=E*Ixx, K[My,My]=E*Iyy, m_tot=rho*A, K_xy=G*A.
    """
    import tempfile
    from pathlib import Path

    import numpy as np

    from b3_secfem import IsotropicMaterial, RegionMat, SectionInput, solve
    from b3_secfem.bench import make_rectangle_xdmf

    a, b = 0.2, 0.1  # make_rectangle_xdmf defaults
    E, nu, rho = 210e9, 0.3, 7850.0
    G = E / (2 * (1 + nu))
    A = a * b
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.xdmf"
        make_rectangle_xdmf(p, nx=16, ny=12)
        inp = SectionInput(
            mesh_path=p,
            region_materials={1: RegionMat(material=IsotropicMaterial(E=E, nu=nu, rho=rho))},
            degree=2, backend="mfem",
        )
        res = solve(inp)

    assert res.backend == "mfem"
    assert np.isclose(res.K[2, 2], E * A, rtol=1e-9)               # E*A
    assert np.isclose(res.K[3, 3], E * a * b**3 / 12, rtol=1e-9)   # E*Ixx
    assert np.isclose(res.K[4, 4], E * b * a**3 / 12, rtol=1e-9)   # E*Iyy
    assert np.isclose(res.M[0, 0], rho * A, rtol=1e-9)            # rho*A
    assert np.isclose(res.K_section_xy, G * A, rtol=1e-6)         # G*A
