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


@pytest.mark.parametrize("mat_key", ["iso", "ortho"])
def test_strain_recovery_matches_fenicsx(mat_key):
    """mfem's recovered basis + unit-load strain fields equal fenicsx's.

    Per-cell comparison with cells matched by centroid (the two engines
    number cells differently). Both recoveries are DG0 cell averages of the
    same continuous solution fields, so agreement tracks the solve agreement
    (~1e-8 on K).
    """
    from b3_secfem.bench import run_strain_comparison

    res = run_strain_comparison(mat_key, nx=12, ny=8)
    c = res["compare"]
    assert c["centroid_match_dist"] < 1e-9, c
    assert c["areas_rel"] < 1e-10, c
    assert c["eps_rel"] < 1e-6, c
    assert c["sig_rel"] < 1e-6, c
    assert c["eps_unit_rel"] < 1e-6, c
    assert c["sig_unit_rel"] < 1e-6, c


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
            region_materials={
                1: RegionMat(material=IsotropicMaterial(E=E, nu=nu, rho=rho))
            },
            degree=2,
            backend="mfem",
        )
        res = solve(inp)

    assert res.backend == "mfem"
    assert np.isclose(res.K[2, 2], E * A, rtol=1e-9)  # E*A
    assert np.isclose(res.K[3, 3], E * a * b**3 / 12, rtol=1e-9)  # E*Ixx
    assert np.isclose(res.K[4, 4], E * b * a**3 / 12, rtol=1e-9)  # E*Iyy
    assert np.isclose(res.M[0, 0], rho * A, rtol=1e-9)  # rho*A
    assert np.isclose(res.K_section_xy, G * A, rtol=1e-6)  # G*A


def test_legacy_region_key_backends_agree(tmp_path):
    """A meshio file tagged ``region`` gives the same K on both backends.

    The two solves run in separate processes. Loading dolfinx and mfem in one
    interpreter assembles, then segfaults at teardown.
    """
    import json
    import subprocess
    import sys

    import meshio
    import numpy as np

    a, b, nx, ny = 0.2, 0.04, 6, 4
    xs = np.linspace(-a / 2, a / 2, nx + 1)
    ys = np.linspace(-b / 2, b / 2, ny + 1)
    coords = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)
    quads, tags = [], []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            quads.append([n0, n0 + 1, n0 + 1 + (nx + 1), n0 + (nx + 1)])
            tags.append(2 if i >= nx // 2 else 1)
    path = tmp_path / "strip.xdmf"
    meshio.write(
        path,
        meshio.Mesh(
            coords,
            [("quad", np.asarray(quads, dtype=np.int64))],
            cell_data={"region": [np.asarray(tags, dtype=np.int32)]},
        ),
    )
    script = """
import json, sys
from b3_secfem import IsotropicMaterial, RegionMat, SectionInput, solve
path, backend = sys.argv[1], sys.argv[2]
steel = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
alu = IsotropicMaterial(E=70e9, nu=0.33, rho=2700.0)
inp = SectionInput(
    mesh_path=path,
    region_materials={
        1: RegionMat(material=steel),
        2: RegionMat(material=alu),
    },
    backend=backend,
    degree=2,
)
res = solve(inp)
json.dump(res.K.tolist(), sys.stdout)
"""

    def run(backend: str) -> np.ndarray:
        proc = subprocess.run(
            [sys.executable, "-c", script, str(path), backend],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(f"{backend} failed:\n{proc.stderr}")
        return np.asarray(json.loads(proc.stdout), dtype=float)

    fen = run("fenicsx")
    mf = run("mfem")
    # Elementwise rtol treats ~0 coupling entries as failures. Scale by ||K||.
    scale = max(float(np.abs(fen).max()), float(np.abs(mf).max()), 1.0)
    assert float(np.max(np.abs(fen - mf))) / scale < 1e-10
