"""Same-problem ANBA vs secfem recovered unit-load strain/stress.

Compares area-weighted moments (no quad↔tri pairing). Stage-1 σ_zz is
the pin. Fy on this thin rectangle is a known residual — recorded, not
failed. Skips when neither docker ``anba4:latest`` nor a local ANBA
python is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

from _anba_runner_host import (  # noqa: E402
    _docker_image_ok,
    _local_anba_ok,
    anba_section_from_arrays,
)
from _gx_vtu import write_gx_vtu  # noqa: E402
from validation._common import ASYM, rectangle  # noqa: E402

from b3_secfem import (  # noqa: E402
    IsotropicMaterial,
    SectionInput,
    anba_to_secfem_input,
    from_gxbeam_vtu,
    recover_unit_load_strains,
    solve,
)

STEEL = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0, name="iso")
LOADS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
STAGE1 = (2, 3, 4)  # Fz, Mx, My
_SECFEM_ORDER = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def _anba_available() -> bool:
    return _docker_image_ok("anba4:latest") or _local_anba_ok()


pytestmark = pytest.mark.skipif(not _anba_available(), reason="ANBA not available")


def _mat_row(mat) -> np.ndarray:
    if isinstance(mat, IsotropicMaterial):
        g = mat.E / (2 * (1 + mat.nu))
        return np.array([mat.E, mat.E, mat.E, g, g, g, mat.nu, mat.nu, mat.nu, mat.rho])
    return np.array(
        [
            mat.E1,
            mat.E2,
            mat.E3,
            mat.G12,
            mat.G13,
            mat.G23,
            mat.nu12,
            mat.nu13,
            mat.nu23,
            mat.rho,
        ]
    )


def _moments(field: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted mean and RMS per Voigt component. field (n_cells, 6)."""
    w = np.asarray(areas, dtype=float)
    tot = float(w.sum())
    mean = (field * w[:, None]).sum(axis=0) / tot
    rms = np.sqrt((field**2 * w[:, None]).sum(axis=0) / tot)
    return mean, rms


def _rel(a: float, b: float, floor: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), floor)


def _solve_secfem_fields(coords, quads, mat, beta, alpha, work: Path, tag: str):
    n = quads.shape[0]
    vtu = work / f"{tag}.vtu"
    write_gx_vtu(vtu, coords, quads, np.tile(_mat_row(mat), (n, 1)), np.zeros(n))
    info = from_gxbeam_vtu(vtu)
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=[mat] * info["n_cells"],
        per_cell_beta_deg=np.full(info["n_cells"], beta),
        per_cell_alpha_deg=np.full(info["n_cells"], alpha),
        backend="fenicsx",
        linear_solver="lu",
        degree=2,
    )
    res = solve(inp)
    ul = recover_unit_load_strains(res)
    return ul


def _solve_anba_fields(coords, quads, mat, fiber, plane):
    n = quads.shape[0]
    return anba_section_from_arrays(
        coords,
        quads,
        mat,
        fiber_orientation_deg=np.full(n, fiber),
        plane_orientation_deg=np.full(n, plane),
        recover_fields=True,
    )


def _compare_sigma_zz(ul, anba, loads=STAGE1, tol=0.05):
    assert list(anba["anba_order"]) == _SECFEM_ORDER
    a_area = anba["tri_area"]
    s_area = ul.cell_areas
    floor_scale = max(
        abs(_moments(ul.sigma[2], s_area)[0][2]),
        abs(_moments(anba["sigma"][2], a_area)[0][2]),
        1.0,
    )
    floor = 1e-6 * floor_scale
    rec = {}
    for k in loads:
        sm, sr = _moments(ul.sigma[k], s_area)
        am, ar = _moments(anba["sigma"][k], a_area)
        rec[LOADS[k]] = {
            "mean_rel": _rel(sm[2], am[2], floor),
            "rms_rel": _rel(sr[2], ar[2], floor),
            "secfem_mean": float(sm[2]),
            "anba_mean": float(am[2]),
        }
        assert rec[LOADS[k]]["mean_rel"] < tol, rec
        assert rec[LOADS[k]]["rms_rel"] < tol, rec
    return rec


@pytest.mark.parametrize(
    "fiber,plane,mat,tag",
    [
        (0.0, 0.0, STEEL, "iso"),
        (0.0, 0.0, ASYM, "asym_f0"),
        (90.0, 0.0, ASYM, "asym_f90"),
    ],
    ids=["iso", "asym_a0", "asym_a90"],
)
def test_iso_and_asym_unit_sigma_zz_matches_anba(tmp_path, fiber, plane, mat, tag):
    coords, quads, _area = rectangle(0.04, 0.01, 8, 4)
    place = anba_to_secfem_input(mat, fiber, plane)
    ul = _solve_secfem_fields(
        coords, quads, place.material, place.beta_deg, place.alpha_deg, tmp_path, tag
    )
    anba = _solve_anba_fields(coords, quads, mat, fiber, plane)
    rec = _compare_sigma_zz(ul, anba)
    # Fy residual is recorded, not asserted.
    sm, _ = _moments(ul.sigma[1], ul.cell_areas)
    am, _ = _moments(anba["sigma"][1], anba["tri_area"])
    fy_rel = _rel(sm[3], am[3], max(abs(sm[3]), abs(am[3]), 1.0) * 1e-6)
    print(f"{tag} Fy mean σ_yz rel={fy_rel:.3f} rec={rec}")


def test_iso_unit_eps_zz_and_shear_moments_match_anba(tmp_path):
    coords, quads, _area = rectangle(0.04, 0.01, 8, 4)
    ul = _solve_secfem_fields(coords, quads, STEEL, 0.0, 0.0, tmp_path, "iso_eps")
    anba = _solve_anba_fields(coords, quads, STEEL, 0.0, 0.0)
    assert list(anba["anba_order"]) == _SECFEM_ORDER
    s_area, a_area = ul.cell_areas, anba["tri_area"]
    for k in STAGE1:
        sm, _ = _moments(ul.epsilon[k], s_area)
        am, _ = _moments(anba["epsilon"][k], a_area)
        floor = 1e-6 * max(abs(sm[2]), abs(am[2]), 1e-12)
        assert _rel(sm[2], am[2], floor) < 0.05, (k, sm[2], am[2])

    sm, _ = _moments(ul.sigma[0], s_area)
    am, _ = _moments(anba["sigma"][0], a_area)
    floor = 1e-6 * max(abs(sm[4]), abs(am[4]), 1.0)
    assert _rel(sm[4], am[4], floor) < 0.08, (sm[4], am[4])

    sm, sr = _moments(ul.sigma[5], s_area)
    am, ar = _moments(anba["sigma"][5], a_area)
    assert abs(sm[2]) < 1e-4 and abs(am[2]) < 1e-4
    floor = 1e-6 * max(sr[4], ar[4], sr[3], ar[3], 1.0)
    assert _rel(sr[4], ar[4], floor) < 0.10, (sr[4], ar[4])
    assert _rel(sr[3], ar[3], floor) < 0.10, (sr[3], ar[3])
