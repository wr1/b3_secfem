"""Torture test for b3_secfem vs gxbeam_section vs ANBA4.

Designed to flag any and all formulation differences between the three
engines. Single NACA airfoil mesh, but with:

  * Two distinct *truly* orthotropic materials (E1 ≠ E2 ≠ E3, all G's
    different, all ν's different — no transverse-isotropy symmetries
    that could mask rotation handling).
  * Four cell regions, each with a different odd fibre angle
    (23°, −17°, 35°, −41°). No symmetry: NE, NW, SE, SW each get a
    different (material, theta) pair.
  * Mesh origin at chord centre (not at any of sc / tc / mc), so the
    raw K[Mz, Mz] picks up the shear-centre offset coupling.

For each engine we report:

  1. Section centres (sc, tc, mc via .mass_center; .elastic_center kept as alias).
  2. Full 6×6 K diagonal at the *mesh origin* — these will disagree.
  3. K diagonal **shifted to that engine's own shear centre** —
     the reference-frame artefact should drop out, leaving only real
     formulation differences.
  4. Compliance-decoupled stiffnesses ``1 / S[i, i]`` — these are
     reference-point-invariant in the sense that they correspond to
     applying generalised force i alone (other forces zero), so they
     give the cleanest cross-engine comparison.
  5. Pairwise relative error matrix on the full K (max element-wise).

Run::

    /home/wr1/projects/b3/b3_secfem/.venv/bin/python examples/probe_airfoil_torture.py
"""

from __future__ import annotations

import dolfinx  # noqa: F401  - load before juliacall
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cross_check_b3_gx import (  # noqa: E402
    _gxbeam_section_from_arrays,
    ortho_row,
)
from _anba_runner_host import anba_section_from_arrays  # noqa: E402
from _gx_vtu import write_gx_vtu  # noqa: E402
from b3_secfem import (  # noqa: E402
    OrthotropicMaterial,
    SectionInput,
    from_gxbeam_vtu,
    solve,
    to_gxbeam_order,
)
from probe_airfoil_3way import gen_airfoil  # noqa: E402


# ── two truly orthotropic materials ──────────────────────────────────────────
# All E's distinct, all G's distinct, all ν's distinct — nothing transversely
# isotropic, nothing equal-by-coincidence. Densities also distinct.

MAT_A = OrthotropicMaterial(
    E1=160e9, E2=80e9, E3=20e9,
    G12=8e9, G13=6e9, G23=4e9,
    nu12=0.25, nu13=0.30, nu23=0.35,
    rho=1500.0, name="ortho-A",
)

MAT_B = OrthotropicMaterial(
    E1=110e9, E2=40e9, E3=15e9,
    G12=7e9, G13=5.5e9, G23=3.5e9,
    nu12=0.22, nu13=0.27, nu23=0.32,
    rho=1900.0, name="ortho-B",
)


# ── helpers ──────────────────────────────────────────────────────────────────


def shift_K(K: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate 6×6 K in [Fx,Fy,Fz,Mx,My,Mz] from origin to (dx, dy, 0).

    Strain transform (rigid kinematic):
        u(P') = u(P) + θ × (P' − P)
    Force transform (work conservation):
        F(P) = T_d^T · F(P'),  K(P') = T_d^{−T} K(P) T_d^{−1}
    """
    T_d = np.eye(6)
    T_d[0, 5] = -dy
    T_d[1, 5] = +dx
    T_d[2, 3] = +dy
    T_d[2, 4] = -dx
    Ti = np.linalg.inv(T_d)
    return Ti.T @ K @ Ti


def assign_regions(coords: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Per-cell region id ∈ {0, 1, 2, 3} based on cell-centroid quadrant.

    Region 0: front-upper  (x_c < 0, y_c ≥ 0)  → MAT_A, θ =  23°
    Region 1: front-lower  (x_c < 0, y_c < 0)  → MAT_A, θ = −17°
    Region 2: back-upper   (x_c ≥ 0, y_c ≥ 0)  → MAT_B, θ =  35°
    Region 3: back-lower   (x_c ≥ 0, y_c < 0)  → MAT_B, θ = −41°
    """
    cell_pts = coords[quads]
    cx = cell_pts[:, :, 0].mean(axis=1)
    cy = cell_pts[:, :, 1].mean(axis=1)
    region = np.zeros(quads.shape[0], dtype=np.int64)
    region[(cx < 0) & (cy >= 0)] = 0
    region[(cx < 0) & (cy < 0)] = 1
    region[(cx >= 0) & (cy >= 0)] = 2
    region[(cx >= 0) & (cy < 0)] = 3
    return region


REGION_MATERIAL = [MAT_A, MAT_A, MAT_B, MAT_B]
REGION_THETA_DEG = [23.0, -17.0, 35.0, -41.0]


def fmt_diag(K: np.ndarray, names: list[str]) -> str:
    return " | ".join(f"{n}={v:>11.4e}" for n, v in zip(names, np.diag(K)))


def main() -> int:
    coords, quads = gen_airfoil(
        chord=1.0, thickness=0.24, n_chord=64, n_thick=16, eta_clip=0.01,
    )
    n_cells = quads.shape[0]
    region = assign_regions(coords, quads)

    print(f"NACA 0024  chord=1.0  thickness=0.24  eta_clip=0.01  n_cells={n_cells}")
    counts = np.bincount(region, minlength=4)
    for r, (mat, th, c) in enumerate(zip(REGION_MATERIAL, REGION_THETA_DEG, counts)):
        print(f"  region {r}: {mat.name:<8}  θ = {th:>+5.1f}°   ({c} cells)")
    print()

    # ── per-cell mat_props + theta arrays for gxbeam ─────────────────────────
    mat_props = np.zeros((n_cells, 10))
    theta_rad = np.zeros(n_cells)
    for r, (mat, th_deg) in enumerate(zip(REGION_MATERIAL, REGION_THETA_DEG)):
        mask = region == r
        mat_props[mask] = ortho_row(mat)
        theta_rad[mask] = np.radians(th_deg)

    out = Path(__file__).parent / "cross_check_out"
    out.mkdir(parents=True, exist_ok=True)
    vtu = out / "torture.vtu"
    write_gx_vtu(vtu, coords, quads, mat_props, theta_rad)

    # ── gxbeam ────────────────────────────────────────────────────────────────
    K_gx_gxorder, _S, _M, sc_gx, tc_gx, mc_gx = _gxbeam_section_from_arrays(
        coords, quads, mat_props, theta_rad,
    )
    # Permute gxbeam [Fz, Fx, Fy, Mz, Mx, My] → natural [Fx, Fy, Fz, Mx, My, Mz]
    perm_gx_to_nat = np.array([1, 2, 0, 4, 5, 3])
    K_gx = K_gx_gxorder[np.ix_(perm_gx_to_nat, perm_gx_to_nat)]

    sc_gx = np.asarray(sc_gx).reshape(-1)
    tc_gx = np.asarray(tc_gx).reshape(-1)
    mc_gx = np.asarray(mc_gx).reshape(-1)

    # ── b3_secfem ─────────────────────────────────────────────────────────────
    info = from_gxbeam_vtu(vtu)
    per_cell_mat = [REGION_MATERIAL[r] for r in region]
    per_cell_alpha = np.array([REGION_THETA_DEG[r] for r in region])
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=per_cell_mat,
        per_cell_alpha_deg=per_cell_alpha,
    )
    res = solve(inp)
    K_b3 = res.K   # natural [Fx, Fy, Fz, Mx, My, Mz]
    sc_b3 = np.asarray(res.shear_center)
    tc_b3 = np.asarray(res.tension_center)
    mc_b3 = np.asarray(res.mass_center)
    ec_b3 = np.asarray(res.elastic_center)  # compat alias for mass

    # ── ANBA4 ─────────────────────────────────────────────────────────────────
    fiber_orient = 90.0 - per_cell_alpha       # ANBA convention: fiber=90 → fibre along z
    plane_orient = np.zeros(n_cells)
    anba = anba_section_from_arrays(
        coords, quads,
        material=REGION_MATERIAL,
        material_id=region,
        fiber_orientation_deg=fiber_orient,
        plane_orientation_deg=plane_orient,
    )
    K_anba = anba["K"]   # natural [Fx, Fy, Fz, Mx, My, Mz]
    # ANBA does not currently report sc / tc / mc through this wrapper.

    # ── compute compliances and shear-centre-shifted K ────────────────────────
    S_gx = np.linalg.inv(K_gx)
    S_b3 = np.linalg.inv(K_b3)
    S_anba = np.linalg.inv(K_anba)
    sc_gx_xy = (-S_gx[5, 1] / S_gx[5, 5], +S_gx[5, 0] / S_gx[5, 5])
    sc_b3_xy = (-S_b3[5, 1] / S_b3[5, 5], +S_b3[5, 0] / S_b3[5, 5])
    sc_anba_xy = (-S_anba[5, 1] / S_anba[5, 5], +S_anba[5, 0] / S_anba[5, 5])

    K_gx_sc = shift_K(K_gx, *sc_gx_xy)
    K_b3_sc = shift_K(K_b3, *sc_b3_xy)
    K_anba_sc = shift_K(K_anba, *sc_anba_xy)

    # ── report ────────────────────────────────────────────────────────────────
    names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    print("Section centres (sc, tc, mc):")
    print(f"  gxbeam      sc={tuple(sc_gx)}  tc={tuple(tc_gx)}  mc={tuple(mc_gx)}")
    print(f"  b3_secfem   sc={tuple(sc_b3)}  tc={tuple(tc_b3)}  mc={tuple(mc_b3)}  (elastic alias={tuple(ec_b3)})")
    print(f"  ANBA  (sc derived from S^-1 only): sc={sc_anba_xy}")
    print()

    print("(1) Raw K diagonal at mesh origin:")
    print(f"  gxbeam:    {fmt_diag(K_gx, names)}")
    print(f"  b3_secfem: {fmt_diag(K_b3, names)}")
    print(f"  ANBA:      {fmt_diag(K_anba, names)}")
    print()

    print("(2) K diagonal SHIFTED to each engine's shear centre:")
    print(f"  gxbeam@sc: {fmt_diag(K_gx_sc, names)}")
    print(f"  b3@sc:     {fmt_diag(K_b3_sc, names)}")
    print(f"  ANBA@sc:   {fmt_diag(K_anba_sc, names)}")
    print()

    print("(3) Compliance-decoupled stiffness  1 / S[i, i]:")
    inv_diag = lambda S: 1.0 / np.diag(S)  # noqa: E731
    print(f"  gxbeam:    " + " | ".join(f"{n}={v:>11.4e}" for n, v in zip(names, inv_diag(S_gx))))
    print(f"  b3_secfem: " + " | ".join(f"{n}={v:>11.4e}" for n, v in zip(names, inv_diag(S_b3))))
    print(f"  ANBA:      " + " | ".join(f"{n}={v:>11.4e}" for n, v in zip(names, inv_diag(S_anba))))
    print()

    eps = 1e-30

    def relmax(A: np.ndarray, B: np.ndarray) -> float:
        gmean = np.sqrt(np.outer(np.maximum(np.abs(np.diag(A)), eps),
                                 np.maximum(np.abs(np.diag(A)), eps)))
        gmean[gmean == 0] = 1.0
        return float(np.max(np.abs(A - B) / gmean))

    print("(4) Pairwise max element-wise relative error on full 6×6 K (scaled by gmean of diags):")
    pairs = [
        ("gxbeam@sc vs b3_secfem@sc", K_gx_sc, K_b3_sc),
        ("gxbeam@sc vs ANBA@sc",      K_gx_sc, K_anba_sc),
        ("b3_secfem@sc vs ANBA@sc",   K_b3_sc, K_anba_sc),
        ("gxbeam@origin vs b3@origin", K_gx, K_b3),
        ("gxbeam@origin vs ANBA@origin", K_gx, K_anba),
        ("b3@origin vs ANBA@origin", K_b3, K_anba),
    ]
    for label, A, B in pairs:
        print(f"  {label:<32}: max rel = {100*relmax(A, B):.3f}%")

    # ── compliance-diagonal direct comparison (most invariant) ───────────────
    print()
    print("(5) 1/S[i,i] cross-engine relative errors (b3 vs gxbeam, ANBA vs gxbeam):")
    sd_gx = inv_diag(S_gx)
    sd_b3 = inv_diag(S_b3)
    sd_an = inv_diag(S_anba)
    for i, n in enumerate(names):
        b3_rel = (sd_b3[i] - sd_gx[i]) / max(abs(sd_gx[i]), eps)
        an_rel = (sd_an[i] - sd_gx[i]) / max(abs(sd_gx[i]), eps)
        print(f"  {n:<3}: b3 vs gx = {100*b3_rel:>+7.3f}%   ANBA vs gx = {100*an_rel:>+7.3f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
