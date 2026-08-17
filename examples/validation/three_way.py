#!/usr/bin/env python3
"""Three-way same-problem campaign: secfem, ANBA, gxbeam.

Cases are named by the *physical* placement. Each solver gets the
mapped input — not the same angle number.

    fibre on the beam:   secfem α=90,  ANBA fiber=90,  gxbeam θ=0
    fibre along +x:      secfem α=0,   ANBA fiber=0,   gxbeam θ=π/2
    mid-tilt:            secfem α=45,  ANBA fiber=45,  gxbeam θ=π/4
    plane roll:          secfem β,     ANBA plane=β,   gxbeam has no β

    micromamba run -n b3secfem python examples/validation/three_way.py
    micromamba run -n b3secfem python examples/validation/three_way.py --skip-anba
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_EXAMPLES = Path(__file__).resolve().parent.parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from validation._common import (  # noqa: E402
    ASYM,
    CARBON_UD,
    FORCE_ORDER_SECFEM,
    GLASS_BIAX,
    OUT,
    closest_modulus,
    rectangle,
    rel_per_dof,
)
from _gx_vtu import write_gx_vtu  # noqa: E402
from b3_secfem import (  # noqa: E402
    IsotropicMaterial,
    OrthotropicMaterial,
    SectionInput,
    from_gxbeam_order,
    from_gxbeam_vtu,
    secfem_to_anba_input,
    solve,
)

DOF = list(FORCE_ORDER_SECFEM)
STEEL = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0, name="steel")


def _row(mat) -> np.ndarray:
    if isinstance(mat, IsotropicMaterial):
        g = mat.E / (2.0 * (1.0 + mat.nu))
        return np.array(
            [mat.E, mat.E, mat.E, g, g, g, mat.nu, mat.nu, mat.nu, mat.rho]
        )
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


def _gx_row_swap23(mat: OrthotropicMaterial) -> np.ndarray:
    nu32 = mat.nu23 * mat.E3 / mat.E2
    return np.array(
        [
            mat.E1,
            mat.E3,
            mat.E2,
            mat.G13,
            mat.G12,
            mat.G23,
            mat.nu13,
            mat.nu12,
            nu32,
            mat.rho,
        ]
    )


def ensure_ccw(coords: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Flip CW quads. secfem's from_gxbeam_vtu does this; ANBA/gxbeam do not."""
    q = np.asarray(quads, dtype=np.int64).copy()
    p = coords[q]
    shoelace = 0.5 * (
        p[:, 0, 0] * p[:, 1, 1]
        + p[:, 1, 0] * p[:, 2, 1]
        + p[:, 2, 0] * p[:, 3, 1]
        + p[:, 3, 0] * p[:, 0, 1]
        - p[:, 1, 0] * p[:, 0, 1]
        - p[:, 2, 0] * p[:, 1, 1]
        - p[:, 3, 0] * p[:, 2, 1]
        - p[:, 0, 0] * p[:, 3, 1]
    )
    flip = shoelace < 0.0
    if np.any(flip):
        q[flip] = q[flip][:, [0, 3, 2, 1]]
    return q


def shear_center_from_K(K: np.ndarray) -> tuple[float, float]:
    """xs, ys from secfem-order K. Same formula as post.compute_centres."""
    try:
        S = np.linalg.inv(np.asarray(K, float))
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    if not np.isfinite(S[5, 5]) or abs(S[5, 5]) < 1e-30:
        return float("nan"), float("nan")
    return float(-S[5, 1] / S[5, 5]), float(S[5, 0] / S[5, 5])


def _pack_K(K: np.ndarray, extra: dict | None = None) -> dict:
    K = np.asarray(K, float)
    xs, ys = shear_center_from_K(K)
    rec = {
        "K": K.tolist(),
        "K_diag": np.diag(K).tolist(),
        "EA": float(K[2, 2]),
        "EIx": float(K[3, 3]),
        "EIy": float(K[4, 4]),
        "GJ": float(K[5, 5]),
        "shear_center_from_K": [xs, ys],
    }
    if extra:
        rec.update(extra)
    return rec


def solve_secfem(
    coords, quads, mats, beta, alpha, work: Path, tag: str
) -> dict:
    n = quads.shape[0]
    if not isinstance(mats, list):
        mats = [mats] * n
        beta = np.full(n, float(beta))
        alpha = np.full(n, float(alpha))
    vtu = work / f"{tag}.vtu"
    write_gx_vtu(
        vtu,
        coords,
        quads,
        np.stack([_row(m) for m in mats]),
        np.zeros(n),
    )
    info = from_gxbeam_vtu(vtu)
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=mats,
        per_cell_beta_deg=np.asarray(beta, float),
        per_cell_alpha_deg=np.asarray(alpha, float),
        backend="fenicsx",
        linear_solver="lu",
        degree=2,
    )
    res = solve(inp)
    return _pack_K(
        res.K,
        {
            "shear_center": [float(res.shear_center[0]), float(res.shear_center[1])],
            "tension_center": [
                float(res.tension_center[0]),
                float(res.tension_center[1]),
            ],
            "mass_center": [float(res.mass_center[0]), float(res.mass_center[1])],
            "n_cells": int(info["n_cells"]),
        },
    )


def solve_anba(coords, quads, mats, fiber, plane) -> dict:
    from _anba_runner_host import anba_section_from_arrays

    n = quads.shape[0]
    if not isinstance(mats, list):
        mats = [mats] * n
        fiber = np.full(n, float(fiber))
        plane = np.full(n, float(plane))
    uniq, ids, index = [], [], {}
    for m in mats:
        key = (m.name, getattr(m, "E1", getattr(m, "E", 0.0)))
        if key not in index:
            index[key] = len(uniq)
            uniq.append(m)
        ids.append(index[key])
    out = anba_section_from_arrays(
        coords,
        quads,
        uniq,
        fiber_orientation_deg=np.asarray(fiber, float),
        plane_orientation_deg=np.asarray(plane, float),
        material_id=np.asarray(ids, np.int64),
    )
    return _pack_K(out["K"], {"anba_order": out.get("anba_order")})


def solve_gxbeam(coords, quads, mats, theta_rad, *, swap23: bool = False) -> dict:
    from cross_check_b3_gx import _gxbeam_section_from_arrays

    n = quads.shape[0]
    if not isinstance(mats, list):
        mats = [mats] * n
        theta_rad = np.full(n, float(theta_rad))
    rows = []
    for m in mats:
        if isinstance(m, OrthotropicMaterial) and swap23:
            rows.append(_gx_row_swap23(m))
        else:
            rows.append(_row(m))
    K_gx, _S, M_gx, sc, tc, mc = _gxbeam_section_from_arrays(
        coords, quads, np.stack(rows), np.asarray(theta_rad, float)
    )
    K = from_gxbeam_order(np.asarray(K_gx, float))
    sc = np.asarray(sc, float).reshape(-1)
    tc = np.asarray(tc, float).reshape(-1)
    mc = np.asarray(mc, float).reshape(-1)
    return _pack_K(
        K,
        {
            "K_gxbeam_order": np.asarray(K_gx, float).tolist(),
            "M_diag_gx": np.diag(np.asarray(M_gx, float)).tolist(),
            "shear_center": [float(sc[0]), float(sc[1])],
            "tension_center": [float(tc[0]), float(tc[1])],
            "mass_center": [float(mc[0]), float(mc[1])],
            "swap23": swap23,
        },
    )


def _gen_ring(R=0.05, t=0.005, n_circ=32, n_rad=2):
    from cross_check_b3_gx import gen_ring

    return gen_ring(R=R, t=t, n_circ=n_circ, n_rad=n_rad)


def _gen_ellipse():
    from cross_check_b3_gx import gen_ellipse_solid

    return gen_ellipse_solid(a=0.06, b=0.02, n_circ=32, n_rad=8)


@dataclass
class Case:
    name: str
    title: str
    geom: Callable
    material: object
    # secfem angles
    beta: float
    alpha: float
    # gxbeam theta (rad) for fibre tilt; None = skip gxbeam
    gx_theta: float | None
    gx_swap23: bool = False
    kind: str = "ladder"


def ladder_cases() -> list[Case]:
    return [
        Case(
            "iso_rect",
            "isotropic rectangle",
            lambda: rectangle(0.04, 0.01, 8, 2)[:2],
            STEEL,
            0.0,
            0.0,
            0.0,
        ),
        Case(
            "iso_ring",
            "isotropic ring",
            _gen_ring,
            STEEL,
            0.0,
            0.0,
            0.0,
        ),
        Case(
            "asym_beam",
            "ASYM fibre on the beam",
            lambda: rectangle(0.04, 0.01, 8, 2)[:2],
            ASYM,
            0.0,
            90.0,
            0.0,
            kind="pin",
        ),
        Case(
            "asym_inplane",
            "ASYM fibre along +x",
            lambda: rectangle(0.04, 0.01, 8, 2)[:2],
            ASYM,
            0.0,
            0.0,
            np.pi / 2,
            gx_swap23=True,
            kind="pin",
        ),
        Case(
            "asym_tilt",
            "ASYM mid-tilt α=45",
            lambda: rectangle(0.04, 0.01, 8, 2)[:2],
            ASYM,
            0.0,
            45.0,
            np.pi / 4,
            kind="tilt",
        ),
        Case(
            "ud_ellipse_beam",
            "UD ellipse fibre on the beam",
            _gen_ellipse,
            CARBON_UD,
            0.0,
            90.0,
            0.0,
        ),
    ]


def _build_airfoil() -> dict:
    from validation.afmesh_naca import build_mesh

    mesh = build_mesh()
    # afmesh Layer angle 0 = spanwise. That is secfem α=90, ANBA fiber=90,
    # gxbeam θ=0. β is the shell tangent (secfem / ANBA only).
    n = mesh["quads"].shape[0]
    ply = np.asarray(mesh["alpha_deg"], float)
    mesh["secfem_alpha"] = 90.0 - ply
    mesh["secfem_beta"] = np.asarray(mesh["beta_deg"], float)
    mesh["anba_fiber"] = 90.0 - ply
    mesh["anba_plane"] = np.asarray(mesh["beta_deg"], float)
    mesh["gx_theta"] = np.radians(ply)
    mesh["n_cells"] = n
    return mesh


def _run_one(case: Case, work: Path, *, skip_anba: bool, skip_gx: bool) -> dict:
    coords, quads = case.geom()
    quads = ensure_ccw(coords, quads)
    mat = case.material
    place = secfem_to_anba_input(mat, case.beta, case.alpha)
    rec: dict = {
        "name": case.name,
        "title": case.title,
        "kind": case.kind,
        "n_cells": int(quads.shape[0]),
        "material": getattr(mat, "name", None),
        "secfem_angles": {"beta": case.beta, "alpha": case.alpha},
        "anba_angles": {"fiber": place.fiber_deg, "plane": place.plane_deg},
        "gx_theta_deg": None
        if case.gx_theta is None
        else float(np.degrees(case.gx_theta)),
        "gx_swap23": case.gx_swap23,
        "force_order": DOF,
    }
    sec = solve_secfem(coords, quads, mat, case.beta, case.alpha, work, case.name)
    rec["secfem"] = sec
    if isinstance(mat, OrthotropicMaterial):
        rec["closest_E"] = closest_modulus(sec["EA"], mat)
    if not skip_anba:
        try:
            an = solve_anba(coords, quads, mat, place.fiber_deg, place.plane_deg)
            rec["anba"] = an
            rec["rel_vs_anba"] = rel_per_dof(sec["K_diag"], an["K_diag"]).tolist()
        except Exception as exc:  # noqa: BLE001
            rec["anba"] = {"skipped": True, "reason": str(exc)[:400]}
    if not skip_gx and case.gx_theta is not None:
        try:
            gx = solve_gxbeam(
                coords, quads, mat, case.gx_theta, swap23=case.gx_swap23
            )
            rec["gxbeam"] = gx
            rec["rel_vs_gx"] = rel_per_dof(sec["K_diag"], gx["K_diag"]).tolist()
        except Exception as exc:  # noqa: BLE001
            rec["gxbeam"] = {"skipped": True, "reason": str(exc)[:400]}
    return rec


def _run_airfoil(work: Path, *, skip_anba: bool, skip_gx: bool) -> dict:
    mesh = _build_airfoil()
    mesh["quads"] = ensure_ccw(mesh["coords"], mesh["quads"])
    rec: dict = {
        "name": "naca0018",
        "title": "NACA 0018 hollow + caps + web",
        "kind": "airfoil",
        "n_cells": mesh["n_cells"],
        "geometry": {
            k: mesh[k]
            for k in ("naca", "chord", "web_loc", "skin_t", "spar_t", "web_t")
        },
        "counts": mesh["counts"],
        "force_order": DOF,
        "map": "ply 0° = spanwise → secfem α=90, ANBA fiber=90, gxbeam θ=0; "
        "β = shell tangent (secfem/ANBA only)",
        "coords": np.asarray(mesh["coords"]).tolist(),
        "quads": np.asarray(mesh["quads"]).tolist(),
        "roles": mesh["roles"],
    }
    sec = solve_secfem(
        mesh["coords"],
        mesh["quads"],
        mesh["mats"],
        mesh["secfem_beta"],
        mesh["secfem_alpha"],
        work,
        "naca0018",
    )
    rec["secfem"] = sec
    if not skip_anba:
        try:
            an = solve_anba(
                mesh["coords"],
                mesh["quads"],
                mesh["mats"],
                mesh["anba_fiber"],
                mesh["anba_plane"],
            )
            rec["anba"] = an
            rec["rel_vs_anba"] = rel_per_dof(sec["K_diag"], an["K_diag"]).tolist()
        except Exception as exc:  # noqa: BLE001
            rec["anba"] = {"skipped": True, "reason": str(exc)[:400]}
    if not skip_gx:
        try:
            gx = solve_gxbeam(
                mesh["coords"],
                mesh["quads"],
                mesh["mats"],
                mesh["gx_theta"],
                swap23=False,
            )
            rec["gxbeam"] = gx
            rec["rel_vs_gx"] = rel_per_dof(sec["K_diag"], gx["K_diag"]).tolist()
        except Exception as exc:  # noqa: BLE001
            rec["gxbeam"] = {"skipped": True, "reason": str(exc)[:400]}
    return rec


def _fmt_rel(rel) -> str:
    if rel is None:
        return "skip"
    return f"{max(rel):.2e}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-anba", action="store_true")
    p.add_argument("--skip-gx", action="store_true")
    p.add_argument("--skip-airfoil", action="store_true")
    p.add_argument("--airfoil-only", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "meshes"
    work.mkdir(exist_ok=True)

    print("three_way  —  same physical problem, mapped inputs")
    print("  fibre on beam:  secfem α=90  ANBA fiber=90  gxbeam θ=0")
    print("  fibre along x:  secfem α=0   ANBA fiber=0   gxbeam θ=90 (+2↔3 on gx)")
    print()
    print(f"{'case':<20} {'n':>4} {'EA':>11} {'~E':>4} {'vs ANBA':>10} {'vs gx':>10}")

    report = {
        "force_order": DOF,
        "map": {
            "anba": "fiber=α, plane=β, same card",
            "gxbeam": "θ=0 fibre on beam; θ=90 fibre in plane (2↔3 on gx if marked)",
        },
        "cases": [],
    }
    if args.airfoil_only:
        prev = OUT / "three_way.json"
        if prev.exists():
            report = json.loads(prev.read_text())
    for case in [] if args.airfoil_only else ladder_cases():
        rec = _run_one(
            case, work, skip_anba=args.skip_anba, skip_gx=args.skip_gx
        )
        report["cases"].append(rec)
        e = (rec.get("closest_E") or {}).get("closest", "—")
        an = rec.get("rel_vs_anba")
        gx = rec.get("rel_vs_gx")
        print(
            f"{case.name:<20} {rec['n_cells']:4d} {rec['secfem']['EA']:11.3e} "
            f"{str(e):>4} {_fmt_rel(an):>10} {_fmt_rel(gx):>10}"
        )
        if rec.get("anba", {}).get("skipped"):
            print(f"  anba skip: {rec['anba'].get('reason', '')[:80]}")
        if rec.get("gxbeam", {}).get("skipped"):
            print(f"  gx skip:   {rec['gxbeam'].get('reason', '')[:80]}")

    if not args.skip_airfoil:
        print()
        print("airfoil …")
        air = _run_airfoil(
            work, skip_anba=args.skip_anba, skip_gx=args.skip_gx
        )
        report["airfoil"] = air
        an = air.get("rel_vs_anba")
        gx = air.get("rel_vs_gx")
        print(
            f"{'naca0018':<20} {air['n_cells']:4d} {air['secfem']['EA']:11.3e} "
            f"{'—':>4} {_fmt_rel(an):>10} {_fmt_rel(gx):>10}"
        )
        if air.get("anba", {}).get("skipped"):
            print(f"  anba skip: {air['anba'].get('reason', '')[:80]}")
        if air.get("gxbeam", {}).get("skipped"):
            print(f"  gx skip:   {air['gxbeam'].get('reason', '')[:80]}")

    dest = OUT / "three_way.json"
    dest.write_text(json.dumps(report, indent=2))
    print()
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
