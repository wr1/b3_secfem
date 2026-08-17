#!/usr/bin/env python3
"""NACA 0018 hollow foil via afmesh: biax skin, carbon spar caps, one web.

Reasonably realistic mid-span station for the two3 check: skin is IEA
``glass_biax`` (E3 ≠ E1=E2), caps are carbon UD (E2=E3), one shear web
at 35 % chord. Per-cell ``β`` is the element tangent (shell plane);
``α`` is the ply fibre angle (0° = along the beam).

Runs secfem 1:1 vs the adapter 2↔3 swap on the same mesh. gxbeam is
optional (single ``theta`` = α; no β).

    micromamba run -n b3secfem python examples/validation/afmesh_naca.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parent.parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from validation._common import (  # noqa: E402
    CARBON_UD,
    FORCE_ORDER_SECFEM,
    GLASS_BIAX,
    OUT,
    anba_angles,
    rel_per_dof,
    swap23,
)
from _gx_vtu import write_gx_vtu  # noqa: E402
from b3_secfem import SectionInput, from_gxbeam_vtu, solve  # noqa: E402


def _cell_beta_deg(coords: np.ndarray, quad: np.ndarray) -> float:
    """Section-plane tangent of a quad, in degrees from +x.

    Picks the longer pair of opposite edges (skin/web tangent) rather
    than the through-thickness pair.
    """
    p = coords[quad]
    t0 = 0.5 * ((p[1] - p[0]) + (p[2] - p[3]))
    t1 = 0.5 * ((p[3] - p[0]) + (p[2] - p[1]))
    t = t0 if np.linalg.norm(t0) >= np.linalg.norm(t1) else t1
    return float(np.degrees(np.arctan2(t[1], t[0])))


def _role(name: str) -> str:
    n = name.lower()
    if "web" in n:
        return "web"
    if "carbon" in n or "cap" in n or "spar" in n:
        return "spar"
    return "skin"


def build_mesh() -> dict:
    from b3_af import Layer, Material, WebDef, afmesh, naca4, prepare_wire

    naca = "0018"
    chord = 1.0
    skin_t, spar_t, web_t = 0.004, 0.018, 0.006
    web_loc = 0.35
    ds = 0.05
    xaf, yaf = naca4(naca, n=80)
    # TE / LE / mid-chord skin; spar-cap band 18–50 % chord.
    xbreak = np.array([0.0, 0.18, 0.50, 1.0])
    ply_skin = Material("glass_biax")
    ply_cap = Material("carbon_ud")
    ply_web = Material("web_biax")
    skin = [Layer(ply_skin, skin_t, 0.0), Layer(ply_skin, skin_t, 0.0)]
    spar = [Layer(ply_skin, skin_t, 0.0), Layer(ply_cap, spar_t, 0.0)]
    web_layers = [Layer(ply_web, web_t, 0.0)]
    xu, _yu, xl, _yl, t_u, t_l, _t_le = prepare_wire(
        xaf, yaf, xbreak, chord=chord, ds=ds, te_strategy="zip",
    )
    x_w = web_loc * chord
    wd = WebDef(
        t_upper=float(np.interp(x_w, xu, t_u)),
        t_lower=float(np.interp(x_w, xl, t_l)),
        layers=web_layers,
        ne=6,
    )
    nodes, elements, _surf = afmesh(
        xaf, yaf,
        chord=chord, twist=0.0, paxis=0.5,
        xbreak=xbreak,
        webloc=np.array([]),
        segments=[skin, spar, skin],
        webs=[wd],
        ds=ds, wns=3,
    )
    coords = np.array([[n.x, n.y] for n in nodes], dtype=np.float64)
    quads = np.array([e.nodenum for e in elements], dtype=np.int64)
    names = [e.material.name for e in elements]
    alpha = np.degrees(np.array([e.theta for e in elements], dtype=float))
    beta = np.array(
        [_cell_beta_deg(coords, q) for q in quads], dtype=float
    )
    catalog = {
        "glass_biax": GLASS_BIAX,
        "carbon_ud": CARBON_UD,
        "web_biax": GLASS_BIAX.model_copy(update={"name": "web_biax"}),
    }
    mats = [catalog[n] for n in names]
    roles = [_role(n) for n in names]
    counts = {r: roles.count(r) for r in ("skin", "spar", "web")}
    return {
        "naca": naca, "chord": chord, "web_loc": web_loc,
        "skin_t": skin_t, "spar_t": spar_t, "web_t": web_t,
        "coords": coords, "quads": quads,
        "names": names, "roles": roles, "counts": counts,
        "alpha_deg": alpha, "beta_deg": beta, "mats": mats,
    }


def _row(mat) -> np.ndarray:
    return np.array(
        [mat.E1, mat.E2, mat.E3, mat.G12, mat.G13, mat.G23,
         mat.nu12, mat.nu13, mat.nu23, mat.rho],
        dtype=np.float64,
    )


def _solve_secfem(mesh: dict, *, swap: bool, work: Path) -> dict:
    mats = [swap23(m) if swap else m for m in mesh["mats"]]
    vtu = work / ("naca_swap23.vtu" if swap else "naca_1to1.vtu")
    write_gx_vtu(
        vtu, mesh["coords"], mesh["quads"],
        np.stack([_row(m) for m in mats]),
        np.radians(mesh["alpha_deg"]),
    )
    info = from_gxbeam_vtu(vtu)
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=mats,
        per_cell_beta_deg=mesh["beta_deg"],
        per_cell_alpha_deg=mesh["alpha_deg"],
        backend="fenicsx",
        linear_solver="lu",
        degree=2,
    )
    res = solve(inp)
    K = np.asarray(res.K, dtype=float)
    return {
        "K": K.tolist(),
        "K_diag": np.diag(K).tolist(),
        "EA": float(K[2, 2]),
        "EIx": float(K[3, 3]),
        "EIy": float(K[4, 4]),
        "GJ": float(K[5, 5]),
        "shear_center": [float(x) for x in res.shear_center],
        "n_cells": int(info["n_cells"]),
        "swapped": swap,
    }


def _try_gxbeam(mesh: dict, *, swap: bool) -> dict | None:
    try:
        from cross_check_b3_gx import _gxbeam_section_from_arrays
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": str(exc)[:300]}
    mats = [swap23(m) if swap else m for m in mesh["mats"]]
    mp = np.stack([_row(m) for m in mats])
    try:
        K, *_rest = _gxbeam_section_from_arrays(
            mesh["coords"], mesh["quads"], mp, np.radians(mesh["alpha_deg"]),
        )
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": str(exc)[:400]}
    K = np.asarray(K, dtype=float)
    return {
        "K_gxbeam_order": K.tolist(),
        "K_diag_gx": np.diag(K).tolist(),
        "EA_gx_F1": float(K[0, 0]),
        "swapped": swap,
        "skipped": False,
        "note": "gxbeam theta = α only; β (shell tangent) is not sent",
    }


def _try_anba(mesh: dict) -> dict:
    try:
        from _anba_runner_host import anba_section_from_arrays
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": str(exc)[:300]}
    fiber = []
    plane = []
    for a, b in zip(mesh["alpha_deg"], mesh["beta_deg"], strict=True):
        f, p = anba_angles(float(b), float(a))
        fiber.append(f)
        plane.append(p)
    # unique materials + ids
    mats = mesh["mats"]
    names = [m.name for m in mats]
    uniq, ids = [], []
    index = {}
    for m, n in zip(mats, names, strict=True):
        if n not in index:
            index[n] = len(uniq)
            uniq.append(m)
        ids.append(index[n])
    try:
        out = anba_section_from_arrays(
            mesh["coords"],
            mesh["quads"],
            uniq,
            fiber_orientation_deg=np.asarray(fiber, float),
            plane_orientation_deg=np.asarray(plane, float),
            material_id=np.asarray(ids, dtype=np.int64),
        )
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": str(exc)[:400]}
    K = np.asarray(out["K"], dtype=float)
    return {
        "skipped": False,
        "K": K.tolist(),
        "K_diag": np.diag(K).tolist(),
        "EA": float(K[2, 2]),
    }


def _relmax(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-30)))


def _plot(mesh: dict, rec: dict) -> None:
    from validation._viz import plot_afmesh_k, plot_afmesh_mesh

    dest_png = OUT / "afmesh_naca.png"
    plot_afmesh_mesh(mesh, rec, dest_png)
    print(f"wrote {dest_png}")
    dest_k = plot_afmesh_k(rec)
    if dest_k:
        print(f"wrote {dest_k}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-gxbeam", action="store_true")
    p.add_argument("--skip-plot", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "meshes"
    work.mkdir(exist_ok=True)

    print("building NACA 0018 + spar caps + shear web (afmesh) …")
    mesh = build_mesh()
    print(
        f"  n_cells={mesh['quads'].shape[0]}  "
        + "  ".join(f"{k}={v}" for k, v in mesh["counts"].items())
    )
    print(
        f"  β (tangent) range [{mesh['beta_deg'].min():+.1f}, "
        f"{mesh['beta_deg'].max():+.1f}] deg"
    )

    one = _solve_secfem(mesh, swap=False, work=work)
    sw = _solve_secfem(mesh, swap=True, work=work)
    rel = _relmax(one["K_diag"], sw["K_diag"])
    labels = FORCE_ORDER_SECFEM
    print()
    print(f"{'':8}" + "  ".join(f"{lab:>10}" for lab in labels))
    print("1:1     " + "  ".join(f"{v:10.3e}" for v in one["K_diag"]))
    print("swap23  " + "  ".join(f"{v:10.3e}" for v in sw["K_diag"]))
    drel = [
        abs(a - b) / max(abs(a), 1e-30)
        for a, b in zip(one["K_diag"], sw["K_diag"], strict=True)
    ]
    print("rel     " + "  ".join(f"{100*v:9.2f}%" for v in drel))
    print(f"K diag relmax 1:1 vs swap23 = {rel:.3e}")
    print(
        f"sc 1:1 = ({one['shear_center'][0]:+.4f}, {one['shear_center'][1]:+.4f})"
    )

    gx = None
    if not args.skip_gxbeam:
        gx = _try_gxbeam(mesh, swap=False)
        if gx and not gx.get("skipped"):
            print(f"gxbeam EA (F1) = {gx['EA_gx_F1']:.3e}  (θ=α only)")
        elif gx:
            print(f"gxbeam skip: {gx.get('reason', '')[:80]}")

    anba = _try_anba(mesh)
    if anba and not anba.get("skipped"):
        rel_an = _relmax(one["K_diag"], anba["K_diag"])
        anba["Kdiag_relmax_vs_secfem_1to1"] = rel_an
        print("anba     " + "  ".join(f"{v:10.3e}" for v in anba["K_diag"]))
        print(f"anba vs secfem 1:1 Kdiag relmax = {rel_an:.3e}")
    elif anba:
        print(f"anba skip: {anba.get('reason', '')[:80]}")

    rec = {
        "geometry": {
            k: mesh[k]
            for k in ("naca", "chord", "web_loc", "skin_t", "spar_t", "web_t")
        },
        "counts": mesh["counts"],
        "n_cells": int(mesh["quads"].shape[0]),
        "beta_deg_minmax": [
            float(mesh["beta_deg"].min()),
            float(mesh["beta_deg"].max()),
        ],
        "force_order": FORCE_ORDER_SECFEM,
        "secfem_1to1": one,
        "secfem_swap23": sw,
        "Kdiag_relmax_1to1_vs_swap23": rel,
        "Kdiag_rel_per_dof": rel_per_dof(one["K_diag"], sw["K_diag"]).tolist(),
        "gxbeam": gx,
        "anba": anba,
        "note": (
            "skin+web = IEA glass_biax (E3≠E1=E2); spar = carbon UD (E2=E3). "
            "β = element tangent; α = ply fibre (0°). "
            "swap23 is the NLR b3_fix / option-A adapter, not a secfem change."
        ),
    }
    dest_json = OUT / "afmesh_naca.json"
    dest_json.write_text(json.dumps(rec, indent=2))
    print(f"wrote {dest_json}")

    if not args.skip_plot:
        _plot(mesh, rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
