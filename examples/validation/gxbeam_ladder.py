#!/usr/bin/env python3
"""gxbeam section ladder on secfem (fenicsx, then mfem) and optional ANBA.

Wraps ``examples/cross_check_b3_gx.py`` and adds the E2≠E3 case the
existing UD carbon ladder cannot see.

    micromamba run -n b3secfem python examples/validation/gxbeam_ladder.py
    micromamba run -n b3secfem python examples/validation/gxbeam_ladder.py --full --anba
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
    ASYM,
    FORCE_ORDER_SECFEM,
    OUT,
    closest_modulus,
    from_gxbeam_order,
    rel_per_dof,
)

from _gx_vtu import write_gx_vtu  # noqa: E402
from b3_secfem import (  # noqa: E402
    OrthotropicMaterial,
    SectionInput,
    from_gxbeam_vtu,
    solve,
    to_gxbeam_order,
)
from cross_check_b3_gx import (  # noqa: E402
    CASES,
    Case,
    _gxbeam_section_from_arrays,
    gen_ellipse_solid,
    gen_ring,
    iso_row,
    ortho_row,
)


def _gx_row_swap23(mat: OrthotropicMaterial) -> np.ndarray:
    """Same bridge as b3_section.backends.gxbeam._gx_row_with_axis_swap."""
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
        ],
        dtype=np.float64,
    )


def _run_secfem(coords, quads, mat, alpha_deg, backend: str, vtu: Path) -> dict:
    n = quads.shape[0]
    row = iso_row(mat) if not isinstance(mat, OrthotropicMaterial) else ortho_row(mat)
    write_gx_vtu(
        vtu, coords, quads, np.tile(row, (n, 1)), np.full(n, np.radians(alpha_deg))
    )
    info = from_gxbeam_vtu(vtu)
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=[mat] * info["n_cells"],
        per_cell_alpha_deg=np.degrees(info["theta"]),
        backend=backend,  # type: ignore[arg-type]
        linear_solver="lu",
    )
    res = solve(inp)
    return {
        "K_secfem_order": np.asarray(res.K).tolist(),
        "K_gxbeam_order": to_gxbeam_order(np.asarray(res.K)).tolist(),
        "EA": float(res.K[2, 2]),
    }


def _try_anba(coords, quads, mat, alpha_deg) -> dict:
    from _anba_runner_host import anba_section_from_arrays

    n = quads.shape[0]
    out = anba_section_from_arrays(
        coords,
        quads,
        mat,
        fiber_orientation_deg=np.full(n, alpha_deg),
        plane_orientation_deg=np.zeros(n),
    )
    K = np.asarray(out["K"])
    return {
        "K_secfem_order": K.tolist(),
        "K_gxbeam_order": to_gxbeam_order(K).tolist(),
        "EA": float(K[2, 2]),
    }


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))


def _ladder_cases(full: bool) -> list[Case]:
    extra = [
        Case(
            "asym_ring_a0",
            lambda: gen_ring(R=0.05, t=0.005, n_circ=32, n_rad=2),
            ASYM,
            theta_rad=0.0,
            is_iso=False,
        ),
        Case(
            "asym_ring_a90",
            lambda: gen_ring(R=0.05, t=0.005, n_circ=32, n_rad=2),
            ASYM,
            theta_rad=np.pi / 2,
            is_iso=False,
        ),
        Case(
            "asym_ellipse_a0",
            lambda: gen_ellipse_solid(a=0.06, b=0.02, n_circ=32, n_rad=8),
            ASYM,
            theta_rad=0.0,
            is_iso=False,
        ),
    ]
    if full:
        return list(CASES) + extra
    # Quick: iso ring + UD (E2=E3) + the discriminating asym cases.
    keep = {"iso_ring", "ud_ellipse_t0", "ud_ellipse_t45"}
    return [c for c in CASES if c.name in keep] + extra


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    p.add_argument("--anba", action="store_true", default=True)
    p.add_argument("--skip-anba", action="store_true")
    p.add_argument(
        "--mfem",
        action="store_true",
        help="also solve each case with mfem (own process)",
    )
    p.add_argument(
        "--gx-swap23",
        action="store_true",
        help="feed gxbeam the b3_section 2↔3-bridged row",
    )
    args = p.parse_args()
    if args.skip_anba:
        args.anba = False

    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "meshes"
    work.mkdir(exist_ok=True)
    cases = _ladder_cases(args.full)
    report = {"full": args.full, "gx_swap23": args.gx_swap23, "cases": []}

    print(
        f"{'case':<22} {'α':>5} {'EA_s/A?':>10} {'gx relK':>10} "
        f"{'gx-swap':>8} {'anba':>10}"
    )
    for case in cases:
        coords, quads = case.geom()
        n = quads.shape[0]
        alpha = float(np.degrees(case.theta_rad))
        if case.is_iso:
            row = iso_row(case.material)
            row_swap = row
        else:
            row = ortho_row(case.material)
            row_swap = (
                _gx_row_swap23(case.material)
                if isinstance(case.material, OrthotropicMaterial)
                else row
            )
        mp = np.tile(row_swap if args.gx_swap23 else row, (n, 1))
        th = np.full(n, case.theta_rad)
        K_gx, *_rest = _gxbeam_section_from_arrays(coords, quads, mp, th)
        K_gx = np.asarray(K_gx)

        vtu = work / f"{case.name}_fx.vtu"
        sec = _run_secfem(coords, quads, case.material, alpha, "fenicsx", vtu)
        K_us = np.asarray(sec["K_gxbeam_order"])
        rel = _rel(np.diag(K_us), np.diag(K_gx))

        anba_rel = None
        anba_blk = None
        if args.anba:
            try:
                anba_blk = _try_anba(coords, quads, case.material, alpha)
                K_an = np.asarray(anba_blk["K_secfem_order"], float)
                anba_blk["K_diag_secfem"] = np.diag(K_an).tolist()
                anba_rel = _rel(np.diag(K_us_wait := np.asarray(sec["K_secfem_order"], float)), np.diag(K_an))
                anba_blk["Kdiag_rel_per_dof"] = rel_per_dof(
                    np.diag(K_us_wait), np.diag(K_an)
                ).tolist()
            except Exception as exc:  # noqa: BLE001
                anba_blk = {"skipped": True, "reason": str(exc)[:400]}

        closest = None
        if isinstance(case.material, OrthotropicMaterial):
            # ring/ellipse area from mesh bbox is wrong; use EA vs E*A later
            closest = closest_modulus(sec["EA"], case.material)

        K_us_sec = np.asarray(sec["K_secfem_order"], float)
        K_gx_sec = from_gxbeam_order(K_gx)
        rec = {
            "name": case.name,
            "alpha_deg": alpha,
            "is_iso": case.is_iso,
            "n_cells": n,
            "force_order": FORCE_ORDER_SECFEM,
            "K_gx": K_gx.tolist(),
            "secfem_fenicsx": sec,
            "K_diag_secfem": np.diag(K_us_sec).tolist(),
            "K_diag_gx_secfem_order": np.diag(K_gx_sec).tolist(),
            "Kdiag_rel_per_dof": rel_per_dof(np.diag(K_us_sec), np.diag(K_gx_sec)).tolist(),
            "Kdiag_relmax_vs_gx": rel,
            "anba": anba_blk,
            "Kdiag_relmax_vs_anba": anba_rel,
        }
        report["cases"].append(rec)
        anba_s = f"{anba_rel:.2e}" if anba_rel is not None else "skip"
        print(
            f"{case.name:<22} {alpha:5.1f} {sec['EA']:10.3e} {rel:10.2e} "
            f"{'yes' if args.gx_swap23 else 'no':>8} {anba_s:>10}"
        )

        if args.mfem:
            try:
                mf = _run_secfem(
                    coords,
                    quads,
                    case.material,
                    alpha,
                    "mfem",
                    work / f"{case.name}_mf.vtu",
                )
                rec["secfem_mfem"] = mf
                rec["Kdiag_relmax_fx_vs_mfem"] = _rel(
                    np.diag(K_us),
                    np.diag(np.asarray(mf["K_gxbeam_order"])),
                )
                print(
                    f"{'  mfem':<22} {alpha:5.1f} {mf['EA']:10.3e} "
                    f"{rec['Kdiag_relmax_fx_vs_mfem']:10.2e}"
                )
            except Exception as exc:  # noqa: BLE001
                rec["secfem_mfem"] = {"error": str(exc)[:400]}
                print(f"  mfem failed: {exc}")

        _ = closest  # EA-vs-E*A needs area; printed via two3_probe

    dest = OUT / (
        "gxbeam_ladder_swap23.json" if args.gx_swap23 else "gxbeam_ladder.json"
    )
    dest.write_text(json.dumps(report, indent=2))
    print(f"wrote {dest}")
    try:
        from validation._viz import plot_gxbeam

        raw_p = OUT / "gxbeam_ladder.json"
        sw_p = OUT / "gxbeam_ladder_swap23.json"
        raw = json.loads(raw_p.read_text()) if raw_p.exists() else None
        sw = json.loads(sw_p.read_text()) if sw_p.exists() else None
        if raw:
            p = plot_gxbeam(raw, sw)
            if p:
                print(f"wrote {p}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot skip: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
