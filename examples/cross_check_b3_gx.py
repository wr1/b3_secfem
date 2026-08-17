"""Cross-check b3_secfem against gxbeam_section over a small geometry ladder.

For each case, a single gxbeam-format VTU is written and ingested by both
engines so they see the exact same mesh and per-cell material data:

  * gxbeam reference: ``compute_section_from_mesh(coords, quads, mat_props, theta)``
    (the VTU-loading path silently uses a hard-coded Al material — we go via
    the array entry point so the per-cell mat_props are actually consumed).
  * b3_secfem SUT: ``from_gxbeam_vtu(p)`` then ``solve(SectionInput(...))``.

Outputs:

  * ``examples/cross_check_out/results.json`` — full per-case dump (K, M,
    centres for both engines plus relative-error metrics).
  * Markdown table to stdout — quick eyeball summary.

Run::

    uv run python examples/cross_check_b3_gx.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Load dolfinx BEFORE juliacall starts. Julia ships its own libcurl.so.4
# which lacks the CURL_OPENSSL_4 symbols required by system libhdf5_openmpi
# (pulled in by dolfinx). Importing dolfinx first pins the system libcurl.
import dolfinx  # noqa: F401, E402

import numpy as np

from b3_secfem import (
    IsotropicMaterial,
    OrthotropicMaterial,
    SectionInput,
    from_gxbeam_vtu,
    solve,
    to_gxbeam_order,
)

sys.path.insert(0, str(Path(__file__).parent))
from _anba_runner_host import anba_section_from_arrays  # noqa: E402
from _gx_vtu import write_gx_vtu  # noqa: E402


# ── materials ────────────────────────────────────────────────────────────────

STEEL = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0, name="steel")

UD_NU23 = 0.4
UD_E2 = 10e9
UD_CARBON = OrthotropicMaterial(
    E1=135e9, E2=UD_E2, E3=UD_E2,
    G12=5e9, G13=5e9, G23=UD_E2 / (2 * (1 + UD_NU23)),
    nu12=0.3, nu13=0.3, nu23=UD_NU23,
    rho=1600.0, name="UD carbon",
)


def iso_row(mat: IsotropicMaterial) -> np.ndarray:
    """gxbeam ``mat_props`` row for an isotropic material."""
    G = mat.E / (2 * (1 + mat.nu))
    return np.array(
        [mat.E, mat.E, mat.E, G, G, G, mat.nu, mat.nu, mat.nu, mat.rho],
        dtype=np.float64,
    )


def ortho_row(mat: OrthotropicMaterial) -> np.ndarray:
    """gxbeam ``mat_props`` row for an orthotropic material.

    Layout: ``[E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho]``
    (matches the GXBeam.Material constructor ordering used in
    `b3_gx/src/gxbeam_section/backend/section_from_vtk.jl:92`).
    """
    return np.array(
        [
            mat.E1, mat.E2, mat.E3,
            mat.G12, mat.G13, mat.G23,
            mat.nu12, mat.nu13, mat.nu23,
            mat.rho,
        ],
        dtype=np.float64,
    )


# ── geometry generators ──────────────────────────────────────────────────────


def gen_ring(R: float, t: float, n_circ: int, n_rad: int) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    for k in range(n_rad + 1):
        r = R - t / 2 + k * t / n_rad
        for i in range(n_circ):
            ang = 2 * np.pi * i / n_circ
            coords.append([r * np.cos(ang), r * np.sin(ang)])
    quads = []
    for k in range(n_rad):
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            n0 = k * n_circ + i
            n1 = (k + 1) * n_circ + i
            n2 = (k + 1) * n_circ + i_next
            n3 = k * n_circ + i_next
            quads.append([n0, n1, n2, n3])
    return np.asarray(coords, dtype=np.float64), np.asarray(quads, dtype=np.int64)


def gen_ellipse_solid(
    a: float, b: float, n_circ: int, n_rad: int, hole_frac: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    for k in range(n_rad + 1):
        r = hole_frac + k * (1.0 - hole_frac) / n_rad
        for i in range(n_circ):
            ang = 2 * np.pi * i / n_circ
            coords.append([r * a * np.cos(ang), r * b * np.sin(ang)])
    quads = []
    for k in range(n_rad):
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            n0 = k * n_circ + i
            n1 = (k + 1) * n_circ + i
            n2 = (k + 1) * n_circ + i_next
            n3 = k * n_circ + i_next
            quads.append([n0, n1, n2, n3])
    return np.asarray(coords, dtype=np.float64), np.asarray(quads, dtype=np.int64)


def gen_ellipse_hollow(
    a: float, b: float, t: float, n_circ: int, n_rad: int,
) -> tuple[np.ndarray, np.ndarray]:
    a_i, b_i = a - t, b - t
    coords = []
    for k in range(n_rad + 1):
        sa = a_i + k * (a - a_i) / n_rad
        sb = b_i + k * (b - b_i) / n_rad
        for i in range(n_circ):
            ang = 2 * np.pi * i / n_circ
            coords.append([sa * np.cos(ang), sb * np.sin(ang)])
    quads = []
    for k in range(n_rad):
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            n0 = k * n_circ + i
            n1 = (k + 1) * n_circ + i
            n2 = (k + 1) * n_circ + i_next
            n3 = k * n_circ + i_next
            quads.append([n0, n1, n2, n3])
    return np.asarray(coords, dtype=np.float64), np.asarray(quads, dtype=np.int64)


def gen_airfoil_solid(
    chord: float, thickness: float, n_chord: int, n_thick: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solid (filled) NACA 00xx airfoil, mirrors tests/_meshlib.airfoil_solid."""
    eta = np.linspace(0.0, 1.0, n_chord + 1)
    half_t = (thickness / 0.2) * chord * (
        0.2969 * np.sqrt(eta) - 0.1260 * eta - 0.3516 * eta ** 2
        + 0.2843 * eta ** 3 - 0.1036 * eta ** 4
    )
    xi = np.linspace(-1.0, 1.0, n_thick + 1)
    coords = np.zeros(((n_chord + 1) * (n_thick + 1), 2))
    for i, e in enumerate(eta):
        for j, s in enumerate(xi):
            coords[i * (n_thick + 1) + j] = [chord * (e - 0.5), s * half_t[i]]
    quads = []
    for i in range(n_chord):
        for j in range(n_thick):
            n00 = i * (n_thick + 1) + j
            n10 = (i + 1) * (n_thick + 1) + j
            n11 = (i + 1) * (n_thick + 1) + j + 1
            n01 = i * (n_thick + 1) + j + 1
            quads.append([n00, n10, n11, n01])
    return coords, np.asarray(quads, dtype=np.int64)


# ── case definitions ─────────────────────────────────────────────────────────


@dataclass
class Case:
    name: str
    geom: Callable[[], tuple[np.ndarray, np.ndarray]]
    material: Any
    theta_rad: float = 0.0
    is_iso: bool = True


CASES: list[Case] = [
    Case(
        "iso_ring",
        lambda: gen_ring(R=0.05, t=0.005, n_circ=64, n_rad=4),
        STEEL,
        is_iso=True,
    ),
    Case(
        "iso_ellipse_solid",
        lambda: gen_ellipse_solid(a=0.06, b=0.02, n_circ=64, n_rad=12),
        STEEL,
        is_iso=True,
    ),
    Case(
        "iso_ellipse_hollow",
        lambda: gen_ellipse_hollow(a=0.06, b=0.02, t=0.005, n_circ=64, n_rad=4),
        STEEL,
        is_iso=True,
    ),
    Case(
        "ud_ellipse_t0",
        lambda: gen_ellipse_solid(a=0.06, b=0.02, n_circ=64, n_rad=12),
        UD_CARBON,
        theta_rad=0.0,
        is_iso=False,
    ),
    Case(
        "ud_ellipse_t45",
        lambda: gen_ellipse_solid(a=0.06, b=0.02, n_circ=64, n_rad=12),
        UD_CARBON,
        theta_rad=np.pi / 4,
        is_iso=False,
    ),
    Case(
        "iso_airfoil_solid",
        lambda: gen_airfoil_solid(chord=1.0, thickness=0.24, n_chord=32, n_thick=8),
        STEEL,
        is_iso=True,
    ),
    Case(
        "ud_airfoil_t0",
        lambda: gen_airfoil_solid(chord=1.0, thickness=0.24, n_chord=32, n_thick=8),
        UD_CARBON,
        theta_rad=0.0,
        is_iso=False,
    ),
    Case(
        "ud_airfoil_t45",
        lambda: gen_airfoil_solid(chord=1.0, thickness=0.24, n_chord=32, n_thick=8),
        UD_CARBON,
        theta_rad=np.pi / 4,
        is_iso=False,
    ),
]


# ── driver ───────────────────────────────────────────────────────────────────


def _gxbeam_section_from_arrays(coords, quads, mat_props, theta):
    """Call GXBeam directly via juliacall.

    Bypasses ``gxbeam_section.compute_section_from_mesh`` because its
    ``_unpack`` does ``np.array(julia_arr)`` which, on numpy<2 paired with
    a recent juliacall, raises ``ValueError: NoneType copy mode not allowed``
    (the Julia ArrayValue's ``__array__`` forwards ``copy=None``). We use
    ``np.asarray`` which routes through ``__array_interface__`` instead.

    numpy<2 is not optional here: petsc4py from system apt is built against
    numpy 1.x ABI, so the b3_secfem env is locked to that.
    """
    import gxbeam_section  # noqa: F401  - side effect: starts Julia, loads jl.compute_section_from_arrays
    from juliacall import Main as jl

    xy = np.asfortranarray(coords, dtype=np.float64)
    ec = np.asfortranarray(np.asarray(quads, dtype=np.int64) + 1)
    mp = np.asfortranarray(mat_props, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    res = jl.compute_section_from_arrays(xy, ec, mp, th)
    # GXBeam returns M as a Symmetric{Float64}; on numpy<2 the Symmetric
    # wrapper has no __array_interface__ so np.asarray falls through to
    # __array__(copy=None) which is rejected. Materialise via jl.Matrix
    # before crossing the boundary.
    return tuple(np.asarray(jl.Matrix(res[i])) if i in (0, 1, 2)
                 else np.asarray(res[i]) for i in range(6))


def run_case(case: Case, out_dir: Path, with_anba: bool = False) -> dict:
    """Build VTU, run gxbeam_section + b3_secfem (+ optional ANBA), compare."""
    coords, quads = case.geom()
    n_cells = quads.shape[0]
    row = iso_row(case.material) if case.is_iso else ortho_row(case.material)
    mat_props = np.tile(row, (n_cells, 1))
    theta = np.full(n_cells, case.theta_rad)

    vtu = out_dir / f"{case.name}.vtu"
    write_gx_vtu(vtu, coords, quads, mat_props, theta)

    K_gx, _S_gx, M_gx, sc_gx, tc_gx, mc_gx = _gxbeam_section_from_arrays(
        coords, quads, mat_props, theta,
    )
    sc_gx = sc_gx.reshape(-1)
    tc_gx = tc_gx.reshape(-1)
    mc_gx = mc_gx.reshape(-1)

    info = from_gxbeam_vtu(vtu)
    # gxbeam's per-element ``theta`` is the out-of-plane fibre tilt, not the
    # in-plane rotation about z. Maps to b3_secfem's ``alpha_deg``. Verified
    # in examples/probe_theta_convention.py: K[Fz,Fz] matches gxbeam exactly
    # at θ ∈ {0°, 90°}; a residual ~5–12 % gap remains at mid-range angles
    # from a rotation-axis convention difference noted in notes/claude.md.
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=[case.material] * info["n_cells"],
        per_cell_alpha_deg=np.degrees(info["theta"]),
    )
    res = solve(inp)
    K_us = to_gxbeam_order(res.K)
    M_us = to_gxbeam_order(res.M)

    K_anba = None
    if with_anba:
        # Same zero as secfem: fiber = α, plane = β (= 0 here).
        alpha_deg = np.degrees(case.theta_rad)
        anba = anba_section_from_arrays(
            coords, quads, case.material,
            fiber_orientation_deg=np.full(n_cells, alpha_deg),
            plane_orientation_deg=np.full(n_cells, 0.0),
        )
        # ANBA returns K in [Fx, Fy, Fz, Mx, My, Mz] — same as b3_secfem.K.
        # Permute to gxbeam order so we can compare against K_gx.
        K_anba = to_gxbeam_order(anba["K"])

    return _compare(
        case, n_cells, K_gx, K_us, M_gx, M_us,
        sc_gx, tc_gx, mc_gx,
        np.asarray(res.shear_center),
        np.asarray(res.tension_center),
        np.asarray(res.mass_center),
        K_anba=K_anba,
    )


def _compare(
    case, n_cells, K_gx, K_us, M_gx, M_us,
    sc_gx, tc_gx, mc_gx, sc_us, tc_us, mc_us,
    K_anba=None,
):
    eps = 1e-30
    diag_gx = np.diag(K_gx)
    diag_us = np.diag(K_us)
    K_diag_rel_err = np.abs(diag_us - diag_gx) / np.maximum(np.abs(diag_gx), eps)

    gmean = np.sqrt(np.outer(np.abs(diag_gx), np.abs(diag_gx)))
    gmean[gmean == 0] = 1.0
    off = (K_us - K_gx) - np.diag(np.diag(K_us - K_gx))
    K_offdiag_rel_max = float(np.max(np.abs(off) / gmean))

    M00_rel_err = abs(M_us[0, 0] - M_gx[0, 0]) / max(abs(M_gx[0, 0]), eps)

    sc_l2 = float(np.linalg.norm(sc_us - sc_gx))
    tc_l2 = float(np.linalg.norm(tc_us - tc_gx))
    mc_l2 = float(np.linalg.norm(mc_us - mc_gx))

    K_tol = 0.02 if case.is_iso else 0.05
    c_tol = 1e-4 if case.is_iso else 1e-3
    K_diag_ok = bool(np.all(K_diag_rel_err < K_tol))
    centres_ok = (sc_l2 < c_tol) and (tc_l2 < c_tol) and (mc_l2 < c_tol)
    ok = bool(K_diag_ok and centres_ok)

    rec = {
        "name": case.name,
        "n_cells": int(n_cells),
        "is_iso": bool(case.is_iso),
        "theta_deg": float(np.degrees(case.theta_rad)),
        "K_gxbeam": K_gx.tolist(),
        "K_b3_secfem": K_us.tolist(),
        "K_diag_rel_err": K_diag_rel_err.tolist(),
        "K_offdiag_rel_max": K_offdiag_rel_max,
        "M00_gxbeam": float(M_gx[0, 0]),
        "M00_b3_secfem": float(M_us[0, 0]),
        "M00_rel_err": float(M00_rel_err),
        "centres_gxbeam": {
            "sc": sc_gx.tolist(), "tc": tc_gx.tolist(), "mc": mc_gx.tolist(),
        },
        "centres_b3_secfem": {
            "sc": sc_us.tolist(), "tc": tc_us.tolist(), "mc": mc_us.tolist(),
        },
        "centre_l2_m": {"sc": sc_l2, "tc": tc_l2, "mc": mc_l2},
        "ok": ok,
    }

    if K_anba is not None:
        diag_anba = np.diag(K_anba)
        anba_diag_rel_err_vs_gx = np.abs(diag_anba - diag_gx) / np.maximum(np.abs(diag_gx), eps)
        anba_diag_rel_err_vs_us = np.abs(diag_anba - diag_us) / np.maximum(np.abs(diag_us), eps)
        rec.update({
            "K_anba": K_anba.tolist(),
            "K_diag_rel_err_anba_vs_gx": anba_diag_rel_err_vs_gx.tolist(),
            "K_diag_rel_err_anba_vs_b3": anba_diag_rel_err_vs_us.tolist(),
        })
    return rec


def _print_md_table(records: list[dict], with_anba: bool = False) -> None:
    cols = ["case", "n", "EA", "GAx", "GAy", "GJ", "EIx", "EIy",
            "M00", "sc[m]", "tc[m]", "mc[m]", "ok"]
    widths = [22, 5, 7, 7, 7, 7, 7, 7, 7, 9, 9, 9, 3]

    def pct(x: float) -> str:
        if not np.isfinite(x):
            return "    inf"
        return f"{100.0 * x:6.2f}%"

    def m(x: float) -> str:
        if not np.isfinite(x):
            return "    inf"
        return f"{x:8.1e}"

    head = " | ".join(f"{c:<{w}}" for c, w in zip(cols, widths, strict=True))
    sep = "-+-".join("-" * w for w in widths)
    print(head)
    print(sep)
    for r in records:
        if "K_diag_rel_err" not in r:
            print(f"{r['name']:<22} | error: {r.get('error', '?')}")
            continue
        e = r["K_diag_rel_err"]
        c = r["centre_l2_m"]
        cells = [
            f"{r['name']:<22}",
            f"{r['n_cells']:>5d}",
            pct(e[0]), pct(e[1]), pct(e[2]),
            pct(e[3]), pct(e[4]), pct(e[5]),
            pct(r["M00_rel_err"]),
            m(c["sc"]), m(c["tc"]), m(c["mc"]),
            " ✓ " if r["ok"] else " ✗ ",
        ]
        print(" | ".join(cells))
        if with_anba and "K_diag_rel_err_anba_vs_gx" in r:
            ea = r["K_diag_rel_err_anba_vs_gx"]
            eb = r["K_diag_rel_err_anba_vs_b3"]
            blanks = [
                f"{'':<22}", f"{'':>5}",
                pct(ea[0]), pct(ea[1]), pct(ea[2]), pct(ea[3]), pct(ea[4]), pct(ea[5]),
                f"{'':<7}", f"{'':<9}", f"{'':<9}", f"{'':<9}", f"{'ANBA-gx':<3}",
            ]
            print(" | ".join(blanks))
            blanks = [
                f"{'':<22}", f"{'':>5}",
                pct(eb[0]), pct(eb[1]), pct(eb[2]), pct(eb[3]), pct(eb[4]), pct(eb[5]),
                f"{'':<7}", f"{'':<9}", f"{'':<9}", f"{'':<9}", f"{'ANBA-b3':<3}",
            ]
            print(" | ".join(blanks))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--anba", action="store_true",
                    help="also run ANBA4 in Docker as a third reference")
    ap.add_argument("--cases", default="",
                    help="comma-separated case names to run (default: all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    out_dir = Path(__file__).parent / "cross_check_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = CASES
    if args.cases:
        wanted = {n.strip() for n in args.cases.split(",") if n.strip()}
        selected = [c for c in CASES if c.name in wanted]
        if not selected:
            msg = f"no cases match: {wanted}; have {[c.name for c in CASES]}"
            raise SystemExit(msg)

    records: list[dict] = []
    for case in selected:
        print(f"Running {case.name} ...", flush=True)
        try:
            rec = run_case(case, out_dir, with_anba=args.anba)
        except Exception as e:
            logging.getLogger(__name__).exception("case %s failed", case.name)
            rec = {"name": case.name, "error": repr(e), "ok": False}
        records.append(rec)

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "tolerances": {
            "isotropic": {"K_diag": 0.02, "centre_m": 1e-4},
            "composite": {"K_diag": 0.05, "centre_m": 1e-3},
            "excluded_K_indices": [1, 2],
        },
        "cases": records,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Wrote {out_path}")
    print()
    _print_md_table(records, with_anba=args.anba)

    return 0 if all(r.get("ok", False) for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
