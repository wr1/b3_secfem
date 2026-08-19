#!/usr/bin/env python3
"""Compare a SONATA dump (or a synthetic SONATA-like ply) on secfem.

Does **not** choose a mapping. For each dumped section it runs the 1:1
card at α=0 and the 2↔3 swap, fenicsx and optionally mfem, and reports
whether K moved.

Synthetic mode (no SONATA env) reproduces the suite claim on a rectangle:

    micromamba run -n b3secfem python examples/validation/sonata_compare.py --synthetic

Dump mode (after sonata_dump.py)::

    micromamba run -n b3secfem python examples/validation/sonata_compare.py out/sonata/beam0.json
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
    FORCE_ORDER_SECFEM,
    GLASS_BIAX,
    GLASS_UNI,
    OUT,
    SMITH_CHOPRA,
    anba_angles,
    closest_modulus,
    rectangle,
    rel_per_dof,
    swap23,
)
from _gx_vtu import write_gx_vtu  # noqa: E402
from b3_secfem import (  # noqa: E402
    IsotropicMaterial,
    OrthotropicMaterial,
    SectionInput,
    from_gxbeam_vtu,
    solve,
)


def _row(mat) -> np.ndarray:
    if isinstance(mat, IsotropicMaterial):
        G = mat.E / (2 * (1 + mat.nu))
        return np.array([mat.E, mat.E, mat.E, G, G, G, mat.nu, mat.nu, mat.nu, mat.rho])
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
        ],
        dtype=np.float64,
    )


def _solve(coords, quads, mat, *, alpha, beta, backend, tag, work: Path) -> dict:
    n = quads.shape[0]
    vtu = work / f"{tag}.vtu"
    write_gx_vtu(vtu, coords, quads, np.tile(_row(mat), (n, 1)), np.zeros(n))
    info = from_gxbeam_vtu(vtu)
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=[mat] * info["n_cells"],
        per_cell_beta_deg=np.full(info["n_cells"], beta),
        per_cell_alpha_deg=np.full(info["n_cells"], alpha),
        backend=backend,  # type: ignore[arg-type]
        linear_solver="lu",
    )
    res = solve(inp)
    K = np.asarray(res.K, dtype=float)
    return {"K": K.tolist(), "K_diag": np.diag(K).tolist(), "EA": float(K[2, 2])}


def _relmax(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.maximum(np.abs(a), 1e-30)
    return float(np.max(np.abs(a - b) / den))


def _card_from_sonata(card: dict):
    if card["orth"] == 0:
        return IsotropicMaterial(
            E=card["E"], nu=card["nu"], rho=card["rho"], name=card.get("name")
        )
    e, g, nu = card["E"], card["G"], card["nu"]
    return OrthotropicMaterial(
        E1=e[0],
        E2=e[1],
        E3=e[2],
        G12=g[0],
        G13=g[1],
        G23=g[2],
        nu12=nu[0],
        nu13=nu[1],
        nu23=nu[2],
        rho=card["rho"],
        name=card.get("name"),
    )


def _compare_pair(name: str, coords, quads, mat, backends, work: Path) -> dict:
    e2eq = bool(np.isclose(getattr(mat, "E2", 0), getattr(mat, "E3", 0)))
    rec = {
        "name": name,
        "e2_eq_e3": e2eq,
        "material": getattr(mat, "name", None),
        "backends": {},
    }
    if isinstance(mat, OrthotropicMaterial):
        rec["E"] = [mat.E1, mat.E2, mat.E3]
    for be in backends:
        one = _solve(
            coords,
            quads,
            mat,
            alpha=0.0,
            beta=0.0,
            backend=be,
            tag=f"{name}_{be}_1to1",
            work=work,
        )
        sw = _solve(
            coords,
            quads,
            swap23(mat) if isinstance(mat, OrthotropicMaterial) else mat,
            alpha=0.0,
            beta=0.0,
            backend=be,
            tag=f"{name}_{be}_swap23",
            work=work,
        )
        rec["backends"][be] = {
            "1to1": one,
            "swap23": sw,
            "Kdiag_relmax_1to1_vs_swap23": _relmax(one["K_diag"], sw["K_diag"]),
            "Kdiag_rel_per_dof": rel_per_dof(one["K_diag"], sw["K_diag"]).tolist(),
        }
        if isinstance(mat, OrthotropicMaterial):
            rec["backends"][be]["1to1"]["closest_E"] = closest_modulus(
                one["EA"] / max(1e-30, _shoelace_area(coords, quads)),
                mat,
            )
    rec["anba"] = _try_anba(coords, quads, mat)
    return rec


def _try_anba(coords, quads, mat) -> dict:
    try:
        from _anba_runner_host import anba_section_from_arrays
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"import: {exc}"}
    n = quads.shape[0]
    fiber, plane = anba_angles(0.0, 0.0)
    try:
        out = anba_section_from_arrays(
            coords,
            quads,
            mat,
            fiber_orientation_deg=np.full(n, fiber),
            plane_orientation_deg=np.full(n, plane),
        )
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": str(exc)[:500]}
    K = np.asarray(out["K"], dtype=float)
    return {
        "skipped": False,
        "K": K.tolist(),
        "K_diag": np.diag(K).tolist(),
        "EA": float(K[2, 2]),
        "anba_fiber_deg": fiber,
        "anba_plane_deg": plane,
    }


def _shoelace_area(coords, quads) -> float:
    p = coords[quads]

    # split each quad into two tris
    def tri(a, b, c):
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        )

    return float(
        tri(p[:, 0], p[:, 1], p[:, 2]).sum() + tri(p[:, 0], p[:, 2], p[:, 3]).sum()
    )


def synthetic(backends: list[str]) -> dict:
    work = OUT / "meshes"
    work.mkdir(parents=True, exist_ok=True)
    coords, quads, _ = rectangle(0.04, 0.01, 8, 2)
    rows = []
    for mat in (SMITH_CHOPRA, GLASS_UNI, GLASS_BIAX):
        rec = _compare_pair(mat.name, coords, quads, mat, backends, work)
        rows.append(rec)
        for be, blk in rec["backends"].items():
            moved = blk["Kdiag_relmax_1to1_vs_swap23"]
            print(
                f"{mat.name:<16} e2=e3={rec['e2_eq_e3']!s:<5} {be:<8} "
                f"1to1-vs-swap23 Kdiag relmax={moved:.3e}"
            )
        an = rec.get("anba") or {}
        if an.get("skipped"):
            print(f"{mat.name:<16} anba     skip  {str(an.get('reason', ''))[:70]}")
        elif "K_diag" in an:
            rel_an = _relmax(rec["backends"]["fenicsx"]["1to1"]["K_diag"], an["K_diag"])
            rec["anba"]["Kdiag_relmax_vs_secfem_1to1"] = rel_an
            print(f"{mat.name:<16} anba     vs secfem 1:1 Kdiag relmax={rel_an:.3e}")
    return {"mode": "synthetic", "force_order": FORCE_ORDER_SECFEM, "cases": rows}


def from_dump(path: Path, backends: list[str], *, max_cells: int) -> dict:
    work = OUT / "meshes"
    work.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text())
    rows = []
    for i, sec in enumerate(data["sections"]):
        coords = np.asarray(sec["node_xy"], dtype=float)
        cells = sec["cells"]
        # secfem path is quad-oriented; skip non-quads (SONATA often tris).
        quads = [c for c in cells if len(c) == 4]
        tris = [c for c in cells if len(c) == 3]
        if quads:
            conn = np.asarray(quads, dtype=np.int64)
        elif tris:
            # degenerate quads: repeat last node (fenicsx/mfem want quads or
            # mixed — from_gxbeam_vtu is quad-only). Split is wrong; skip.
            print(
                f"  section {i}: triangle-only mesh ({len(tris)} tris) — "
                f"secfem gxbeam VTU path is quad-only; skip"
            )
            continue
        else:
            continue
        if conn.shape[0] > max_cells:
            print(
                f"  section {i}: {conn.shape[0]} quads > --max-cells {max_cells}; "
                f"skip (raise the cap to run)"
            )
            continue
        # Homogeneous first material only for the cheap dump compare; mixed
        # regions need per-cell materials (follow-up).
        mats = sec["materials"]
        cards = list(mats.values())
        ortho = [c for c in cards if c.get("orth") == 1]
        if not ortho:
            print(f"  section {i}: no orthotropic card")
            continue
        # Prefer a card with E2 ≠ E3 if present.
        card = next((c for c in ortho if not c.get("e2_eq_e3", True)), ortho[0])
        mat = _card_from_sonata(card)
        rec = _compare_pair(
            f"{data['case']}_s{i}_{card.get('name')}", coords, conn, mat, backends, work
        )
        rec["grid"] = sec.get("grid")
        rec["n_cells_used"] = int(conn.shape[0])
        rec["homogeneous_approximation"] = True
        rec["note"] = (
            "uses one orthotropic card on the whole mesh — isolates 2/3, "
            "not a full multi-material SONATA replay"
        )
        if "ANBA_TS" in sec:
            rec["ANBA_TS_diag"] = np.diag(np.asarray(sec["ANBA_TS"])).tolist()
        rows.append(rec)
        for be, blk in rec["backends"].items():
            print(
                f"{rec['name']:<28} e2=e3={rec['e2_eq_e3']!s:<5} {be:<8} "
                f"1to1-vs-swap23={blk['Kdiag_relmax_1to1_vs_swap23']:.3e}"
            )
    return {"mode": "dump", "source": str(path), "cases": rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dumps", nargs="*", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument(
        "--with-mfem",
        action="store_true",
        help="also run mfem in-process (unsafe with fenicsx teardown)",
    )
    p.add_argument("--max-cells", type=int, default=400)
    args = p.parse_args()
    if not args.synthetic and not args.dumps:
        args.synthetic = True

    backends = ["fenicsx", "mfem"] if args.with_mfem else ["fenicsx"]
    OUT.mkdir(parents=True, exist_ok=True)

    reports = []
    if args.synthetic:
        print("synthetic SONATA-like cards on a rectangle (α=0 1:1 vs swap23)")
        reports.append(synthetic(backends))
    for dump in args.dumps:
        print(f"dump {dump}")
        reports.append(from_dump(dump, backends, max_cells=args.max_cells))

    dest = OUT / "sonata_compare.json"
    dest.write_text(json.dumps(reports, indent=2))
    print(f"wrote {dest}")
    try:
        from validation._viz import plot_sonata

        p = plot_sonata(reports)
        if p:
            print(f"wrote {p}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot skip: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
