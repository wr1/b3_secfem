#!/usr/bin/env python3
"""Tight loop: which of E1/E2/E3 sits on the beam axis, and where 2/3 point.

Runs secfem fenicsx and mfem on the same rectangle + asymmetric material.
ANBA is optional (Docker ``anba4:latest``). Mapping variants are recorded,
not preferred.

    micromamba run -n b3secfem python examples/validation/two3_probe.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# examples/ on path for _gx_vtu / _anba_runner_host
_EXAMPLES = Path(__file__).resolve().parent.parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from validation._common import (  # noqa: E402
    ASYM,
    FORCE_ORDER_SECFEM,
    OUT,
    closest_modulus,
    default_mappings,
    describe_inplane,
    mat_to_dict,
    principal_axes,
    rectangle,
)

from _gx_vtu import write_gx_vtu  # noqa: E402
from b3_secfem import SectionInput, from_gxbeam_vtu, solve  # noqa: E402
from b3_secfem.rotation3d import _bond_T, _rotation_matrix_3x3  # noqa: E402


def _ortho_row(mat) -> np.ndarray:
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


def _solve_secfem(coords, quads, mapping, backend: str, work: Path) -> dict:
    n = quads.shape[0]
    vtu = work / f"{mapping.name}_{backend}.vtu"
    write_gx_vtu(
        vtu,
        coords,
        quads,
        np.tile(_ortho_row(mapping.material), (n, 1)),
        np.zeros(n),
    )
    info = from_gxbeam_vtu(vtu)
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=[mapping.material] * info["n_cells"],
        per_cell_beta_deg=np.full(info["n_cells"], mapping.beta_deg),
        per_cell_alpha_deg=np.full(info["n_cells"], mapping.alpha_deg),
        backend=backend,  # type: ignore[arg-type]
        linear_solver="lu",
        degree=2,
    )
    res = solve(inp)
    K = np.asarray(res.K, dtype=float)
    area = float(np.asarray(res.area)) if hasattr(res, "area") else None
    rec: dict = {
        "backend": backend,
        "mapping": mapping.name,
        "K": K.tolist(),
        "K_order": FORCE_ORDER_SECFEM,
        "EA": float(K[2, 2]),
        "K_diag": np.diag(K).tolist(),
    }
    if area is not None:
        rec["area"] = area
        rec["EA_over_A"] = rec["EA"] / area
        rec["closest_E"] = closest_modulus(rec["EA_over_A"], mapping.material)
    try:
        from b3_secfem import recover_unit_load_strains

        fields = recover_unit_load_strains(res)
        w = np.asarray(fields.cell_areas, dtype=float)
        w = w / w.sum()
        # Fz = index 2. Global Voigt (11,22,33,23,13,12) = (xx,yy,zz,...).
        sig_g = np.einsum("c,cv->v", w, fields.sigma[2])
        R = _rotation_matrix_3x3(mapping.beta_deg, mapping.alpha_deg)
        sig_l = np.linalg.inv(_bond_T(R)) @ sig_g
        rec["sigma_Fz_global_mean"] = sig_g.tolist()
        rec["sigma_Fz_local_mean"] = sig_l.tolist()
        rec["sigma_local_order"] = ["s11", "s22", "s33", "s23", "s13", "s12"]
    except Exception as exc:  # noqa: BLE001 — recovery is diagnostic, not required
        rec["recovery_error"] = f"{type(exc).__name__}: {exc}"
    return rec


def _try_anba(coords, quads, mapping) -> dict | None:
    if mapping.anba_fiber_deg is None:
        return None
    try:
        from _anba_runner_host import anba_section_from_arrays
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"import: {exc}"}
    n = quads.shape[0]
    try:
        out = anba_section_from_arrays(
            coords,
            quads,
            mapping.material,
            fiber_orientation_deg=np.full(n, mapping.anba_fiber_deg),
            plane_orientation_deg=np.full(n, mapping.anba_plane_deg or 0.0),
        )
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": str(exc)[:500]}
    K = np.asarray(out["K"], dtype=float)
    return {
        "backend": "anba",
        "mapping": mapping.name,
        "anba_fiber_deg": mapping.anba_fiber_deg,
        "anba_plane_deg": mapping.anba_plane_deg,
        "anba_order": out.get("anba_order"),
        "K": K.tolist(),
        "EA": float(K[2, 2]),
        "K_diag": np.diag(K).tolist(),
        "skipped": False,
    }


def _run_worker(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "meshes"
    work.mkdir(exist_ok=True)
    coords, quads, area = rectangle(0.04, 0.01, 8, 2)
    mappings = {m.name: m for m in default_mappings(ASYM)}
    mapping = mappings[args.mapping]
    rec = _solve_secfem(coords, quads, mapping, args.backend, work)
    rec["area_geom"] = area
    rec["EA_over_A"] = rec["EA"] / area
    rec["closest_E"] = closest_modulus(rec["EA_over_A"], mapping.material)
    rec["axes"] = principal_axes(mapping.beta_deg, mapping.alpha_deg)
    rec["axes_roles_horizontal_flange"] = {
        k: describe_inplane(v) for k, v in rec["axes"].items() if k.startswith("axis")
    }
    rec["material"] = mat_to_dict(mapping.material)
    rec["user_material"] = mat_to_dict(ASYM)
    rec["note"] = mapping.note
    args.out.write_text(json.dumps(rec, indent=2))
    return 0


def _fmt_vec(v) -> str:
    a = np.asarray(v, dtype=float)
    return f"({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f})"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("fenicsx", "mfem"))
    p.add_argument("--mapping")
    p.add_argument("--out", type=Path)
    p.add_argument("--skip-anba", action="store_true")
    p.add_argument("--skip-mfem", action="store_true")
    args = p.parse_args()

    if args.backend:
        if not args.mapping or not args.out:
            p.error("--backend requires --mapping and --out")
        return _run_worker(args)

    OUT.mkdir(parents=True, exist_ok=True)
    coords, quads, area = rectangle(0.04, 0.01, 8, 2)
    mappings = default_mappings(ASYM)
    backends = ["fenicsx"] if args.skip_mfem else ["fenicsx", "mfem"]

    print("two3 probe")
    print(f"  rectangle 0.04 x 0.01 m, A={area:.4e} m², {len(quads)} quads")
    print(
        f"  user material {ASYM.name}: E1={ASYM.E1 / 1e9:.1f}  E2={ASYM.E2 / 1e9:.1f}  "
        f"E3={ASYM.E3 / 1e9:.1f} GPa"
    )
    print(f"  force order: {FORCE_ORDER_SECFEM}")
    print()

    print("analytic principal axes (R = Rz(β) Ry(α); (0,0)=I; axis 1 = fibre)")
    print(f"  {'(β,α)':>10}  {'axis1 fibre':<28}  {'axis2':<28}  axis3")
    for beta, alpha in ((0.0, 0.0), (90.0, 0.0), (0.0, 90.0)):
        ax = principal_axes(beta, alpha)
        print(
            f"  {beta:4.0f},{alpha:<4.0f}  {_fmt_vec(ax['axis1_fibre']):<28}  "
            f"{_fmt_vec(ax['axis2']):<28}  {_fmt_vec(ax['axis3'])}"
        )
        print(f"           2: {describe_inplane(ax['axis2'])}")
        print(f"           3: {describe_inplane(ax['axis3'])}")
    print()

    report: dict = {
        "area": area,
        "user_material": mat_to_dict(ASYM),
        "force_order": FORCE_ORDER_SECFEM,
        "axes_table": {
            f"b{int(b)}_a{int(a)}": principal_axes(b, a)
            for b, a in ((0.0, 0.0), (90.0, 0.0), (0.0, 90.0))
        },
        "runs": [],
    }

    py = [sys.executable, str(Path(__file__).resolve())]
    print(
        f"{'mapping':<22} {'be':<8} {'EA/A GPa':>10} {'~E':>4} "
        f"{'%E1':>7} {'%E2':>7} {'%E3':>7}  note"
    )
    for mapping in mappings:
        for backend in backends:
            outp = OUT / f"{mapping.name}_{backend}.json"
            cmd = py + [
                "--backend",
                backend,
                "--mapping",
                mapping.name,
                "--out",
                str(outp),
            ]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0 or not outp.exists():
                print(f"{mapping.name:<22} {backend:<8} FAIL")
                print(proc.stderr[-800:] if proc.stderr else proc.stdout[-400:])
                report["runs"].append(
                    {
                        "mapping": mapping.name,
                        "backend": backend,
                        "error": (proc.stderr or proc.stdout)[-1500:],
                    }
                )
                continue
            rec = json.loads(outp.read_text())
            c = rec["closest_E"]
            print(
                f"{mapping.name:<22} {backend:<8} {rec['EA_over_A'] / 1e9:10.3f} "
                f"{c['closest']:>4} {100 * c['rel_vs_E1']:6.2f}% "
                f"{100 * c['rel_vs_E2']:6.2f}% {100 * c['rel_vs_E3']:6.2f}%  "
                f"{mapping.note[:48]}"
            )
            report["runs"].append(rec)

        if not args.skip_anba:
            an = _try_anba(coords, quads, mapping)
            if an is None:
                continue
            an["mapping"] = mapping.name
            if an.get("skipped"):
                print(
                    f"{mapping.name:<22} {'anba':<8} skip  {an.get('reason', '')[:70]}"
                )
            else:
                c = closest_modulus(an["EA"] / area, mapping.material)
                an["area_geom"] = area
                an["EA_over_A"] = an["EA"] / area
                an["closest_E"] = c
                print(
                    f"{mapping.name:<22} {'anba':<8} {an['EA_over_A'] / 1e9:10.3f} "
                    f"{c['closest']:>4} {100 * c['rel_vs_E1']:6.2f}% "
                    f"{100 * c['rel_vs_E2']:6.2f}% {100 * c['rel_vs_E3']:6.2f}%"
                )
            report["runs"].append(an)

    out_json = OUT / "two3_probe.json"
    out_json.write_text(json.dumps(report, indent=2))
    print()
    print(f"wrote {out_json}")
    try:
        from validation._viz import plot_two3

        for p in plot_two3(report):
            print(f"wrote {p}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot skip: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
