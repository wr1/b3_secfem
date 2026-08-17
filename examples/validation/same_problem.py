#!/usr/bin/env python3
"""Same physical problem: ANBA input → secfem via anba_to_secfem_input.

Starts from what SONATA would send ANBA. Does not rewrite the card.

    micromamba run -n b3secfem python examples/validation/same_problem.py

Pass bar: K[Fz], K[Mx], K[My] on the ASYM rectangle within the quad→tri
split. At (fiber, plane)=(0,0), EA/A is E3. At (90,0), EA/A is E1.
"""

from __future__ import annotations

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
    rectangle,
    rel_per_dof,
)
from _anba_runner_host import anba_section_from_arrays  # noqa: E402
from _gx_vtu import write_gx_vtu  # noqa: E402
from b3_secfem import (  # noqa: E402
    IsotropicMaterial,
    SectionInput,
    anba_to_secfem_input,
    from_gxbeam_vtu,
    solve,
)

DOF = FORCE_ORDER_SECFEM
# Indices of the pass-bar terms in secfem order.
_AXIAL_BEND = (2, 3, 4)  # Fz, Mx, My


def _row(mat) -> np.ndarray:
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


def _solve_secfem(coords, quads, mat, beta, alpha, work: Path, tag: str) -> np.ndarray:
    n = quads.shape[0]
    vtu = work / f"{tag}.vtu"
    write_gx_vtu(vtu, coords, quads, np.tile(_row(mat), (n, 1)), np.zeros(n))
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
    return np.asarray(solve(inp).K, dtype=float)


def _solve_anba(coords, quads, mat, fiber, plane) -> tuple[np.ndarray, list]:
    n = quads.shape[0]
    out = anba_section_from_arrays(
        coords,
        quads,
        mat,
        fiber_orientation_deg=np.full(n, fiber),
        plane_orientation_deg=np.full(n, plane),
    )
    return np.asarray(out["K"], dtype=float), list(out["anba_order"])


def _identify_k_order(coords, quads, area: float, work: Path) -> dict:
    """Isotropic w≠h rectangle: EA, EIxx, EIyy are distinct."""
    steel = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0, name="iso")
    w, h = 0.04, 0.01
    # geometry is already that rectangle from caller
    K_s = _solve_secfem(coords, quads, steel, 0.0, 0.0, work, "iso_id")
    K_a, raw_order = _solve_anba(coords, quads, steel, 0.0, 0.0)
    EA = steel.E * area
    Ixx = w * h**3 / 12.0
    Iyy = h * w**3 / 12.0
    expect = {
        "Fz": EA,
        "Mx": steel.E * Ixx,
        "My": steel.E * Iyy,
    }
    raw_diag = np.diag(K_a)
    identified = {}
    for name, val in expect.items():
        idx = int(np.argmin(np.abs(raw_diag - val)))
        identified[name] = {"raw_index": idx, "raw_value": float(raw_diag[idx]), "expect": val}
    rec = {
        "secfem_diag": np.diag(K_s).tolist(),
        "anba_raw_diag": raw_diag.tolist(),
        "anba_raw_order_label": raw_order,
        "identified": identified,
        "secfem_vs_anba_raw_rel": rel_per_dof(K_s, K_a).tolist(),
    }
    print("K-order identification (iso rectangle 0.04 x 0.01)")
    print(f"  expect EA={EA:.4e}  EIxx={expect['Mx']:.4e}  EIyy={expect['My']:.4e}")
    print(f"  ANBA raw diag: {np.array2string(raw_diag, precision=3)}")
    print(f"  runner label:  {raw_order}")
    for name, blk in identified.items():
        print(f"  {name} closest raw index {blk['raw_index']}  value={blk['raw_value']:.4e}")
    # If raw already matches secfem (Fz at 2, Mx at 3, My at 4), permutation is I.
    perm_ok = (
        identified["Fz"]["raw_index"] == 2
        and identified["Mx"]["raw_index"] == 3
        and identified["My"]["raw_index"] == 4
    )
    rec["raw_already_secfem_order"] = perm_ok
    print(f"  raw already secfem order: {perm_ok}")
    return rec


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "meshes"
    work.mkdir(exist_ok=True)
    coords, quads, area = rectangle(0.04, 0.01, 8, 2)

    print("same_problem — ANBA input via anba_to_secfem_input, card unchanged")
    print(f"  rectangle 0.04 x 0.01 m, A={area:.4e} m²")
    print(f"  ASYM E1={ASYM.E1/1e9:.0f} E2={ASYM.E2/1e9:.0f} E3={ASYM.E3/1e9:.0f} GPa")
    print()

    ident = _identify_k_order(coords, quads, area, work)
    print()

    # (fiber, plane, expect_E or None, pin). Pins are the pass bar.
    # Mid-tilt is printed; it is a known residual, not a convention fail.
    cases = [
        (0.0, 0.0, "E3", True),
        (90.0, 0.0, "E1", True),
        (0.0, 90.0, "E3", True),
        (45.0, 0.0, None, False),
        (30.0, 45.0, None, False),
    ]
    report = {
        "area": area,
        "force_order": DOF,
        "identification": ident,
        "runs": [],
    }
    failed = False
    print(
        f"{'(fiber,plane)':<16} {'EA/A':>10} {'~E':>4} "
        f"{'Fz%':>8} {'Mx%':>8} {'My%':>8}  note"
    )
    for fiber, plane, expect_e, pin in cases:
        place = anba_to_secfem_input(ASYM, fiber, plane)
        K_s = _solve_secfem(
            coords,
            quads,
            place.material,
            place.beta_deg,
            place.alpha_deg,
            work,
            f"f{fiber:g}_p{plane:g}",
        )
        try:
            K_a, _ = _solve_anba(coords, quads, ASYM, fiber, plane)
        except Exception as exc:  # noqa: BLE001
            print(f"({fiber:g},{plane:g})  ANBA FAIL  {exc}"[:120])
            report["runs"].append({"fiber": fiber, "plane": plane, "error": str(exc)})
            failed = True
            continue
        rel = rel_per_dof(K_s, K_a)
        ea_a = float(K_s[2, 2]) / area
        close = closest_modulus(ea_a, ASYM)
        note = "" if pin else "residual"
        if pin:
            if expect_e is not None and close["closest"] != expect_e:
                note = f"EXPECTED {expect_e}"
                failed = True
            for i in _AXIAL_BEND:
                if rel[i, i] > 0.05:
                    note = (note + f" {DOF[i]}={100*rel[i,i]:.1f}%").strip()
                    failed = True
        print(
            f"({fiber:5.1f},{plane:5.1f})  {ea_a/1e9:10.3f} {close['closest']:>4} "
            f"{100*rel[2,2]:7.2f}% {100*rel[3,3]:7.2f}% {100*rel[4,4]:7.2f}%  {note}"
        )
        report["runs"].append(
            {
                "fiber": fiber,
                "plane": plane,
                "beta": place.beta_deg,
                "alpha": place.alpha_deg,
                "card_unchanged": place.material.E2 == ASYM.E2,
                "K_secfem": K_s.tolist(),
                "K_anba": K_a.tolist(),
                "rel_per_dof": rel.tolist(),
                "EA_over_A": ea_a,
                "closest_E": close,
                "expect_E": expect_e,
                "pin": pin,
            }
        )

    outp = OUT / "same_problem.json"
    outp.write_text(json.dumps(report, indent=2))
    print()
    print(f"wrote {outp}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
