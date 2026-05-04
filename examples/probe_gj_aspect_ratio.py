"""Aspect-ratio sweep on iso steel rectangles to characterise the airfoil
GJ disagreement seen in the cross-check.

Compares K[Mz,Mz] (GJ) from gxbeam_section, b3_secfem, and the analytical
Saint-Venant solution ``J = β(AR) · h · b³``  for a rectangle ``b × h``
with b > h, using the standard tabulated β coefficients.

Run::

    /home/wr1/projects/b3/b3_secfem/.venv/bin/python examples/probe_gj_aspect_ratio.py
"""

from __future__ import annotations

import dolfinx  # noqa: F401  - load before juliacall
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cross_check_b3_gx import (  # noqa: E402
    STEEL,
    _gxbeam_section_from_arrays,
    iso_row,
)
from b3_secfem import (  # noqa: E402
    SectionInput,
    from_gxbeam_vtu,
    solve,
    to_gxbeam_order,
)
from _gx_vtu import write_gx_vtu  # noqa: E402


# Saint-Venant torsion constant for a rectangle b × h (b ≥ h):
#     J = β · b · h^3   (from Timoshenko & Goodier, Theory of Elasticity, Table 5)
# Tabulated for the limiting solution at large b/h, β → 1/3.
SV_BETA = {
    1.0:  0.141,    # square
    1.5:  0.196,
    2.0:  0.229,
    2.5:  0.249,
    3.0:  0.263,
    4.0:  0.281,
    5.0:  0.291,
    6.0:  0.298,
    8.0:  0.307,
    10.0: 0.312,
    20.0: 0.323,
    50.0: 0.331,
    100.0: 0.333,   # → 1/3
}


def gen_rect(b: float, h: float, n_b: int, n_h: int):
    coords = []
    for j in range(n_h + 1):
        y = -h / 2 + j * h / n_h
        for i in range(n_b + 1):
            x = -b / 2 + i * b / n_b
            coords.append([x, y])
    quads = []
    for j in range(n_h):
        for i in range(n_b):
            n00 = j * (n_b + 1) + i
            n10 = j * (n_b + 1) + i + 1
            n11 = (j + 1) * (n_b + 1) + i + 1
            n01 = (j + 1) * (n_b + 1) + i
            quads.append([n00, n10, n11, n01])
    return np.asarray(coords, dtype=np.float64), np.asarray(quads, dtype=np.int64)


def main() -> int:
    out = Path(__file__).parent / "cross_check_out"
    out.mkdir(parents=True, exist_ok=True)

    h = 0.04                                      # fixed height
    G = STEEL.E / (2 * (1 + STEEL.nu))            # iso shear modulus

    aspect_ratios = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    # Shrink h further at high AR if needed; let's pick a target n_cells ~ 256
    # by giving cells a roughly square shape.
    n_h = 16

    print(f"iso steel rectangle (E={STEEL.E:.0f} Pa, nu={STEEL.nu}, "
          f"G={G:.3e} Pa), h={h} m fixed.")
    print(f"Saint-Venant analytical: J = β(AR) · b · h³;  GJ = G · J\n")

    print(
        f"  {'AR':>5} | {'b/h':>6} | {'n_cells':>7} | "
        f"{'GJ_SV (analytical)':>20} | "
        f"{'GJ_gxbeam':>14} | {'rel SV vs gx':>12} | "
        f"{'GJ_b3_secfem':>14} | {'rel SV vs b3':>12} | "
        f"{'rel gx vs b3':>12}"
    )
    print("-" * 165)

    for ar in aspect_ratios:
        b = ar * h
        n_b = int(round(n_h * ar))
        n_b = max(n_b, 4)
        coords, quads = gen_rect(b, h, n_b, n_h)
        n_cells = quads.shape[0]
        row = iso_row(STEEL)
        mat_props = np.tile(row, (n_cells, 1))
        theta = np.zeros(n_cells)

        vtu = out / f"rect_ar{ar:g}.vtu"
        write_gx_vtu(vtu, coords, quads, mat_props, theta)

        # gxbeam reference
        K_gx, _S, _M, *_ = _gxbeam_section_from_arrays(coords, quads, mat_props, theta)
        GJ_gx = float(K_gx[3, 3])    # M1 = Mz in gxbeam order

        # b3_secfem
        info = from_gxbeam_vtu(vtu)
        inp = SectionInput(
            mesh_path=vtu,
            per_cell_material=[STEEL] * info["n_cells"],
            per_cell_alpha_deg=np.degrees(info["theta"]),
        )
        res = solve(inp)
        K_us = to_gxbeam_order(res.K)
        GJ_us = float(K_us[3, 3])

        # Analytical
        beta_ar = np.interp(ar, list(SV_BETA.keys()), list(SV_BETA.values()))
        J_sv = beta_ar * b * h ** 3
        GJ_sv = G * J_sv

        rel_sv_gx = (GJ_gx - GJ_sv) / GJ_sv
        rel_sv_b3 = (GJ_us - GJ_sv) / GJ_sv
        rel_gx_b3 = (GJ_us - GJ_gx) / GJ_gx

        print(
            f"  {ar:>5.1f} | {ar:>6.2f} | {n_cells:>7d} | "
            f"{GJ_sv:>20.4e} | "
            f"{GJ_gx:>14.4e} | {100*rel_sv_gx:>11.2f}% | "
            f"{GJ_us:>14.4e} | {100*rel_sv_b3:>11.2f}% | "
            f"{100*rel_gx_b3:>11.2f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
