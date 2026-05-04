"""Probe what gxbeam's per-element ``theta`` rotates relative to b3_secfem.

b3_secfem has two angles:
  * ``beta_deg``  — fibre rotation about the beam axis z. β=0 → fibre along z.
  * ``alpha_deg`` — fibre tilt out of beam axis toward the section plane.
                    α=0 → fibre along z. α=90 → fibre fully in the section.

gxbeam_section has one angle ``theta`` per element. The cross-check showed
that at θ=π/4 gxbeam's K[Fz,Fz] drops 8× — incompatible with rotation about
z (which leaves K[Fz,Fz] invariant). Hypothesis: gxbeam.theta == b3_secfem.alpha
(out-of-plane tilt), not beta.

Run with::

    /home/wr1/projects/b3/b3_secfem/.venv/bin/python examples/probe_theta_convention.py
"""

from __future__ import annotations

import dolfinx  # noqa: F401  - load before juliacall (libcurl conflict)
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cross_check_b3_gx import (  # noqa: E402
    UD_CARBON,
    _gxbeam_section_from_arrays,
    ortho_row,
)
from b3_secfem import (  # noqa: E402
    OrthotropicMaterial,
    SectionInput,
    from_gxbeam_vtu,
    solve,
    to_gxbeam_order,
)
from _gx_vtu import write_gx_vtu  # noqa: E402


def make_strip(width: float = 0.04, height: float = 0.01, n_w: int = 4, n_h: int = 2):
    coords = []
    for j in range(n_h + 1):
        y = -height / 2 + j * height / n_h
        for i in range(n_w + 1):
            x = -width / 2 + i * width / n_w
            coords.append([x, y])
    quads = []
    for j in range(n_h):
        for i in range(n_w):
            n00 = j * (n_w + 1) + i
            n10 = j * (n_w + 1) + i + 1
            n11 = (j + 1) * (n_w + 1) + i + 1
            n01 = (j + 1) * (n_w + 1) + i
            quads.append([n00, n10, n11, n01])
    return np.asarray(coords, dtype=np.float64), np.asarray(quads, dtype=np.int64)


def main() -> int:
    out = Path(__file__).parent / "cross_check_out"
    out.mkdir(parents=True, exist_ok=True)

    coords, quads = make_strip()
    n_cells = quads.shape[0]
    row = ortho_row(UD_CARBON)
    mat_props = np.tile(row, (n_cells, 1))

    angles_deg = [0.0, 30.0, 45.0, 60.0, 90.0]

    print(f"strip {coords[:,0].max()-coords[:,0].min():.3f} x "
          f"{coords[:,1].max()-coords[:,1].min():.3f} m, n_cells={n_cells}, "
          f"UD carbon E1=135 GPa, E2=E3=10 GPa")
    print()

    print("---- gxbeam: theta sweep ----")
    print(f"  {'theta':>6} | {'K[Fz,Fz]':>14} | {'K[Mx,Mx]':>14} | {'K[Mz,Mz]':>14}")
    gx_results = {}
    for ang_deg in angles_deg:
        theta = np.full(n_cells, np.radians(ang_deg))
        vtu = out / f"strip_th{ang_deg:g}.vtu"
        write_gx_vtu(vtu, coords, quads, mat_props, theta)
        K_gx, _S, _M, *_ = _gxbeam_section_from_arrays(coords, quads, mat_props, theta)
        gx_results[ang_deg] = K_gx
        print(f"  {ang_deg:>6.1f} | {K_gx[0,0]:>14.4e} | {K_gx[4,4]:>14.4e} | {K_gx[3,3]:>14.4e}")
    print()

    print("---- b3_secfem: alpha sweep (beta=0) ----")
    print(f"  {'alpha':>6} | {'K[Fz,Fz]':>14} | {'K[Mx,Mx]':>14} | {'K[Mz,Mz]':>14}")
    b3_alpha = {}
    for ang_deg in angles_deg:
        theta_zero = np.zeros(n_cells)
        vtu = out / f"strip_a{ang_deg:g}.vtu"
        write_gx_vtu(vtu, coords, quads, mat_props, theta_zero)
        info = from_gxbeam_vtu(vtu)
        inp = SectionInput(
            mesh_path=vtu,
            per_cell_material=[UD_CARBON] * info["n_cells"],
            per_cell_beta_deg=np.zeros(info["n_cells"]),
            per_cell_alpha_deg=np.full(info["n_cells"], ang_deg),
        )
        res = solve(inp)
        K = to_gxbeam_order(res.K)
        b3_alpha[ang_deg] = K
        print(f"  {ang_deg:>6.1f} | {K[0,0]:>14.4e} | {K[4,4]:>14.4e} | {K[3,3]:>14.4e}")
    print()

    print("---- b3_secfem: beta sweep (alpha=0) ----")
    print(f"  {'beta':>6} | {'K[Fz,Fz]':>14} | {'K[Mx,Mx]':>14} | {'K[Mz,Mz]':>14}")
    b3_beta = {}
    for ang_deg in angles_deg:
        theta_zero = np.zeros(n_cells)
        vtu = out / f"strip_b{ang_deg:g}.vtu"
        write_gx_vtu(vtu, coords, quads, mat_props, theta_zero)
        info = from_gxbeam_vtu(vtu)
        inp = SectionInput(
            mesh_path=vtu,
            per_cell_material=[UD_CARBON] * info["n_cells"],
            per_cell_beta_deg=np.full(info["n_cells"], ang_deg),
            per_cell_alpha_deg=np.zeros(info["n_cells"]),
        )
        res = solve(inp)
        K = to_gxbeam_order(res.K)
        b3_beta[ang_deg] = K
        print(f"  {ang_deg:>6.1f} | {K[0,0]:>14.4e} | {K[4,4]:>14.4e} | {K[3,3]:>14.4e}")
    print()

    print("---- agreement check: b3_secfem alpha vs gxbeam theta ----")
    print(f"  {'angle':>6} | {'K[Fz,Fz] rel err':>18} | {'K[Mx,Mx] rel err':>18} | {'K[Mz,Mz] rel err':>18}")
    for ang_deg in angles_deg:
        K_us = b3_alpha[ang_deg]
        K_gx = gx_results[ang_deg]

        def relerr(a, b):
            return abs(a - b) / max(abs(b), 1e-30)

        e0 = relerr(K_us[0, 0], K_gx[0, 0])
        e1 = relerr(K_us[4, 4], K_gx[4, 4])
        e2 = relerr(K_us[3, 3], K_gx[3, 3])
        print(f"  {ang_deg:>6.1f} | {100*e0:>17.4f}% | {100*e1:>17.4f}% | {100*e2:>17.4f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
