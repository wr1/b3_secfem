"""UD-on-ellipse cross-check probe.

The iso solid ellipse agrees between b3_secfem and gxbeam to <0.5 %.
The UD solid airfoil disagrees by 7–28 % at θ = 45° even after the
``alpha_deg`` fix. Question: is the residual due to the airfoil's mesh
shape (already a suspect on the iso GJ disagreement), or is it a
material-side rotation / warping-suppression convention issue?

Test: same UD carbon ladder on the ellipse where iso agrees. If b3_secfem
still disagrees with gxbeam by ~7-28 %, it's a material-side issue.
If the disagreement collapses to <1 %, it's mesh-specific.

Run::

    /home/wr1/projects/b3/b3_secfem/.venv/bin/python examples/probe_ud_ellipse.py
"""

from __future__ import annotations

import dolfinx  # noqa: F401  - load before juliacall
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cross_check_b3_gx import (  # noqa: E402
    UD_CARBON,
    _gxbeam_section_from_arrays,
    gen_ellipse_solid,
    ortho_row,
)
from b3_secfem import (  # noqa: E402
    SectionInput,
    from_gxbeam_vtu,
    solve,
    to_gxbeam_order,
)
from _gx_vtu import write_gx_vtu  # noqa: E402


def main() -> int:
    out = Path(__file__).parent / "cross_check_out"
    out.mkdir(parents=True, exist_ok=True)

    coords, quads = gen_ellipse_solid(a=0.06, b=0.02, n_circ=64, n_rad=12)
    n_cells = quads.shape[0]
    row = ortho_row(UD_CARBON)
    mat_props = np.tile(row, (n_cells, 1))

    angles_deg = [0.0, 30.0, 45.0, 60.0, 90.0]

    print(f"Solid ellipse a=0.06, b=0.02, n_cells={n_cells}, UD carbon")
    print(f"  E1=135 GPa, E2=E3=10 GPa")
    print()

    cols = ["theta", "EA", "GAx", "GAy", "GJ", "EIx", "EIy"]
    print(
        f"  {'angle':>6} | {'engine':<10} | "
        + " | ".join(f"{c:>14}" for c in cols[1:])
    )
    print("-" * 116)

    for ang_deg in angles_deg:
        # gxbeam — pass theta as the per-element angle
        theta_arr = np.full(n_cells, np.radians(ang_deg))
        vtu_gx = out / f"ell_ud_th{ang_deg:g}.vtu"
        write_gx_vtu(vtu_gx, coords, quads, mat_props, theta_arr)
        K_gx, *_ = _gxbeam_section_from_arrays(coords, quads, mat_props, theta_arr)

        # b3_secfem — pass alpha_deg = ang_deg (theta in the VTU is what
        # from_gxbeam_vtu reads back as info["theta"], i.e. radians)
        theta_arr_zero = np.zeros(n_cells)
        vtu_b3 = out / f"ell_ud_a{ang_deg:g}.vtu"
        write_gx_vtu(vtu_b3, coords, quads, mat_props, theta_arr_zero)
        info = from_gxbeam_vtu(vtu_b3)
        inp = SectionInput(
            mesh_path=vtu_b3,
            per_cell_material=[UD_CARBON] * info["n_cells"],
            per_cell_alpha_deg=np.full(info["n_cells"], ang_deg),
        )
        res = solve(inp)
        K_us = to_gxbeam_order(res.K)

        # Print rows: gxbeam, b3_secfem, rel-err
        # gxbeam-order indexing: [0]=Fz, [1]=Fx, [2]=Fy, [3]=Mz, [4]=Mx, [5]=My
        # so EA=K[0,0], GAx=K[1,1], GAy=K[2,2], GJ=K[3,3], EIx=K[4,4], EIy=K[5,5]
        gx_vals = [K_gx[i, i] for i in range(6)]
        us_vals = [K_us[i, i] for i in range(6)]
        rel = [abs(u - g) / max(abs(g), 1e-30) for u, g in zip(us_vals, gx_vals)]
        print(
            f"  {ang_deg:>6.1f} | gxbeam     | "
            + " | ".join(f"{v:>14.4e}" for v in gx_vals)
        )
        print(
            f"  {'':>6} | b3_secfem  | "
            + " | ".join(f"{v:>14.4e}" for v in us_vals)
        )
        print(
            f"  {'':>6} | rel diff   | "
            + " | ".join(f"{100*r:>13.2f}%" for r in rel)
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
