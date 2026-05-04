"""Check whether the airfoil GJ disagreement is driven by degenerate
(zero-area) leading-edge and trailing-edge cells.

NACA 00xx half-thickness goes to zero at η=0 and η=1 in the closed-TE
form, which collapses one edge of the boundary cells to length 0. The
aspect-ratio rectangle sweep showed both engines agree to <0.5 % up to
AR=32, so the airfoil disagreement is **not** a slenderness or
formulation issue.

Test: re-run the iso airfoil with eta clipped to [eta_clip, 1 - eta_clip]
for several values of eta_clip. If the disagreement collapses with
eta_clip > 0, degenerate cells are the cause.

Run::

    /home/wr1/projects/b3/b3_secfem/.venv/bin/python examples/probe_airfoil_degenerate.py
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


def gen_airfoil_clipped(
    chord: float, thickness: float, n_chord: int, n_thick: int, eta_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """NACA 00xx solid with eta restricted to [eta_clip, 1 - eta_clip]."""
    eta = np.linspace(eta_clip, 1.0 - eta_clip, n_chord + 1)
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


def cell_aspect(coords: np.ndarray, quads: np.ndarray) -> tuple[float, float]:
    """Min and max edge length / area summary across cells."""
    p = coords[quads]
    edge_lens = []
    for i in range(4):
        d = p[:, (i + 1) % 4] - p[:, i]
        edge_lens.append(np.linalg.norm(d, axis=1))
    edge_lens = np.stack(edge_lens, axis=1)
    return float(edge_lens.min()), float(edge_lens.max())


def main() -> int:
    out = Path(__file__).parent / "cross_check_out"
    out.mkdir(parents=True, exist_ok=True)

    print("iso steel NACA 0024, chord=1, thickness=0.24, n_chord=64, n_thick=16.")
    print("Sweep eta_clip to remove degenerate cells at LE / TE.\n")

    print(
        f"  {'eta_clip':>9} | {'min_edge':>10} | {'max_edge':>10} | "
        f"{'GJ_gxbeam':>14} | {'GJ_b3_secfem':>14} | {'GJ rel_err':>10} | "
        f"{'EIy_gxbeam':>14} | {'EIy_b3_secfem':>14} | {'EIy rel_err':>10}"
    )
    print("-" * 145)

    for eta_clip in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]:
        coords, quads = gen_airfoil_clipped(
            chord=1.0, thickness=0.24, n_chord=64, n_thick=16, eta_clip=eta_clip,
        )
        n_cells = quads.shape[0]
        row = iso_row(STEEL)
        mat_props = np.tile(row, (n_cells, 1))
        theta = np.zeros(n_cells)

        vtu = out / f"af_clip_{eta_clip:g}.vtu"
        write_gx_vtu(vtu, coords, quads, mat_props, theta)

        K_gx, _S, _M, *_ = _gxbeam_section_from_arrays(coords, quads, mat_props, theta)
        info = from_gxbeam_vtu(vtu)
        inp = SectionInput(
            mesh_path=vtu,
            per_cell_material=[STEEL] * info["n_cells"],
            per_cell_alpha_deg=np.degrees(info["theta"]),
        )
        res = solve(inp)
        K_us = to_gxbeam_order(res.K)

        GJ_gx = float(K_gx[3, 3])
        GJ_us = float(K_us[3, 3])
        EIy_gx = float(K_gx[5, 5])     # M3 = My in gxbeam order
        EIy_us = float(K_us[5, 5])

        gj_re = (GJ_us - GJ_gx) / GJ_gx if GJ_gx else float("nan")
        eiy_re = (EIy_us - EIy_gx) / EIy_gx if EIy_gx else float("nan")

        emin, emax = cell_aspect(coords, quads)
        print(
            f"  {eta_clip:>9.4f} | {emin:>10.3e} | {emax:>10.3e} | "
            f"{GJ_gx:>14.4e} | {GJ_us:>14.4e} | {100*gj_re:>9.2f}% | "
            f"{EIy_gx:>14.4e} | {EIy_us:>14.4e} | {100*eiy_re:>9.2f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
