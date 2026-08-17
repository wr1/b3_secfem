"""3-way comparison on a NACA airfoil: b3_secfem vs gxbeam_section vs ANBA4.

The user-requested focused probe: skip iso cases (they all agree), drive
the airfoil through all three solvers with UD at 0° and 45° fibre tilt,
print results side by side.

NACA 0024 leading and trailing edges have zero half-thickness, which
makes the boundary cells collapse. We use a small ``eta_clip`` (start at
1 % chord, end at 99 %) so neither engine has to deal with zero-area
elements — the airfoil shape stays the same minus two thin slivers.

Run::

    /home/wr1/projects/b3/b3_secfem/.venv/bin/python examples/probe_airfoil_3way.py
"""

from __future__ import annotations

import dolfinx  # noqa: F401  - load before juliacall
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cross_check_b3_gx import (  # noqa: E402
    STEEL,
    UD_CARBON,
    _gxbeam_section_from_arrays,
    iso_row,
    ortho_row,
)
from _anba_runner_host import anba_section_from_arrays  # noqa: E402
from _gx_vtu import write_gx_vtu  # noqa: E402
from b3_secfem import (  # noqa: E402
    SectionInput,
    from_gxbeam_vtu,
    solve,
    to_gxbeam_order,
)


def gen_airfoil(
    chord: float = 1.0,
    thickness: float = 0.24,
    n_chord: int = 32,
    n_thick: int = 8,
    eta_clip: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """NACA 00xx solid with eta in [eta_clip, 1 - eta_clip] (no LE/TE singularity)."""
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


def run_three_engines(
    coords: np.ndarray, quads: np.ndarray, mat, alpha_deg: float, label: str,
) -> None:
    n_cells = quads.shape[0]
    is_iso = type(mat).__name__ == "IsotropicMaterial"
    row = iso_row(mat) if is_iso else ortho_row(mat)
    mat_props = np.tile(row, (n_cells, 1))
    theta_arr = np.full(n_cells, np.radians(alpha_deg))

    out = Path(__file__).parent / "cross_check_out"
    out.mkdir(parents=True, exist_ok=True)
    vtu = out / f"af3w_{label}.vtu"
    write_gx_vtu(vtu, coords, quads, mat_props, theta_arr)

    # gxbeam_section
    K_gx, _S, _M, *_ = _gxbeam_section_from_arrays(coords, quads, mat_props, theta_arr)

    # b3_secfem
    info = from_gxbeam_vtu(vtu)
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=[mat] * info["n_cells"],
        per_cell_alpha_deg=np.degrees(info["theta"]),
    )
    res = solve(inp)
    K_us = to_gxbeam_order(res.K)

    # ANBA — fiber=90-alpha, plane=0 (verified mapping)
    anba = anba_section_from_arrays(
        coords, quads, mat,
        fiber_orientation_deg=np.full(n_cells, alpha_deg),
        plane_orientation_deg=np.full(n_cells, 0.0),
    )
    K_anba = to_gxbeam_order(anba["K"])

    # Side-by-side print
    cols = ["EA", "GAx", "GAy", "GJ", "EIx", "EIy"]
    print(f"\n=== {label} (alpha={alpha_deg:g}°) ===")
    print(f"  {'engine':<14} | " + " | ".join(f"{c:>14}" for c in cols))
    print("-" * 116)
    print(f"  {'gxbeam':<14} | " + " | ".join(f"{K_gx[i, i]:>14.4e}" for i in range(6)))
    print(f"  {'b3_secfem':<14} | " + " | ".join(f"{K_us[i, i]:>14.4e}" for i in range(6)))
    print(f"  {'ANBA':<14} | " + " | ".join(f"{K_anba[i, i]:>14.4e}" for i in range(6)))

    eps = 1e-30

    def relto(a, b):
        return [
            (a[i, i] - b[i, i]) / max(abs(b[i, i]), eps) for i in range(6)
        ]

    rel_b3_gx = relto(K_us, K_gx)
    rel_anba_gx = relto(K_anba, K_gx)
    rel_anba_b3 = relto(K_anba, K_us)
    print(f"  {'b3 vs gxbeam':<14} | "
          + " | ".join(f"{100*r:>13.2f}%" for r in rel_b3_gx))
    print(f"  {'ANBA vs gxbeam':<14} | "
          + " | ".join(f"{100*r:>13.2f}%" for r in rel_anba_gx))
    print(f"  {'ANBA vs b3':<14} | "
          + " | ".join(f"{100*r:>13.2f}%" for r in rel_anba_b3))


def main() -> int:
    coords, quads = gen_airfoil(
        chord=1.0, thickness=0.24, n_chord=32, n_thick=8, eta_clip=0.01,
    )
    n_cells = quads.shape[0]
    print(f"NACA 0024, chord=1.0, thickness=0.24, n_chord=32, n_thick=8, "
          f"eta_clip=0.01, n_cells={n_cells}")

    # 1. iso steel as a baseline check (one row only — skip if all agree)
    run_three_engines(coords, quads, STEEL, 0.0, "iso_steel")

    # 2. UD carbon at α=0  (fibre along beam axis z)
    run_three_engines(coords, quads, UD_CARBON, 0.0, "ud_carbon_a0")

    # 3. UD carbon at α=45° (the discriminating case the user asked for)
    run_three_engines(coords, quads, UD_CARBON, 45.0, "ud_carbon_a45")

    return 0


if __name__ == "__main__":
    sys.exit(main())
