#!/usr/bin/env python3
"""Solve a two-material box section with the MFEM backend and visualise the
recovered unit-load strain fields — no dolfinx, no system FEniCS, no Docker.

Demonstrates the pip-wheel-only path (``uv pip install --only-binary=:all:
'mfem>=4.8'``):

- Builds a hollow rectangular box section (carbon-UD caps, glass-UD webs)
  directly with meshio — quad cells, order-preserving.
- Runs the full 6-unit-load Saint-Venant solve with ``backend="mfem"``.
- Calls ``recover_unit_load_strains`` (dispatches to the mfem recovery) and
  plots the axial strain field under unit Mx bending and unit Mz torsion
  shear — the fields the blade 2D workflow consumes.

Run:
    uv run python examples/mfem_strain_recovery.py

Output: ``examples/mfem_strain_out/strain_fields.png``.
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np

from b3_secfem import (
    OrthotropicMaterial,
    SectionInput,
    recover_unit_load_strains,
    solve,
)

OUT = Path(__file__).parent / "mfem_strain_out"

CARBON = OrthotropicMaterial(E1=120e9, E2=8e9, E3=8e9, G12=4e9, G13=4e9,
                             G23=3e9, nu12=0.3, nu13=0.3, nu23=0.45,
                             rho=1570.0, name="carbon_ud")
GLASS = OrthotropicMaterial(E1=38e9, E2=9e9, E3=9e9, G12=3.5e9, G13=3.5e9,
                            G23=3e9, nu12=0.28, nu13=0.28, nu23=0.4,
                            rho=1900.0, name="glass_ud")


def box_mesh(path: Path, a=0.4, b=0.2, t=0.03, n=28):
    """Hollow box outline meshed with quads; returns per-cell material ids
    (0 = caps/carbon on top+bottom walls, 1 = webs/glass on the sides)."""
    xs, ys = np.linspace(0, a, n + 1), np.linspace(0, b, int(n * b / a) + 1)
    pts, quads, mats, idx = [], [], [], {}

    def pid(x, y):
        key = (round(x, 9), round(y, 9))
        if key not in idx:
            idx[key] = len(pts)
            pts.append([x, y])
        return idx[key]

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            inside = t < cx < a - t and t < cy < b - t
            if inside:
                continue
            quads.append([pid(x0, y0), pid(x1, y0), pid(x1, y1), pid(x0, y1)])
            mats.append(0 if (cy < t or cy > b - t) else 1)

    m = meshio.Mesh(np.array(pts), [("quad", np.array(quads))])
    meshio.write(path, m)
    return np.array(mats)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    mesh_path = OUT / "box.vtu"
    mats = box_mesh(mesh_path)
    lib = [CARBON, GLASS]

    inp = SectionInput(
        mesh_path=mesh_path,
        degree=2,
        per_cell_material=[lib[i] for i in mats],
        per_cell_beta_deg=np.zeros(len(mats)),
        per_cell_alpha_deg=np.zeros(len(mats)),
        backend="mfem",
    )
    res = solve(inp)
    print(repr(res))
    fields = recover_unit_load_strains(res)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = meshio.read(mesh_path)
    quads = grid.cells_dict["quad"]
    x, y = grid.points[:, 0], grid.points[:, 1]
    # per-cell values -> flat shading via a PolyCollection
    from matplotlib.collections import PolyCollection

    def panel(ax, values, title):
        pc = PolyCollection(grid.points[quads][:, :, :2], array=values,
                            cmap="RdBu_r", edgecolors="none")
        vmax = np.abs(values).max()
        pc.set_clim(-vmax, vmax)
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        plt.colorbar(pc, ax=ax, shrink=0.8)

    eps = fields.epsilon  # (load case, cell, voigt)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    panel(axes[0], eps[3, :, 2], "unit Mx: axial strain eps_zz")
    panel(axes[1], eps[4, :, 2], "unit My: axial strain eps_zz")
    panel(axes[2], eps[5, :, 5], "unit Mz: shear strain 2 eps_xy")
    fig.suptitle("b3_secfem mfem backend — recovered unit-load strain fields")
    fig.tight_layout()
    out = OUT / "strain_fields.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
