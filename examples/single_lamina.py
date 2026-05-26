"""Single off-axis composite lamina: shows fibre-angle effect on K[Fz,Fz].

For a unidirectional composite lamina, axial stiffness K[Fz, Fz] should
be highest when fibres are aligned with the beam axis (alpha = 0) and
drop sharply as the fibres tilt into the section plane (alpha -> 90 deg)
where transverse modulus E2 dominates.

Also writes representative PNG plots (with centres + neutral axes) to
examples/example_out/ for alpha=0° and 45°.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from rich.console import Console
from rich.table import Table

from b3_secfem import (
    OrthotropicMaterial,
    RegionMat,
    SectionInput,
    plot_section,
    solve,
    write_xdmf,
)


def main() -> None:
    a = b = 0.05  # 50 x 50 mm
    n = 16

    glass = OrthotropicMaterial(
        E1=45e9, E2=12e9, E3=12e9,
        G12=4.5e9, G13=4.5e9, G23=4.0e9,
        nu12=0.3, nu13=0.3, nu23=0.4, rho=2000.0,
    )

    console = Console()
    table = Table(title=f"K[Fz,Fz] vs alpha (fibre tilt) for E1={glass.E1:.1e}, E2={glass.E2:.1e}")
    table.add_column("alpha [deg]", justify="right")
    table.add_column("K[Fz,Fz]", justify="right")
    table.add_column("notional EA", justify="right")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rect.xdmf"
        m = dmesh.create_rectangle(
            MPI.COMM_WORLD,
            [(-a / 2, -b / 2), (a / 2, b / 2)],
            [n, n],
            cell_type=dmesh.CellType.quadrilateral,
        )
        write_xdmf(path, m)

        for alpha in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
            inp = SectionInput(
                mesh_path=path,
                region_materials={1: RegionMat(material=glass, beta_deg=0.0, alpha_deg=alpha)},
            )
            res = solve(inp)
            E_eff = res.K[2, 2] / (a * b)
            table.add_row(f"{alpha:.1f}", f"{res.K[2, 2]:.3e}", f"{E_eff:.3e}")

            # For representative orientations, also emit a plot with centres + neutral axes
            if alpha in (0.0, 45.0):
                out_dir = Path(__file__).parent / "example_out"
                out_dir.mkdir(parents=True, exist_ok=True)
                png = plot_section(
                    inp, res,
                    out_dir / f"single_lamina_alpha{int(alpha)}.png",
                    title=f"Single lamina glass-UD, alpha={alpha:.0f}° (homogeneous → centres at origin)",
                )
                print(f"  wrote {png}")  # console not yet in scope, use print

    console.print(table)


if __name__ == "__main__":
    main()
