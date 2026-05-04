"""Isotropic rectangle: smoke-test the b3_secfem 6-unit-load solver."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from rich.console import Console

from b3_secfem import (
    IsotropicMaterial,
    RegionMat,
    SectionInput,
    solve,
    write_xdmf,
)


def main() -> None:
    a, b = 0.10, 0.10  # 100 x 100 mm steel section
    n = 16
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rect.xdmf"
        m = dmesh.create_rectangle(
            MPI.COMM_WORLD,
            [(-a / 2, -b / 2), (a / 2, b / 2)],
            [n, n],
            cell_type=dmesh.CellType.quadrilateral,
        )
        write_xdmf(path, m)

        inp = SectionInput(
            mesh_path=path,
            region_materials={1: RegionMat(material=iso)},
        )
        res = solve(inp)

    EA = iso.E * a * b
    G = iso.E / (2 * (1 + iso.nu))
    GJ_roark = 0.141 * G * a ** 4

    console = Console()
    np.set_printoptions(precision=3)
    console.print("K (6x6) [Fx, Fy, Fz, Mx, My, Mz]:")
    console.print(res.K)
    console.print(f"\nE * A           = {EA:.3e}     (got K[Fz,Fz] = {res.K[2, 2]:.3e})")
    console.print(f"0.141 G a^4     = {GJ_roark:.3e}     (got K[Mz,Mz] = {res.K[5, 5]:.3e})")
    console.print(f"\nTension centre  = {res.tension_center}")
    console.print(f"Elastic centre  = {res.elastic_center}")


if __name__ == "__main__":
    main()
