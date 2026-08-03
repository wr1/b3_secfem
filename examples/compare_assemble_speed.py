#!/usr/bin/env python3
"""Compare E-assembly + full solve: fenicsx (lu) vs mfem python vs mfem bulk/numba.

Warm in-process timings (second+ call) so JIT/FFCx is not mixed into the comparison
for fenicsx; numba is warmed once before measuring.

Run:
    micromamba run -n b3secfem python examples/compare_assemble_speed.py
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np


def main() -> None:
    from b3_secfem import IsotropicMaterial, RegionMat, SectionInput, solve, warm_up
    from b3_secfem.backends import mfem as mfem_be
    from b3_secfem.backends.bulk_assemble import _HAS_NUMBA, tabulate_fes
    from b3_secfem.bench import make_rectangle_xdmf

    mat = IsotropicMaterial(E=100e9, nu=0.3, rho=2000.0)
    sizes = [(12, 8), (24, 16), (40, 28)]
    repeats = 3

    # Warm fenicsx forms + numba kernels once
    print("warming fenicsx + numba …")
    os.environ["B3_SECFEM_LINEAR_SOLVER"] = "lu"
    warm_up(linear_solver="lu", recover=False)
    if _HAS_NUMBA:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "w.xdmf"
            make_rectangle_xdmf(p, nx=6, ny=4)
            inp = SectionInput(
                mesh_path=p,
                region_materials={1: RegionMat(material=mat)},
                degree=2,
                backend="mfem",
            )
            os.environ["B3_SECFEM_MFEM_ASSEMBLE"] = "numba"
            solve(inp)  # compile numba + path

    head = (
        f"{'mesh':>9} {'ndof':>7} {'cells':>6} | "
        f"{'fx_asm':>8} {'py_asm':>8} {'nb+tab':>8} {'nb_ker':>8} | "
        f"{'fx_sol':>8} {'py_sol':>8} {'nb_sol':>8} | "
        f"{'Kerr_nb':>9}"
    )
    print(head)
    print("-" * len(head))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for nx, ny in sizes:
            mesh_path = td / f"m_{nx}x{ny}.xdmf"
            make_rectangle_xdmf(mesh_path, nx=nx, ny=ny)
            base = dict(
                mesh_path=mesh_path,
                region_materials={1: RegionMat(material=mat)},
                degree=2,
            )

            # --- assembly-only (mfem paths share same mesh load) ---
            mesh, _tags, n_cells = mfem_be._load_mfem_mesh(
                SectionInput(**base, backend="mfem")
            )
            for e in range(n_cells):
                mesh.GetElement(e).SetAttribute(e + 1)
            mesh.Finalize()
            import mfem.ser as mfem

            fec = mfem.H1_FECollection(2, mesh.Dimension())
            fes = mfem.FiniteElementSpace(mesh, fec, 3)
            C = np.tile(mat.C_local(), (n_cells, 1, 1))
            ndof = fes.GetVSize()

            def time_asm(mode: str) -> float:
                """Full E assembly including MFEM tabulation (bulk) or integrator."""
                best = float("inf")
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    if mode == "python":
                        mfem_be._assemble_voigt_form(
                            C, fes, "xy", "xy", mesh=mesh, mode="python"
                        )
                    else:
                        # retabulate every run — fair vs fenicsx form assembly
                        tabs = tabulate_fes(mesh, fes)
                        mfem_be._assemble_voigt_form(
                            C,
                            fes,
                            "xy",
                            "xy",
                            mesh=mesh,
                            tables=tabs,
                            mode=mode,
                        )
                    best = min(best, time.perf_counter() - t0)
                return best

            py_asm = time_asm("python")
            nb_asm = time_asm("numba") if _HAS_NUMBA else float("nan")
            # kernel-only (pre-tabulated) — shows numba ceil after tabulation fixed
            def time_asm_kernel_only() -> float:
                tabs = tabulate_fes(mesh, fes)
                best = float("inf")
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    mfem_be._assemble_voigt_form(
                        C, fes, "xy", "xy", mesh=mesh, tables=tabs, mode="numba"
                    )
                    best = min(best, time.perf_counter() - t0)
                return best

            nb_kern = time_asm_kernel_only() if _HAS_NUMBA else float("nan")

            # fenicsx assemble via public helper
            from b3_secfem.backends import fenicsx as fx
            from b3_secfem.mesh import read_xdmf

            mesh_fx, _ = read_xdmf(mesh_path)
            mesh_fx.topology.create_connectivity(
                mesh_fx.topology.dim, mesh_fx.topology.dim
            )
            n_fx = mesh_fx.topology.index_map(mesh_fx.topology.dim).size_local
            Cfx = np.tile(mat.C_local(), (n_fx, 1, 1))

            def time_fx_asm() -> float:
                best = float("inf")
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    fx.assemble_stiffness_matrix(Cfx, mesh_fx, degree=2)
                    best = min(best, time.perf_counter() - t0)
                return best

            fx_asm = time_fx_asm()

            # --- full solve warm ---
            def time_solve(backend: str, assemble_mode: str | None = None) -> tuple[float, np.ndarray]:
                if assemble_mode:
                    os.environ["B3_SECFEM_MFEM_ASSEMBLE"] = assemble_mode
                elif "B3_SECFEM_MFEM_ASSEMBLE" in os.environ:
                    del os.environ["B3_SECFEM_MFEM_ASSEMBLE"]
                inp = SectionInput(
                    **base,
                    backend=backend,
                    **({"linear_solver": "lu"} if backend == "fenicsx" else {}),
                )
                solve(inp)  # warm
                best = float("inf")
                res = None
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    res = solve(inp)
                    best = min(best, time.perf_counter() - t0)
                assert res is not None
                return best, res.K.copy()

            fx_sol, K_fx = time_solve("fenicsx")
            py_sol, K_py = time_solve("mfem", "python")
            if _HAS_NUMBA:
                nb_sol, K_nb = time_solve("mfem", "numba")
                kerr = float(np.abs(K_fx - K_nb).max() / max(np.abs(K_fx).max(), 1e-300))
            else:
                nb_sol, kerr = float("nan"), float("nan")
                K_nb = K_py

            # sanity vs python mfem
            kerr_py = float(np.abs(K_fx - K_py).max() / max(np.abs(K_fx).max(), 1e-300))

            print(
                f"{f'{nx}x{ny}':>9} {ndof:7d} {n_cells:6d} | "
                f"{fx_asm:8.4f} {py_asm:8.4f} {nb_asm:8.4f} {nb_kern:8.4f} | "
                f"{fx_sol:8.3f} {py_sol:8.3f} {nb_sol:8.3f} | "
                f"{kerr:9.2e}"
            )
            print(f"{'':>9} {'':>7} {'':>6}   (K max-rel fx vs mfem-python: {kerr_py:.2e})")

    print(
        "\nColumns: fx=fenicsx, py=mfem Py integrator, nb+tab=bulk numba+tabulate, "
        "nb_ker=numba with tables reused; sol=full warm solve. "
        f"numba available: {_HAS_NUMBA}"
    )


if __name__ == "__main__":
    main()
