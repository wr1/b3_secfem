"""Cross-backend stiffness-operator comparison + timing harness.

Why this exists
---------------
The fenicsx backend is built on the system (apt) dolfinx, which links PETSc
and MPI; the mfem backend links its own native libraries. Importing both into
a single interpreter works for assembly, but PETSc/MPI atexit teardown then
segfaults the process on exit, and the fenicsx CSR view can alias PETSc memory
that is freed when the PETSc ``Mat`` is destroyed. Both problems vanish if each
backend is exercised in its own short-lived subprocess that serialises its
result to disk -- which is what :func:`assemble_in_subprocess` does. As a bonus
the subprocess measures only the assembly call, not interpreter/library import.

The comparison is **permutation-invariant**: dolfinx and mfem number their
global DOFs differently, so the assembled operators are equal only up to a
symmetric permutation ``P A Pᵀ``. We therefore compare the sorted eigenvalue
spectrum, the trace, and the rigid-body null-space dimension -- all invariant
under such a permutation -- rather than entrywise differences.

This currently covers the core in-plane stiffness operator ``E`` (the bilinear
form shared by every warping solve), which is the part of the mfem backend that
is fully implemented. The end-to-end 6x6 ``K`` comparison will be added here
once the mfem chain solve / R-S assembly is complete.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Material presets the worker understands (kept tiny + JSON-trivial).
MATERIALS: dict[str, dict] = {
    "iso": {"kind": "iso", "E": 100e9, "nu": 0.3, "rho": 2000.0},
    "ortho": {
        "kind": "ortho",
        "E1": 140e9,
        "E2": 10e9,
        "E3": 10e9,
        "G12": 5e9,
        "G13": 5e9,
        "G23": 3.5e9,
        "nu12": 0.3,
        "nu13": 0.3,
        "nu23": 0.4,
        "rho": 1600.0,
    },
}


# ---------------------------------------------------------------------------
# Worker: assemble E with one backend, in a fresh process, dump to .npz
# ---------------------------------------------------------------------------

_WORKER = r"""
import json, sys, time
import numpy as np
import scipy.sparse as sp

backend, mesh_path, mat_key, out_npz = sys.argv[1:5]

from b3_secfem.bench import MATERIALS
from b3_secfem import IsotropicMaterial, OrthotropicMaterial

spec = MATERIALS[mat_key]
if spec["kind"] == "iso":
    mat = IsotropicMaterial(E=spec["E"], nu=spec["nu"], rho=spec["rho"])
else:
    mat = OrthotropicMaterial(**{k: v for k, v in spec.items() if k != "kind"})
C_local = mat.C_local()

from b3_secfem.backends import get_backend
be = get_backend(backend)

if backend == "fenicsx":
    from b3_secfem.mesh import read_xdmf
    mesh, _ = read_xdmf(mesh_path)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    native = mesh
else:
    from b3_secfem.config import SectionInput, RegionMat
    inp = SectionInput(mesh_path=mesh_path,
                       region_materials={1: RegionMat(material=mat)}, degree=2)
    native, _tags, n_cells = be._load_mfem_mesh(inp)

C = np.tile(C_local, (n_cells, 1, 1))

t0 = time.perf_counter()
A = be.assemble_stiffness_matrix(C, native, degree=2)
t1 = time.perf_counter()

# Force an owned copy of the CSR triplet: the fenicsx path can alias PETSc
# memory freed on Mat.destroy(); save_npz then guarantees a real copy on disk.
A = sp.csr_matrix((np.array(A.data, copy=True),
                   np.array(A.indices, copy=True),
                   np.array(A.indptr, copy=True)), shape=A.shape)
sp.save_npz(out_npz, A)

print(json.dumps({
    "backend": backend, "n_cells": int(n_cells),
    "shape": list(A.shape), "nnz": int(A.nnz),
    "assemble_s": t1 - t0,
    "trace": float(A.diagonal().sum()),
}))
"""


def assemble_in_subprocess(
    backend: str, mesh_path: str | Path, mat_key: str, out_npz: str | Path
) -> dict:
    """Assemble the E operator for one backend in an isolated subprocess.

    Returns the worker's metadata dict (timing, shape, nnz, trace). Raises
    ``RuntimeError`` with captured stderr if the worker fails (so test
    failures carry the real traceback).
    """
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, backend, str(mesh_path), mat_key, str(out_npz)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{backend} assembly worker failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# Full solve: run a complete solve() per backend in a subprocess, compare K/M
# ---------------------------------------------------------------------------

_FULL_WORKER = r"""
import json, sys, time
import numpy as np

mat_key, mesh_path, backend = sys.argv[1:4]

from b3_secfem.bench import MATERIALS
from b3_secfem import (IsotropicMaterial, OrthotropicMaterial, RegionMat,
                       SectionInput, solve)

spec = MATERIALS[mat_key]
if spec["kind"] == "iso":
    mat = IsotropicMaterial(E=spec["E"], nu=spec["nu"], rho=spec["rho"])
else:
    mat = OrthotropicMaterial(**{k: v for k, v in spec.items() if k != "kind"})

inp = SectionInput(mesh_path=mesh_path,
                   region_materials={1: RegionMat(material=mat)},
                   degree=2, backend=backend)

t0 = time.perf_counter()
res = solve(inp)
t1 = time.perf_counter()

print(json.dumps({
    "backend": backend,
    "solve_s": t1 - t0,
    "K": res.K.tolist(),
    "M": res.M.tolist(),
    "K_section_xy": res.K_section_xy,
    "tension_center": list(res.tension_center),
    "mass_center": list(res.mass_center),
    "shear_center": list(res.shear_center),
}))
"""


def full_solve_in_subprocess(backend: str, mesh_path: str | Path, mat_key: str) -> dict:
    """Run a complete ``solve()`` for one backend in an isolated subprocess."""
    proc = subprocess.run(
        [sys.executable, "-c", _FULL_WORKER, mat_key, str(mesh_path), backend],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{backend} full-solve worker failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    out = json.loads(lines[-1])
    out["K"] = np.array(out["K"])
    out["M"] = np.array(out["M"])
    return out


def run_full_comparison(mat_key: str = "iso", nx: int = 12, ny: int = 8) -> dict:
    """Run the full solve with both backends on one mesh and diff K/M/centres.

    Returns ``{"fenicsx": ..., "mfem": ..., "compare": {...}}`` where the
    compare block holds relative differences on K, M, K_section_xy and the
    absolute centre offsets.
    """
    with tempfile.TemporaryDirectory() as td:
        mesh_xdmf = Path(td) / "mesh.xdmf"
        make_rectangle_xdmf(mesh_xdmf, nx=nx, ny=ny)
        fen = full_solve_in_subprocess("fenicsx", mesh_xdmf, mat_key)
        mf = full_solve_in_subprocess("mfem", mesh_xdmf, mat_key)

    def rel(a, b):
        scale = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
        return float(np.abs(a - b).max() / scale)

    cmp = {
        "K_rel": rel(fen["K"], mf["K"]),
        "M_rel": rel(fen["M"], mf["M"]),
        "K_section_xy_rel": abs(fen["K_section_xy"] - mf["K_section_xy"])
        / max(abs(fen["K_section_xy"]), 1e-300),
        "tension_center_absdiff": float(
            np.abs(
                np.array(fen["tension_center"]) - np.array(mf["tension_center"])
            ).max()
        ),
        "shear_center_absdiff": float(
            np.abs(np.array(fen["shear_center"]) - np.array(mf["shear_center"])).max()
        ),
    }
    return {"fenicsx": fen, "mfem": mf, "compare": cmp}


# ---------------------------------------------------------------------------
# Mesh generation (dolfinx, written to XDMF both backends can read)
# ---------------------------------------------------------------------------


def make_rectangle_xdmf(path: str | Path, a=0.2, b=0.1, nx=16, ny=10) -> None:
    """Write a structured quad rectangle to XDMF (consumed by both backends)."""
    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    from b3_secfem import write_xdmf

    m = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [(-a / 2, -b / 2), (a / 2, b / 2)],
        [nx, ny],
        cell_type=dmesh.CellType.quadrilateral,
    )
    write_xdmf(path, m)


# ---------------------------------------------------------------------------
# Permutation-invariant comparison metrics
# ---------------------------------------------------------------------------


def compare_operators(npz_a: str | Path, npz_b: str | Path) -> dict:
    """Compare two assembled E operators via permutation-invariant metrics."""
    import scipy.sparse as sp

    A = sp.load_npz(str(npz_a)).toarray()
    B = sp.load_npz(str(npz_b)).toarray()

    ea = np.linalg.eigvalsh(A)
    eb = np.linalg.eigvalsh(B)
    scale = max(ea.max(), eb.max())

    def null_dim(e):
        return int((np.abs(e) < 1e-8 * scale).sum())

    return {
        "spectrum_max_reldiff": float(np.abs(ea - eb).max() / scale),
        "trace_reldiff": float(abs(A.trace() - B.trace()) / abs(A.trace())),
        "nulldim_a": null_dim(ea),
        "nulldim_b": null_dim(eb),
        "top5_a": ea[-5:].tolist(),
        "top5_b": eb[-5:].tolist(),
    }


def run_comparison(mat_key: str = "iso", nx: int = 16, ny: int = 10) -> dict:
    """End-to-end: build a mesh, assemble E with both backends, compare.

    Returns ``{"fenicsx": meta, "mfem": meta, "compare": metrics}``.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mesh_xdmf = tmp / "mesh.xdmf"
        make_rectangle_xdmf(mesh_xdmf, nx=nx, ny=ny)

        fen_npz = tmp / "fen.npz"
        mf_npz = tmp / "mf.npz"
        fen_meta = assemble_in_subprocess("fenicsx", mesh_xdmf, mat_key, fen_npz)
        mf_meta = assemble_in_subprocess("mfem", mesh_xdmf, mat_key, mf_npz)
        cmp = compare_operators(fen_npz, mf_npz)

    return {"fenicsx": fen_meta, "mfem": mf_meta, "compare": cmp}


# ---------------------------------------------------------------------------
# Timing analysis
# ---------------------------------------------------------------------------


def time_backends(
    sizes: list[tuple[int, int]] | None = None,
    mat_key: str = "iso",
    repeats: int = 3,
) -> list[dict]:
    """Time E-operator assembly for both backends across mesh sizes.

    For each (nx, ny) the same mesh is assembled ``repeats`` times per backend
    in fresh subprocesses; the *minimum* wall-clock is reported (least noisy
    estimator). Returns one row dict per size with timings and the speed ratio.

    Note: the subprocess measures only the ``assemble_stiffness_matrix`` call,
    not interpreter/library import -- so this compares assembly engines, not
    startup cost.
    """
    sizes = sizes or [(8, 6), (16, 12), (32, 24), (48, 36)]
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for nx, ny in sizes:
            mesh_xdmf = tmp / f"mesh_{nx}x{ny}.xdmf"
            make_rectangle_xdmf(mesh_xdmf, nx=nx, ny=ny)
            times: dict[str, float] = {}
            ncells = ndof = 0
            for be in ("fenicsx", "mfem"):
                best = float("inf")
                for _ in range(repeats):
                    meta = assemble_in_subprocess(
                        be, mesh_xdmf, mat_key, tmp / f"{be}.npz"
                    )
                    best = min(best, meta["assemble_s"])
                    ncells, ndof = meta["n_cells"], meta["shape"][0]
                times[be] = best
            rows.append(
                {
                    "nx": nx,
                    "ny": ny,
                    "n_cells": ncells,
                    "ndof": ndof,
                    "fenicsx_s": times["fenicsx"],
                    "mfem_s": times["mfem"],
                    "ratio_mfem_over_fenicsx": times["mfem"] / times["fenicsx"],
                }
            )
    return rows


def format_timing_table(rows: list[dict]) -> str:
    """Render :func:`time_backends` rows as a fixed-width text table."""
    head = (
        f"{'cells':>7} {'ndof':>8} {'fenicsx [s]':>13} {'mfem [s]':>12} "
        f"{'mfem/fenicsx':>13}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['n_cells']:>7} {r['ndof']:>8} {r['fenicsx_s']:>13.4f} "
            f"{r['mfem_s']:>12.4f} {r['ratio_mfem_over_fenicsx']:>13.2f}"
        )
    return "\n".join(lines)
