"""Runtime prep and FFCx warm-up for multi-job fenicsx runners.

Surrogate / dataset sweeps launch many medium section solves in spawn workers.
Without care, each worker pays:
  - OpenMP/BLAS oversubscription (if not pinned),
  - first-touch PETSc + dolfinx import,
  - FFCx JIT of every form signature (~2 s cold on small meshes).

Call :func:`prepare_env` in the **parent** before starting a process pool
(children inherit the env). Call :func:`warm_up` **once per worker** (pool
``initializer``) so JIT runs before the first timed design.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Shared JIT cache root. dolfinx stores under ``$XDG_CACHE_HOME/fenics``
# (default ``~/.cache/fenics``). All workers on a machine should share one tree
# so form .so files are built once, not once per process race.
_DEFAULT_CACHE = Path.home() / ".cache"


def prepare_env(*, threads: int = 1, cache_home: str | Path | None = None) -> None:
    """Set process-env knobs for fenicsx multi-job runs.

    Must run in the parent **before** ``ProcessPoolExecutor(..., spawn)`` so
    children inherit the values. Safe to call multiple times (setdefault).

    Parameters
    ----------
    threads :
        BLAS/OpenMP threads per process. Use 1 when parallelising across
        processes (default invsec pattern).
    cache_home :
        ``XDG_CACHE_HOME`` for dolfinx/FFCx. Default leaves existing env or
        ``~/.cache`` so every worker hits the same ``fenics`` JIT directory.
    """
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(threads))
    if cache_home is not None:
        os.environ["XDG_CACHE_HOME"] = str(Path(cache_home).expanduser())
    else:
        os.environ.setdefault("XDG_CACHE_HOME", str(_DEFAULT_CACHE))


def warm_up(
    *,
    linear_solver: str = "lu",
    degree: int = 2,
    nx: int = 6,
    ny: int = 4,
    recover: bool = True,
) -> float:
    """One throwaway fenicsx solve to compile forms and touch PETSc.

    Returns wall seconds spent. Idempotent enough for pool initializers: a
    second call in the same process is a cheap warm solve (forms cached).

    Uses a tiny in-memory rectangle written to a temp XDMF — no caller mesh.
    Skips cleanly (returns 0.0, logs) if dolfinx is not importable.
    """
    import time

    prepare_env()
    try:
        import dolfinx  # noqa: F401
    except ImportError:
        log.info("warm_up: dolfinx not available; skipping")
        return 0.0

    from . import IsotropicMaterial, RegionMat, SectionInput, solve
    from . import recover_unit_load_strains
    from .bench import make_rectangle_xdmf

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="b3_secfem_warm_") as td:
        mesh_path = Path(td) / "warm.xdmf"
        make_rectangle_xdmf(mesh_path, nx=nx, ny=ny)
        mat = IsotropicMaterial(E=100e9, nu=0.3, rho=2000.0)
        inp = SectionInput(
            mesh_path=mesh_path,
            region_materials={1: RegionMat(material=mat)},
            degree=degree,
            backend="fenicsx",
            linear_solver=linear_solver,  # type: ignore[arg-type]
        )
        res = solve(inp)
        if recover:
            recover_unit_load_strains(res)
    dt = time.perf_counter() - t0
    log.info(
        "warm_up done in %.2fs (linear_solver=%s, pid=%s)",
        dt,
        linear_solver,
        os.getpid(),
    )
    return dt
