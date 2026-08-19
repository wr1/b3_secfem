"""Bulk Voigt-form assembly for the mfem backend (numpy + optional numba).

Replaces the per-element ``PyBilinearFormIntegrator`` Python callback with:

1. One MFEM tabulation pass → plain arrays (dN, N, weights, dofs, C)
2. Local Ke accumulation over all elements (numba if available, else numpy)
3. COO scatter → scipy CSR

Same component-major B layout as :mod:`b3_secfem.backends.mfem` so operators
match the Python integrator up to roundoff.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*_args, **_kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        if _args and callable(_args[0]) and not _kwargs:
            return _args[0]
        return _wrap


# ---------------------------------------------------------------------------
# Numba / pure-python local kernels (identical math)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _fill_B_xy(dN: np.ndarray, B: np.ndarray) -> None:
    """B (6, 3*nd) for eps_xy from physical grads dN (nd, 2). Component-major."""
    nd = dN.shape[0]
    B[:, :] = 0.0
    for k in range(nd):
        B[0, k] = dN[k, 0]
        B[5, k] = dN[k, 1]
        B[1, nd + k] = dN[k, 1]
        B[5, nd + k] = dN[k, 0]
        B[3, 2 * nd + k] = dN[k, 1]
        B[4, 2 * nd + k] = dN[k, 0]


@njit(cache=True)
def _fill_B_z(N: np.ndarray, B: np.ndarray) -> None:
    """B (6, 3*nd) for eps_z from shape values N (nd,). Component-major."""
    nd = N.shape[0]
    B[:, :] = 0.0
    for k in range(nd):
        B[2, 2 * nd + k] = N[k]
        B[3, nd + k] = N[k]
        B[4, k] = N[k]


@njit(cache=True)
def _gemm_Bt_C_Bu(
    Bt: np.ndarray, C: np.ndarray, Bu: np.ndarray, scale: float, Ke: np.ndarray
) -> None:
    """Ke += scale * Bt.T @ C @ Bu  with Bt,Bu shaped (6, nloc)."""
    nloc = Bu.shape[1]
    # tmp = C @ Bu  → (6, nloc)
    tmp = np.zeros((6, nloc))
    for i in range(6):
        for j in range(nloc):
            s = 0.0
            for k in range(6):
                s += C[i, k] * Bu[k, j]
            tmp[i, j] = s
    # Ke += scale * Bt.T @ tmp
    for a in range(nloc):
        for b in range(nloc):
            s = 0.0
            for k in range(6):
                s += Bt[k, a] * tmp[k, b]
            Ke[a, b] += scale * s


@njit(cache=True)
def _assemble_Ke_batch(
    dN: np.ndarray,
    N: np.ndarray,
    w: np.ndarray,
    C: np.ndarray,
    test_xy: int,
    trial_xy: int,
    Ke_out: np.ndarray,
) -> None:
    """Fill Ke_out[e] for all elements.

    dN (ne, nq, nd, 2), N (ne, nq, nd), w (ne, nq), C (ne, 6, 6),
    Ke_out (ne, 3*nd, 3*nd). test_xy/trial_xy are 1 for eps_xy, 0 for eps_z.
    """
    ne, nq, nd, _ = dN.shape
    nloc = 3 * nd
    Bt = np.zeros((6, nloc))
    Bu = np.zeros((6, nloc))
    for e in range(ne):
        Ke = Ke_out[e]
        Ke[:, :] = 0.0
        Ce = C[e]
        for q in range(nq):
            if test_xy == 1:
                _fill_B_xy(dN[e, q], Bt)
            else:
                _fill_B_z(N[e, q], Bt)
            if trial_xy == 1:
                _fill_B_xy(dN[e, q], Bu)
            else:
                _fill_B_z(N[e, q], Bu)
            _gemm_Bt_C_Bu(Bt, Ce, Bu, w[e, q], Ke)


@njit(cache=True)
def _scatter_coo(
    dofs: np.ndarray,
    sign: np.ndarray,
    Ke: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
) -> None:
    """Scatter signed element matrices into COO arrays.

    dofs/sign (ne, nloc); Ke (ne, nloc, nloc); rows/cols/data length ne*nloc*nloc.
    Global entry (i,j) gets sign_i * sign_j * Ke[i,j] (MFEM orientation convention).
    """
    ne, nloc = dofs.shape
    p = 0
    for e in range(ne):
        for i in range(nloc):
            gi = dofs[e, i]
            si = sign[e, i]
            for j in range(nloc):
                rows[p] = gi
                cols[p] = dofs[e, j]
                data[p] = si * sign[e, j] * Ke[e, i, j]
                p += 1


def _assemble_Ke_batch_numpy(
    dN: np.ndarray,
    N: np.ndarray,
    w: np.ndarray,
    C: np.ndarray,
    test_xy: bool,
    trial_xy: bool,
) -> np.ndarray:
    """Vectorised-ish numpy fallback (no numba): still bulk, not per-SWIG-call."""
    ne, nq, nd, _ = dN.shape
    nloc = 3 * nd
    Ke_out = np.zeros((ne, nloc, nloc))
    for e in range(ne):
        acc = np.zeros((nloc, nloc))
        Ce = C[e]
        for q in range(nq):
            Bt = np.zeros((6, nloc))
            Bu = np.zeros((6, nloc))
            if test_xy:
                _fill_B_xy_py(dN[e, q], Bt)
            else:
                _fill_B_z_py(N[e, q], Bt)
            if trial_xy:
                _fill_B_xy_py(dN[e, q], Bu)
            else:
                _fill_B_z_py(N[e, q], Bu)
            acc += (Bt.T @ (Ce @ Bu)) * w[e, q]
        Ke_out[e] = acc
    return Ke_out


def _fill_B_xy_py(dN: np.ndarray, B: np.ndarray) -> None:
    nd = dN.shape[0]
    B[:] = 0.0
    for k in range(nd):
        B[0, k] = dN[k, 0]
        B[5, k] = dN[k, 1]
        B[1, nd + k] = dN[k, 1]
        B[5, nd + k] = dN[k, 0]
        B[3, 2 * nd + k] = dN[k, 1]
        B[4, 2 * nd + k] = dN[k, 0]


def _fill_B_z_py(N: np.ndarray, B: np.ndarray) -> None:
    nd = N.shape[0]
    B[:] = 0.0
    for k in range(nd):
        B[2, 2 * nd + k] = N[k]
        B[3, nd + k] = N[k]
        B[4, k] = N[k]


# ---------------------------------------------------------------------------
# MFEM tabulation → arrays
# ---------------------------------------------------------------------------


def _quad_order(el) -> int:
    """Match mfem._quad_order: 2*order+3 integration."""
    return 2 * el.GetOrder() + 3


def tabulate_fes(mesh: Any, fes: Any) -> dict[str, np.ndarray | int]:
    """Pull quadrature geometry + signed vdofs for every element into arrays."""
    import mfem.ser as mfem

    ne = mesh.GetNE()
    if ne == 0:
        raise ValueError("empty mesh")
    el0 = fes.GetFE(0)
    nd = el0.GetDof()
    ir = mfem.IntRules.Get(el0.GetGeomType(), _quad_order(el0))
    nq = ir.GetNPoints()
    nloc = 3 * nd

    dN = np.zeros((ne, nq, nd, 2), dtype=np.float64)
    N = np.zeros((ne, nq, nd), dtype=np.float64)
    w = np.zeros((ne, nq), dtype=np.float64)
    dofs = np.zeros((ne, nloc), dtype=np.int64)
    sign = np.ones((ne, nloc), dtype=np.float64)

    dref = mfem.DenseMatrix(nd, 2)
    dphys = mfem.DenseMatrix(nd, 2)
    shp = mfem.Vector(nd)

    for e in range(ne):
        el = fes.GetFE(e)
        tr = mesh.GetElementTransformation(e)
        # vdofs component-major; decode orientation signs
        raw = list(fes.GetElementVDofs(e))
        for a, v in enumerate(raw):
            if v >= 0:
                dofs[e, a] = v
                sign[e, a] = 1.0
            else:
                dofs[e, a] = -1 - v
                sign[e, a] = -1.0
        for q in range(nq):
            ip = ir.IntPoint(q)
            tr.SetIntPoint(ip)
            el.CalcDShape(ip, dref)
            mfem.Mult(dref, tr.InverseJacobian(), dphys)
            el.CalcShape(ip, shp)
            dN[e, q] = dphys.GetDataArray().reshape(nd, 2)
            N[e, q] = shp.GetDataArray()
            w[e, q] = tr.Weight() * ip.weight

    return {
        "dN": dN,
        "N": N,
        "w": w,
        "dofs": dofs,
        "sign": sign,
        "ndof_global": int(fes.GetVSize()),
        "nd": nd,
        "nq": nq,
        "ne": ne,
    }


def assemble_voigt_bulk(
    C_per_cell: np.ndarray,
    mesh: Any,
    fes: Any,
    test_kind: Literal["xy", "z"],
    trial_kind: Literal["xy", "z"],
    *,
    engine: Literal["auto", "numba", "numpy"] = "auto",
    tables: dict | None = None,
) -> sp.csr_matrix:
    """Assemble ∫ B_test^T C B_trial dA via bulk (numba/numpy) kernels.

    Parameters
    ----------
    engine :
        ``numba`` forces numba (error if missing), ``numpy`` pure bulk numpy,
        ``auto`` prefers numba when importable.
    tables :
        Optional precomputed :func:`tabulate_fes` result (reuse across E/Cmat/Mmat).
    """
    use_numba = engine == "numba" or (engine == "auto" and _HAS_NUMBA)
    if engine == "numba" and not _HAS_NUMBA:
        raise RuntimeError("numba requested but not installed")

    tabs = tables if tables is not None else tabulate_fes(mesh, fes)
    dN = np.ascontiguousarray(tabs["dN"], dtype=np.float64)
    N = np.ascontiguousarray(tabs["N"], dtype=np.float64)
    w = np.ascontiguousarray(tabs["w"], dtype=np.float64)
    dofs = np.ascontiguousarray(tabs["dofs"], dtype=np.int64)
    sign = np.ascontiguousarray(tabs["sign"], dtype=np.float64)
    C = np.ascontiguousarray(C_per_cell, dtype=np.float64)
    ne = int(tabs["ne"])
    nd = int(tabs["nd"])
    nloc = 3 * nd
    nglob = int(tabs["ndof_global"])
    test_xy = 1 if test_kind == "xy" else 0
    trial_xy = 1 if trial_kind == "xy" else 0

    if use_numba:
        Ke = np.zeros((ne, nloc, nloc), dtype=np.float64)
        _assemble_Ke_batch(dN, N, w, C, test_xy, trial_xy, Ke)
    else:
        Ke = _assemble_Ke_batch_numpy(dN, N, w, C, bool(test_xy), bool(trial_xy))

    nnz = ne * nloc * nloc
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)
    if use_numba:
        _scatter_coo(dofs, sign, Ke, rows, cols, data)
    else:
        # numpy scatter
        p = 0
        for e in range(ne):
            for i in range(nloc):
                gi = dofs[e, i]
                si = sign[e, i]
                for j in range(nloc):
                    rows[p] = gi
                    cols[p] = dofs[e, j]
                    data[p] = si * sign[e, j] * Ke[e, i, j]
                    p += 1

    A = sp.coo_matrix((data, (rows, cols)), shape=(nglob, nglob))
    return A.tocsr()


def resolve_assemble_mode() -> str:
    """``B3_SECFEM_MFEM_ASSEMBLE``: bulk (default) | python | numba | numpy."""
    raw = os.environ.get("B3_SECFEM_MFEM_ASSEMBLE", "bulk").strip().lower()
    if raw in ("bulk", "auto", ""):
        return "numba" if _HAS_NUMBA else "numpy"
    if raw in ("python", "py", "integrator"):
        return "python"
    if raw in ("numba", "numpy"):
        return raw
    return "numba" if _HAS_NUMBA else "numpy"
