"""Post-processing: section centres and ordering permutations."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_centres(
    mesh: Any, C_func: Any, rho_func: Any, K: np.ndarray, M: np.ndarray
) -> dict[str, tuple[float, float]]:
    """Tension, elastic, and shear centres of the section.

    - Tension centre (axial-stiffness centroid):
          (xT, yT) = (int E33 x dA, int E33 y dA) / int E33 dA
      where E33 is the (3, 3) component of the rotated stiffness.
    - Elastic / mass centre:
          (xE, yE) = (int rho x dA, int rho y dA) / int rho dA
    - Shear centre (xs, ys): the point where unit transverse forces
      (Vx, Vy) produce no rate of twist about z. Derived from the
      compliance ``S = K^{-1}``:

          xs = -S[5, 1] / S[5, 5]
          ys =  S[5, 0] / S[5, 5]

      A force (Vx, Vy) applied at (xs, ys, 0) is equivalent to the same
      force at the origin plus a moment Mz = xs * Vy - ys * Vx. Setting
      the twist rate row of S * F to zero for arbitrary (Vx, Vy) gives
      the formulas above. Returns (nan, nan) if S[5, 5] is non-finite.
    """
    import ufl
    from dolfinx import fem

    x = ufl.SpatialCoordinate(mesh)
    E33 = ufl.as_tensor(C_func)[2, 2]

    int_E = float(fem.assemble_scalar(fem.form(E33 * ufl.dx)))
    int_E_x = float(fem.assemble_scalar(fem.form(E33 * x[0] * ufl.dx)))
    int_E_y = float(fem.assemble_scalar(fem.form(E33 * x[1] * ufl.dx)))
    tension = (int_E_x / int_E, int_E_y / int_E) if int_E != 0 else (float("nan"),) * 2

    int_rho = float(fem.assemble_scalar(fem.form(rho_func * ufl.dx)))
    if int_rho > 0.0:
        int_rho_x = float(fem.assemble_scalar(fem.form(rho_func * x[0] * ufl.dx)))
        int_rho_y = float(fem.assemble_scalar(fem.form(rho_func * x[1] * ufl.dx)))
        elastic = (int_rho_x / int_rho, int_rho_y / int_rho)
    else:
        elastic = (float("nan"), float("nan"))

    shear: tuple[float, float] = (float("nan"), float("nan"))
    try:
        S = np.linalg.inv(K)
        if np.isfinite(S[5, 5]) and abs(S[5, 5]) > 1e-30:
            shear = (-S[5, 1] / S[5, 5], S[5, 0] / S[5, 5])
    except np.linalg.LinAlgError:
        pass

    return {"tension": tension, "elastic": elastic, "shear": shear}


def to_gxbeam_order(K: np.ndarray) -> np.ndarray:
    """Permute b3_secfem [Fx, Fy, Fz, Mx, My, Mz] -> gxbeam [F1, F2, F3, M1, M2, M3].

    gxbeam: F1 = axial = Fz, F2 = Fx, F3 = Fy, M1 = Mz, M2 = Mx, M3 = My.
    """
    perm = np.array([2, 0, 1, 5, 3, 4])
    return K[np.ix_(perm, perm)]


def to_anba_order(K: np.ndarray) -> np.ndarray:
    """Permute b3_secfem [Fx, Fy, Fz, Mx, My, Mz] -> ANBA chain order.

    ANBA's natural chain ordering after `chains.py` is
        [Fz axial, Mz torsion, Fx, Fy, Mx, My]
    so the permutation is [2, 5, 0, 1, 3, 4].
    """
    perm = np.array([2, 5, 0, 1, 3, 4])
    return K[np.ix_(perm, perm)]
