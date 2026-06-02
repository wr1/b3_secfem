"""6x6 stiffness rotation under (beta, alpha) two-angle convention.

Convention (right-hand rule, ``s = +sin``)
-----------------------------------------
The material principal frame has axis 1 along the fibre. By default —
``(beta_deg, alpha_deg) = (0, 0)`` — axis 1 of the principal frame is
mapped onto the **global beam axis z**. This matches the gxbeam_section
convention where ``theta = 0`` gives axially-aligned fibres (the typical
case for blade unidirectional spar plies).

- ``beta_deg``  : rotation about the beam axis z, in degrees. Sets the
                  in-plane fibre angle once the fibre is in plane
                  (i.e. once ``alpha != 0``). Has no effect on the
                  fibre direction when ``alpha = 0`` because axis 1 is
                  parallel to z.
- ``alpha_deg`` : tilt of the fibre out of the axial direction toward
                  the section plane. ``alpha = 0``  -> fibre along z
                  (axial).  ``alpha = 90`` -> fibre fully in-plane at
                  angle ``beta`` from the global x axis.

Implementation: ``R(beta, alpha) = R_z(beta) @ R_y(alpha - 90)``,
applied to a vector v as ``R @ v``.

Translation to ANBA4 (which uses ``sn = -sin`` in
``material_py.py:44``): flip the sign of both ``beta`` and ``alpha``.
"""

from __future__ import annotations

import numpy as np


def _Rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Ry(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_matrix_3x3(beta_deg: float, alpha_deg: float) -> np.ndarray:
    """Active rotation R = R_z(beta) @ R_y(alpha - 90), right-hand rule.

    A vector v_local in the principal frame maps to v_global = R @ v_local.
    By construction, R(0, 0) @ (1, 0, 0) = (0, 0, 1) so the fibre lies
    along z by default.
    """
    return _Rz(beta_deg) @ _Ry(alpha_deg - 90.0)


def _bond_T(R: np.ndarray) -> np.ndarray:
    """Bond / Voigt 6x6 rotation matrix for stiffness.

    Voigt order: (11, 22, 33, 23, 13, 12). Builds T such that
        sigma_global_voigt = T @ sigma_local_voigt
    and rotated stiffness C_global = T @ C_local @ T.T.
    """
    a = R
    T = np.array(
        [
            [
                a[0, 0] ** 2,
                a[0, 1] ** 2,
                a[0, 2] ** 2,
                2 * a[0, 1] * a[0, 2],
                2 * a[0, 0] * a[0, 2],
                2 * a[0, 0] * a[0, 1],
            ],
            [
                a[1, 0] ** 2,
                a[1, 1] ** 2,
                a[1, 2] ** 2,
                2 * a[1, 1] * a[1, 2],
                2 * a[1, 0] * a[1, 2],
                2 * a[1, 0] * a[1, 1],
            ],
            [
                a[2, 0] ** 2,
                a[2, 1] ** 2,
                a[2, 2] ** 2,
                2 * a[2, 1] * a[2, 2],
                2 * a[2, 0] * a[2, 2],
                2 * a[2, 0] * a[2, 1],
            ],
            [
                a[1, 0] * a[2, 0],
                a[1, 1] * a[2, 1],
                a[1, 2] * a[2, 2],
                a[1, 1] * a[2, 2] + a[1, 2] * a[2, 1],
                a[1, 0] * a[2, 2] + a[1, 2] * a[2, 0],
                a[1, 0] * a[2, 1] + a[1, 1] * a[2, 0],
            ],
            [
                a[0, 0] * a[2, 0],
                a[0, 1] * a[2, 1],
                a[0, 2] * a[2, 2],
                a[0, 1] * a[2, 2] + a[0, 2] * a[2, 1],
                a[0, 0] * a[2, 2] + a[0, 2] * a[2, 0],
                a[0, 0] * a[2, 1] + a[0, 1] * a[2, 0],
            ],
            [
                a[0, 0] * a[1, 0],
                a[0, 1] * a[1, 1],
                a[0, 2] * a[1, 2],
                a[0, 1] * a[1, 2] + a[0, 2] * a[1, 1],
                a[0, 0] * a[1, 2] + a[0, 2] * a[1, 0],
                a[0, 0] * a[1, 1] + a[0, 1] * a[1, 0],
            ],
        ]
    )
    return T


def rotate_stiffness_6x6(
    C_local: np.ndarray, beta_deg: float, alpha_deg: float = 0.0
) -> np.ndarray:
    """Rotate a 6x6 stiffness from principal frame to global frame.

    Voigt order (11, 22, 33, 23, 13, 12). Right-hand rule.

    At ``(beta=0, alpha=0)`` axis 1 of the principal frame coincides with
    the global beam axis z, i.e. fibre is axial.
    """
    R = _rotation_matrix_3x3(beta_deg, alpha_deg)
    T = _bond_T(R)
    return T @ C_local @ T.T


def fibre_direction(beta_deg: float, alpha_deg: float = 0.0) -> np.ndarray:
    """Unit vector pointing in the fibre direction (axis 1) in global coords."""
    R = _rotation_matrix_3x3(beta_deg, alpha_deg)
    return R @ np.array([1.0, 0.0, 0.0])
