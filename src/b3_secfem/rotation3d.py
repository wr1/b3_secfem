"""6x6 stiffness rotation under (beta, alpha).

Three steps (right-hand rule, ``s = +sin``)
-------------------------------------------
1. Section frame: ``sec_1 = x``, ``sec_2 = y``, ``sec_3 = z``
   (RH, beam axis out of the mesh plane).
2. Material frame: ``mat_1`` = fibre, ``mat_2`` = in-ply transverse,
   ``mat_3`` = ply normal. Written on the card before any angle.
3. Transform: ``R(β, α)`` takes a material vector into the section::

       v_section = R(β, α) · v_mat
       R(β, α)   = Rz(β) · Ry(α)

``(β, α) = (0, 0)`` is the identity: ply axes sit on the section axes
(``mat_1 = +x``, ``mat_2 = +y``, ``mat_3 = +z``). That is the same
zero as ANBA ``(fiber, plane) = (0, 0)``.

- ``alpha_deg`` : rotation about ``+y``. ``α = 0`` leaves the fibre
  in the section plane; ``α = 90`` puts it along ``−z`` (beam);
  ``α = −90`` along ``+z``.
- ``beta_deg``  : right-hand rotation about ``+z``. At ``α = 0`` this
  is the in-plane fibre angle from ``+x``.

A material vector sees ``α`` about global ``y``, then ``β`` about
global ``z``. The two angles do **not** commute except when one
factor is ``I``.

ANBA is another solver. Its ``transformation_matrix(plane, fiber)``
(``sn = −sin``) equals this module's bond ``T`` at ``β = plane``,
``α = fiber``. See ``b3_secfem.adapters``.
"""

from __future__ import annotations

import numpy as np

# Section aliases. The solver still names these x, y, z.
SEC_1 = np.array([1.0, 0.0, 0.0])
SEC_2 = np.array([0.0, 1.0, 0.0])
SEC_3 = np.array([0.0, 0.0, 1.0])


def _Rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Ry(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_matrix_3x3(beta_deg: float, alpha_deg: float) -> np.ndarray:
    """Active rotation R = Rz(beta) @ Ry(alpha).

    A vector v_local in the principal frame maps to v_global = R @ v_local.
    R(0, 0) is I.
    """
    return _Rz(beta_deg) @ _Ry(alpha_deg)


def material_axes(beta_deg: float, alpha_deg: float = 0.0) -> dict[str, np.ndarray]:
    """Unit vectors of mat_1, mat_2, mat_3 in section coordinates (x, y, z)."""
    R = _rotation_matrix_3x3(beta_deg, alpha_deg)
    return {
        "mat_1": R @ np.array([1.0, 0.0, 0.0]),
        "mat_2": R @ np.array([0.0, 1.0, 0.0]),
        "mat_3": R @ np.array([0.0, 0.0, 1.0]),
    }


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

    At ``(beta=0, alpha=0)``: fibre along +x, mat_2 along +y, mat_3 along +z.
    """
    R = _rotation_matrix_3x3(beta_deg, alpha_deg)
    T = _bond_T(R)
    return T @ C_local @ T.T


def fibre_direction(beta_deg: float, alpha_deg: float = 0.0) -> np.ndarray:
    """Unit vector pointing in the fibre direction (axis 1) in global coords."""
    R = _rotation_matrix_3x3(beta_deg, alpha_deg)
    return R @ np.array([1.0, 0.0, 0.0])
