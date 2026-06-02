"""Kinematic helpers: 3D gradient on 2D mesh, strain, stress, Voigt mapping.

We model a prismatic beam aligned with z. The cross-section lives in the
(x, y) plane. The displacement field u is 3D (ux, uy, uz) but only varies
in (x, y), so derivatives in z are zero. The 3D infinitesimal strain
tensor still has nine components, and we keep it in Voigt order
(11, 22, 33, 23, 13, 12).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def grad3d(u: Any) -> Any:
    """3x3 gradient tensor of a 3D displacement on a 2D mesh.

    Derivatives in z are zero; the third column of grad(u) is zero.
    """
    import ufl

    g = ufl.grad(u)  # shape (3, 2): du_i / dx_j with j in {x, y}
    return ufl.as_tensor(
        [
            [g[0, 0], g[0, 1], 0.0],
            [g[1, 0], g[1, 1], 0.0],
            [g[2, 0], g[2, 1], 0.0],
        ]
    )


def epsilon(u: Any) -> Any:
    """3x3 small-strain tensor of u (symmetric part of grad3d)."""

    G = grad3d(u)
    return 0.5 * (G + G.T)


def voigt_strain(u: Any) -> Any:
    """Engineering-shear Voigt strain (11, 22, 33, 2*23, 2*13, 2*12)."""
    import ufl

    e = epsilon(u)
    return ufl.as_vector(
        [
            e[0, 0],
            e[1, 1],
            e[2, 2],
            2.0 * e[1, 2],
            2.0 * e[0, 2],
            2.0 * e[0, 1],
        ]
    )


def sigma_voigt(C_func: Any, u: Any) -> Any:
    """Voigt stress sigma = C(x, y) @ voigt_strain(u).

    C_func is a dolfinx Function on a DG-0 (6, 6) tensor space holding
    the rotated stiffness per element.
    """
    import ufl

    return ufl.dot(C_func, voigt_strain(u))


def epsilon_z_voigt(v: Any) -> Any:
    """Voigt-strain contribution from a z-derivative-like 3D field v.

    If u(x, y, z) is expanded as u = (...) + z * v(x, y) + (...), then
    the strain ``epsilon`` of u contains a z-independent part

        epsilon_zz = v_z
        epsilon_xz = (1/2) v_x   ->  Voigt[4] = 2 eps_xz = v_x
        epsilon_yz = (1/2) v_y   ->  Voigt[3] = 2 eps_yz = v_y
        epsilon_xx = epsilon_yy = epsilon_xy = 0   (no in-plane gradient of v)

    Returned in Voigt order (11, 22, 33, 23, 13, 12) with engineering
    shears.
    """
    import ufl

    return ufl.as_vector([0.0, 0.0, v[2], v[1], v[0], 0.0])


def voigt_to_tensor(s_v: Any) -> Any:
    """Voigt-stress 6-vector to 3x3 symmetric stress tensor."""
    import ufl

    return ufl.as_tensor(
        [
            [s_v[0], s_v[5], s_v[4]],
            [s_v[5], s_v[1], s_v[3]],
            [s_v[4], s_v[3], s_v[2]],
        ]
    )


def rigid_body_displacement(x: np.ndarray, mode: int) -> np.ndarray:
    """Six rigid-body modes evaluated at a (3,) point or (n, 3) cloud.

    Mode order matches the 6-DOF generalised displacement
    [u_x, u_y, u_z, theta_x, theta_y, theta_z] at the section origin.
    Returns a (..., 3) displacement vector.
    """
    if x.ndim == 1:
        x = x.reshape(1, 3)
    n = x.shape[0]
    out = np.zeros((n, 3))
    if mode == 0:
        out[:, 0] = 1.0
    elif mode == 1:
        out[:, 1] = 1.0
    elif mode == 2:
        out[:, 2] = 1.0
    elif mode == 3:
        # rotation about x: u = (0, -z, y)
        out[:, 1] = -x[:, 2]
        out[:, 2] = x[:, 1]
    elif mode == 4:
        # rotation about y: u = (z, 0, -x)
        out[:, 0] = x[:, 2]
        out[:, 2] = -x[:, 0]
    elif mode == 5:
        # rotation about z: u = (-y, x, 0)
        out[:, 0] = -x[:, 1]
        out[:, 1] = x[:, 0]
    else:
        msg = f"mode must be in 0..5, got {mode}"
        raise ValueError(msg)
    return out


def section_nullspace_modes() -> list[tuple[str, int]]:
    """The 4-D nullspace of the 2D Saint-Venant problem.

    Translations in x and y, translation in z, and rotation about z
    (torsion) are the rigid-body / unconstrained modes. Bending modes
    about x and y are not in the nullspace because they couple to the
    z-direction warping through axial strain.
    """
    return [("ux", 0), ("uy", 1), ("uz", 2), ("rot_z", 5)]
