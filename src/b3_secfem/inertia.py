"""6x6 cross-section mass matrix.

Direct integration of
    M_ij = int_Omega rho(x, y) * R_i(x, y)^T R_j(x, y) dA

where R_i(x, y) is the i-th rigid-body kinematic mode evaluated at the
in-section point (x, y, 0). For a 2D section in the (x, y) plane the
6 rigid-body modes give the closed-form mass matrix

    [[ m_tot     0         0        0        m_z      -m_y    ]
     [ 0         m_tot     0       -m_z      0        m_x     ]
     [ 0         0         m_tot    m_y     -m_x      0       ]
     [ 0        -m_z       m_y      Iyy+Izz -Ixy     0        ]
     [ m_z       0        -m_x     -Ixy     Ixx+Izz  0        ]
     [-m_y       m_x       0        0       0         Ixx+Iyy ]]

where for a section at z = 0:
    m_tot = int rho dA
    m_x   = int rho * x dA
    m_y   = int rho * y dA
    m_z   = 0   (section sits at z = 0)
    Ixx   = int rho * y^2 dA
    Iyy   = int rho * x^2 dA
    Izz   = 0   (no z-extent)
    Ixy   = int rho * x * y dA

This collapses to the familiar 6x6 form found in beam-section codes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def assemble_mass(mesh: Any, rho_func: Any) -> np.ndarray:
    """Compute the 6x6 cross-section mass matrix by closed-form integration."""
    import ufl
    from dolfinx import fem

    x = ufl.SpatialCoordinate(mesh)
    one = fem.Constant(mesh, 1.0)
    forms = {
        "m": rho_func * one * ufl.dx,
        "mx": rho_func * x[0] * ufl.dx,
        "my": rho_func * x[1] * ufl.dx,
        "Ixx": rho_func * x[1] ** 2 * ufl.dx,
        "Iyy": rho_func * x[0] ** 2 * ufl.dx,
        "Ixy": rho_func * x[0] * x[1] * ufl.dx,
    }
    vals = {k: float(fem.assemble_scalar(fem.form(f))) for k, f in forms.items()}

    m = vals["m"]
    mx, my = vals["mx"], vals["my"]
    Ixx, Iyy, Ixy = vals["Ixx"], vals["Iyy"], vals["Ixy"]

    M = np.array(
        [
            [m, 0, 0, 0, 0, -my],
            [0, m, 0, 0, 0, mx],
            [0, 0, m, my, -mx, 0],
            [0, 0, my, Ixx, -Ixy, 0],
            [0, 0, -mx, -Ixy, Iyy, 0],
            [-my, mx, 0, 0, 0, Ixx + Iyy],
        ],
        dtype=float,
    )
    # Symmetrise small numerical drift
    return 0.5 * (M + M.T)
