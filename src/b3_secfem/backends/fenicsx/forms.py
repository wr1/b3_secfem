"""UFL forms for the cross-section warping (chain) problem.

Chain formulation (Morandini-Chierichetti-Mantegazza 2010, simplified):

The 3D displacement on a prismatic beam is expanded as a polynomial in the
beam axis z:

    u(x, y, z) = d_0(x, y) + z d_1(x, y) + z^2/2 d_2(x, y) + ...

Substituting into 3D equilibrium and projecting in z gives a chain of 2D
section problems. The relevant bilinear forms are

    E(u, v) = int eps_xy(v)^T C eps_xy(u) dA       (the v0.1 stiffness form)
    C(d, v) = int eps_xy(v)^T C eps_z(d) dA        (z-derivative -> in-plane test)
    H(d, v) = C(d, v) - C^T(d, v)                  (skew coupling)
    M(d, v) = int eps_z(v)^T   C eps_z(d) dA       (pure z-derivative form)

Chain equations:

    E d_1 = -H d_0
    E d_2 =  M d_0 - H d_1

For the four well-posed Stage-1 modes (Fz, Mx, My, Mz), d_0 is a rigid
kinematic field with eps_xy(d_0) = 0; the chain equation collapses to
the v0.1 form ``E d_1 = -C d_0``.

For the two transverse-shear modes (Vx, Vy), the chain extends to d_2
with d_0 the bending kinematic and d_1 the bending warping.

Eps_z(v) returns the Voigt strain contribution from a z-derivative-like
field v: ``[0, 0, v_z, v_y, v_x, 0]`` (engineering shears). See
``kinematics.epsilon_z_voigt``.
"""

from __future__ import annotations

from typing import Any

from ...kinematics import epsilon_z_voigt, voigt_strain


def stiffness_bilinear(C_func: Any, u: Any, v: Any) -> Any:
    """E(u, v) = int_Omega voigt_eps_xy(v)^T C(x, y) voigt_eps_xy(u) dA."""
    import ufl

    return ufl.dot(voigt_strain(v), ufl.dot(C_func, voigt_strain(u))) * ufl.dx


def chain_rhs_stage1(C_func: Any, v: Any, d0: Any) -> Any:
    """RHS for ``E d_1 = -H d_0`` when ``eps_xy(d_0) = 0``.

    With d_0 a rigid kinematic field, ``H d_0 = C d_0`` (the C^T term
    vanishes), so the RHS is

        L(v) = -int_Omega eps_xy(v)^T C(x, y) eps_z(d_0) dA.
    """
    import ufl

    return -ufl.dot(voigt_strain(v), ufl.dot(C_func, epsilon_z_voigt(d0))) * ufl.dx


def chain_rhs_stage2(C_func: Any, v: Any, d0: Any, d1: Any) -> Any:
    """RHS for ``E d_2 = M d_0 - H d_1`` (transverse shear).

    L(v) = int eps_z(v)^T C eps_z(d_0) dA           # M d_0
         - int eps_xy(v)^T C eps_z(d_1) dA          # -C  d_1
         + int eps_z(v)^T C eps_xy(d_1) dA          # +C^T d_1
    """
    import ufl

    eps_v_xy = voigt_strain(v)
    eps_v_z = epsilon_z_voigt(v)
    term_M = ufl.dot(eps_v_z, ufl.dot(C_func, epsilon_z_voigt(d0))) * ufl.dx
    term_C = -ufl.dot(eps_v_xy, ufl.dot(C_func, epsilon_z_voigt(d1))) * ufl.dx
    term_CT = ufl.dot(eps_v_z, ufl.dot(C_func, voigt_strain(d1))) * ufl.dx
    return term_M + term_C + term_CT


def d0_kinematic(mode: int, x: Any) -> Any:
    """Rigid kinematic d_0 field for each chain.

    Modes 0 (Vx) and 4 (My) share the same d_0 = (0, 0, x): the bending-
    about-y kinematic. The shear chain extends one further level (d_2)
    while the bending chain stops at d_1.

    Modes 1 (Vy) and 3 (Mx) share d_0 = (0, 0, -y) similarly.
    """
    import ufl

    if mode in (0, 4):
        return ufl.as_vector([0.0, 0.0, x[0]])  # bending about y
    if mode in (1, 3):
        return ufl.as_vector([0.0, 0.0, -x[1]])  # bending about x
    if mode == 2:
        return ufl.as_vector([0.0, 0.0, 1.0])  # axial
    if mode == 5:
        return ufl.as_vector([-x[1], x[0], 0.0])  # twist
    msg = f"d0_kinematic: mode must be 0..5, got {mode}"
    raise ValueError(msg)


def assumed_inplane_shear_voigt() -> Any:
    """Voigt strain (0, 0, 0, 0, 0, 1) -- unit in-plane (xy) engineering shear.

    Used as the 7th assumed-strain case for computing a homogenised
    section-level in-plane shear stiffness ``K_xy``. Note: gamma_xy is
    not a beam-level kinematic DOF, so this is not part of the 6x6 K --
    it is a separate scalar property of the cross-section.
    """
    import ufl

    return ufl.as_vector([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def inplane_shear_rhs_linear(C_func: Any, v: Any) -> Any:
    """RHS for the in-plane-shear cell problem.

        L(v) = -int eps_xy(v)^T C eps_a^xy dA   with eps_a^xy = (0, ..., 0, 1)

    Solving ``a(w_xy, v) = L(v)`` with the same nullspace projection as
    the rest of the chain gives the warping correction whose total
    Voigt strain is ``eps_a^xy + eps_xy(w_xy)``. The energy

        K_xy_section = int (eps_a + eps_xy(w))^T C (eps_a + eps_xy(w)) dA

    is the section-averaged in-plane shear stiffness. For an isotropic
    homogeneous section it equals G * A.
    """
    import ufl

    eps_v = voigt_strain(v)
    eps_a = assumed_inplane_shear_voigt()
    return -ufl.dot(eps_v, ufl.dot(C_func, eps_a)) * ufl.dx


def total_strain_voigt(mode: int, d0: Any, d1: Any, d2: Any | None) -> Any:
    """Total Voigt strain at z = 0 for the i-th unit-load case.

    Stage 1 (mode 2..5):  eps_total = eps_xy(d_1) + eps_z(d_0).
    Stage 2 (mode 0, 1):  eps_total = eps_xy(d_2) + eps_z(d_1).
    """
    if mode in (0, 1):
        if d2 is None:
            msg = f"mode {mode} (shear) needs d_2"
            raise ValueError(msg)
        return epsilon_z_voigt(d1) + voigt_strain(d2)
    return epsilon_z_voigt(d0) + voigt_strain(d1)
