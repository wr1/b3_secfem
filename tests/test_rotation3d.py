"""Rotation contract tests — pure numpy, no FEniCSx required.

R(β, α) = Rz(β) · Ry(α). (0, 0) is the identity: ply axes sit on
section axes. Fibre along the beam is α = ±90, not a hidden offset.
"""

from __future__ import annotations

import numpy as np

from b3_secfem.materials import IsotropicMaterial, OrthotropicMaterial
from b3_secfem.rotation3d import (
    SEC_1,
    SEC_2,
    SEC_3,
    _bond_T,
    _Ry,
    _Rz,
    _rotation_matrix_3x3,
    fibre_direction,
    material_axes,
    rotate_stiffness_6x6,
)


def test_zero_angles_are_identity():
    """At (0, 0) the ply axes sit on the section axes (1=x, 2=y, 3=z)."""
    n = fibre_direction(0.0, 0.0)
    np.testing.assert_allclose(n, [1.0, 0.0, 0.0], atol=1e-15)
    R = _rotation_matrix_3x3(0.0, 0.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-15)


def test_alpha_90_puts_fibre_on_minus_z():
    """Ry(+90) takes e1 to −z. Stiffness on z is still E1."""
    n = fibre_direction(0.0, 90.0)
    np.testing.assert_allclose(n, [0.0, 0.0, -1.0], atol=1e-12)


def test_alpha_minus_90_puts_fibre_on_plus_z():
    n = fibre_direction(0.0, -90.0)
    np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-12)


def test_beta_rolls_in_plane_at_alpha_zero():
    """At α = 0 the fibre stays in the xy plane; β is its angle from +x."""
    for beta in (0.0, 30.0, 90.0, -45.0):
        n = fibre_direction(beta, 0.0)
        b = np.deg2rad(beta)
        np.testing.assert_allclose(n, [np.cos(b), np.sin(b), 0.0], atol=1e-12)


def test_alpha_zero_keeps_fibre_in_xy_regardless_of_beta():
    for beta in (0.0, 17.0, 90.0, -45.0):
        n = fibre_direction(beta, 0.0)
        np.testing.assert_allclose(n[2], 0.0, atol=1e-12)


def test_isotropic_invariant_under_rotation():
    mat = IsotropicMaterial(E=70e9, nu=0.33, rho=2700.0)
    C = mat.C_local()
    for beta in (0.0, 17.0, 45.0, 90.0, -33.0):
        for alpha in (0.0, 23.0, 60.0, 90.0):
            C_rot = rotate_stiffness_6x6(C, beta, alpha)
            np.testing.assert_allclose(C_rot, C, atol=1e-3, rtol=1e-10)


def test_round_trip_identity():
    """Rotating then un-rotating returns the original stiffness."""
    mat = OrthotropicMaterial(
        E1=140e9,
        E2=10e9,
        E3=10e9,
        G12=5e9,
        G13=5e9,
        G23=3.5e9,
        nu12=0.3,
        nu13=0.3,
        nu23=0.4,
        rho=1600.0,
    )
    C = mat.C_local()
    for beta in (15.0, 30.0, 45.0, 90.0):
        for alpha in (0.0, 30.0, 90.0):
            C1 = rotate_stiffness_6x6(C, beta, alpha)
            R = _rotation_matrix_3x3(beta, alpha)
            T = _bond_T(R)
            T_inv = np.linalg.inv(T)
            C2 = T_inv @ C1 @ T_inv.T
            np.testing.assert_allclose(C2, C, atol=1e-3, rtol=1e-10)


def test_rotated_stiffness_is_symmetric():
    mat = OrthotropicMaterial(
        E1=140e9,
        E2=10e9,
        E3=10e9,
        G12=5e9,
        G13=5e9,
        G23=3.5e9,
        nu12=0.3,
        nu13=0.3,
        nu23=0.4,
        rho=1600.0,
    )
    C = mat.C_local()
    for beta, alpha in ((30.0, 0.0), (45.0, 20.0), (-17.0, 33.0), (12.0, 90.0)):
        C_rot = rotate_stiffness_6x6(C, beta, alpha)
        np.testing.assert_allclose(C_rot, C_rot.T, atol=1e-3)


def test_rotated_stiffness_positive_definite():
    mat = OrthotropicMaterial(
        E1=140e9,
        E2=10e9,
        E3=10e9,
        G12=5e9,
        G13=5e9,
        G23=3.5e9,
        nu12=0.3,
        nu13=0.3,
        nu23=0.4,
        rho=1600.0,
    )
    C = mat.C_local()
    for beta in np.linspace(0.0, 180.0, 9):
        for alpha in (0.0, 45.0, 90.0):
            C_rot = rotate_stiffness_6x6(C, beta, alpha)
            eigs = np.linalg.eigvalsh(C_rot)
            assert eigs.min() > 0.0


def test_zero_orientation_has_E1_along_x():
    """At (0, 0), C_global[xx] equals C_local[11] (fibre on x)."""
    mat = OrthotropicMaterial(
        E1=140e9,
        E2=10e9,
        E3=10e9,
        G12=5e9,
        G13=5e9,
        G23=3.5e9,
        nu12=0.3,
        nu13=0.3,
        nu23=0.4,
        rho=1600.0,
    )
    C = mat.C_local()
    C_global = rotate_stiffness_6x6(C, 0.0, 0.0)
    np.testing.assert_allclose(C_global[0, 0], C[0, 0], rtol=1e-12)


def test_alpha90_puts_E1_on_z():
    """At (0, 90) fibre is along −z ⇒ C_global[zz] = C_local[11]."""
    mat = OrthotropicMaterial(
        E1=140e9,
        E2=80e9,
        E3=20e9,
        G12=5e9,
        G13=6e9,
        G23=3.5e9,
        nu12=0.3,
        nu13=0.25,
        nu23=0.2,
        rho=1600.0,
    )
    C = mat.C_local()
    C_global = rotate_stiffness_6x6(C, 0.0, 90.0)
    np.testing.assert_allclose(C_global[2, 2], C[0, 0], rtol=1e-12)


def test_sec_aliases_are_xyz():
    np.testing.assert_array_equal(SEC_1, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(SEC_2, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(SEC_3, [0.0, 0.0, 1.0])


def test_material_axes_zero_match_section_axes():
    ax = material_axes(0.0, 0.0)
    np.testing.assert_allclose(ax["mat_1"], SEC_1, atol=1e-12)
    np.testing.assert_allclose(ax["mat_2"], SEC_2, atol=1e-12)
    np.testing.assert_allclose(ax["mat_3"], SEC_3, atol=1e-12)


def test_anba_fiber90_plane0_axes():
    """secfem (β, α) = (0, 90) is ANBA (fiber, plane) = (90, 0) for T.

    Signed triad: fibre on −z, mat_2 on +y, mat_3 on +x.
    """
    ax = material_axes(0.0, 90.0)
    np.testing.assert_allclose(ax["mat_1"], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(ax["mat_2"], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(ax["mat_3"], [1.0, 0.0, 0.0], atol=1e-12)


def test_rotation_is_proper():
    """R is a proper rotation (det = +1) for all angles."""
    for beta in (0.0, 30.0, 90.0, -45.0):
        for alpha in (0.0, 45.0, 90.0):
            R = _rotation_matrix_3x3(beta, alpha)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def _swapped_R(beta_deg: float, alpha_deg: float) -> np.ndarray:
    """Ry(α) · Rz(β) — not the chosen product."""
    return _Ry(alpha_deg) @ _Rz(beta_deg)


def test_beta_then_alpha_commutes_only_on_the_axes():
    """Rz(β)·Ry(α) equals Ry(α)·Rz(β) only when one factor is I."""
    for beta in (0.0, 30.0, 45.0, 90.0):
        for alpha in (0.0, 45.0, 90.0):
            chosen = _rotation_matrix_3x3(beta, alpha)
            swapped = _swapped_R(beta, alpha)
            trivial = abs(beta) < 1e-12 or abs(alpha) < 1e-12
            if trivial:
                np.testing.assert_allclose(chosen, swapped, atol=1e-12)
            else:
                assert not np.allclose(chosen, swapped, atol=1e-8), (
                    f"expected non-commute at β={beta} α={alpha}"
                )
