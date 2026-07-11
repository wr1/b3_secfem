"""Rotation contract tests — pure numpy, no FEniCSx required."""

from __future__ import annotations

import numpy as np

from b3_secfem.materials import IsotropicMaterial, OrthotropicMaterial
from b3_secfem.rotation3d import (
    _bond_T,
    _rotation_matrix_3x3,
    fibre_direction,
    rotate_stiffness_6x6,
)


def test_default_orientation_is_axial():
    """At (beta=0, alpha=0) the fibre should point along +z."""
    n = fibre_direction(0.0, 0.0)
    np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-15)


def test_alpha_90_puts_fibre_in_plane():
    """At alpha=90 the fibre lies fully in the section (xy) plane.

    beta then sets the angle from the global x axis.
    """
    for beta in (0.0, 30.0, 90.0, -45.0):
        n = fibre_direction(beta, 90.0)
        b = np.deg2rad(beta)
        np.testing.assert_allclose(n, [np.cos(b), np.sin(b), 0.0], atol=1e-12)


def test_alpha_zero_keeps_fibre_axial_regardless_of_beta():
    for beta in (0.0, 17.0, 90.0, -45.0):
        n = fibre_direction(beta, 0.0)
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-12)


def test_zero_rotation_T_is_identity_in_voigt():
    """At (beta=0, alpha=90) we recover the identity T.

    (alpha=90 maps the principal frame onto the global frame: axis 1 -> x.)
    """
    R = _rotation_matrix_3x3(0.0, 90.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
    T = _bond_T(R)
    np.testing.assert_allclose(T, np.eye(6), atol=1e-12)


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
            # Un-rotate via the inverse rotation.
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


def test_axial_orientation_has_E1_along_z():
    """At default orientation, the (3, 3) component of C_global equals E1.

    For an orthotropic with no Poisson coupling under uniaxial sigma_zz, the
    diagonal stiffness in the z direction equals C_local[0, 0] (which is
    NOT the same as E1 because of Poisson — but it IS what the (0, 0)
    diagonal of the unrotated C_local is, regardless of fibre direction
    misalignment).
    """
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
    C_global = rotate_stiffness_6x6(C, 0.0, 0.0)  # fibre along z
    # Voigt order (11, 22, 33, ...). Fibre along z means C_global[2, 2] = C_local[0, 0]
    np.testing.assert_allclose(C_global[2, 2], C[0, 0], rtol=1e-12)
