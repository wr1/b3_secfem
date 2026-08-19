"""Adapter contract — pure numpy, no FEniCSx.

ANBA transformation_matrix(plane, fiber) with sn=−sin equals
secfem _bond_T(Rz(plane) · Ry(fiber)). Drop-in is a rename:
β = plane, α = fiber. The card is not rewritten.
"""

from __future__ import annotations

import math

import numpy as np

from b3_secfem.adapters import (
    anba_to_secfem_angles,
    anba_to_secfem_input,
    secfem_from_gxbeam_theta,
    secfem_to_anba_angles,
    secfem_to_anba_input,
    swap23,
)
from b3_secfem.materials import OrthotropicMaterial
from b3_secfem.rotation3d import _bond_T, _Ry, _Rz, rotate_stiffness_6x6


def _anba_T(plane_deg: float, fiber_deg: float) -> np.ndarray:
    """ANBA material_py.transformation_matrix(alpha=plane, beta=fiber)."""
    pi180 = math.pi / 180.0
    sn_a = -math.sin(plane_deg * pi180)
    cn_a = math.cos(plane_deg * pi180)
    sn_b = -math.sin(fiber_deg * pi180)
    cn_b = math.cos(fiber_deg * pi180)
    T = np.zeros((6, 6))
    T[0, 0] = cn_a * cn_a * cn_b * cn_b
    T[0, 1] = sn_a * sn_a
    T[0, 2] = cn_a * cn_a * sn_b * sn_b
    T[0, 3] = -2.0 * cn_a * sn_a * sn_b
    T[0, 4] = -2.0 * cn_a * cn_a * sn_b * cn_b
    T[0, 5] = 2.0 * cn_a * sn_a * cn_b
    T[1, 0] = sn_a * sn_a * cn_b * cn_b
    T[1, 1] = cn_a * cn_a
    T[1, 2] = sn_a * sn_a * sn_b * sn_b
    T[1, 3] = 2.0 * cn_a * sn_a * sn_b
    T[1, 4] = -2.0 * sn_a * sn_a * sn_b * cn_b
    T[1, 5] = -2.0 * cn_a * sn_a * cn_b
    T[2, 0] = sn_b * sn_b
    T[2, 2] = cn_b * cn_b
    T[2, 4] = 2.0 * cn_b * sn_b
    T[3, 0] = -sn_a * sn_b * cn_b
    T[3, 2] = sn_a * sn_b * cn_b
    T[3, 3] = cn_a * cn_b
    T[3, 4] = -sn_a * cn_b * cn_b + sn_a * sn_b * sn_b
    T[3, 5] = cn_a * sn_b
    T[4, 0] = cn_a * sn_b * cn_b
    T[4, 2] = -cn_a * sn_b * cn_b
    T[4, 3] = sn_a * cn_b
    T[4, 4] = -cn_a * sn_b * sn_b + cn_a * cn_b * cn_b
    T[4, 5] = sn_a * sn_b
    T[5, 0] = -sn_a * cn_a * cn_b * cn_b
    T[5, 1] = cn_a * sn_a
    T[5, 2] = -sn_a * cn_a * sn_b * sn_b
    T[5, 3] = -cn_a * cn_a * sn_b + sn_a * sn_a * sn_b
    T[5, 4] = 2.0 * sn_a * sn_b * cn_a * cn_b
    T[5, 5] = cn_a * cn_a * cn_b - sn_a * sn_a * cn_b
    return T


def _asym() -> OrthotropicMaterial:
    return OrthotropicMaterial(
        E1=160e9,
        E2=80e9,
        E3=20e9,
        G12=8e9,
        G13=6e9,
        G23=4e9,
        nu12=0.25,
        nu13=0.30,
        nu23=0.35,
        rho=1500.0,
        name="asym",
    )


def test_anba_angles_identity():
    beta, alpha = anba_to_secfem_angles(0.0, 0.0)
    assert beta == 0.0 and alpha == 0.0


def test_anba_fiber90_is_secfem_alpha90():
    beta, alpha = anba_to_secfem_angles(90.0, 0.0)
    assert beta == 0.0 and alpha == 90.0


def test_anba_angle_round_trip():
    for fiber in (0.0, 30.0, 90.0):
        for plane in (0.0, 30.0, -45.0):
            beta, alpha = anba_to_secfem_angles(fiber, plane)
            f2, p2 = secfem_to_anba_angles(beta, alpha)
            np.testing.assert_allclose([f2, p2], [fiber, plane])


def test_gxbeam_theta_is_alpha():
    beta, alpha = secfem_from_gxbeam_theta(np.pi / 4)
    assert beta == 0.0
    np.testing.assert_allclose(alpha, 45.0)


def test_anba_T_equals_secfem_bond_T():
    """First-principles map: ANBA T(plane, fiber) = bond_T(Rz(plane) Ry(fiber))."""
    for fiber in (0.0, 30.0, 45.0, 90.0):
        for plane in (0.0, 30.0, -45.0, 90.0):
            beta, alpha = anba_to_secfem_angles(fiber, plane)
            Ts = _bond_T(_Rz(beta) @ _Ry(alpha))
            Ta = _anba_T(plane, fiber)
            np.testing.assert_allclose(Ts, Ta, atol=1e-12)


def test_same_card_same_C_after_angle_map():
    """Drop-in does not rewrite the card. C̄ matches on the unmapped card."""
    mat = _asym()
    C = mat.C_local()
    for fiber in (0.0, 30.0, 90.0):
        for plane in (0.0, 30.0, -45.0):
            beta, alpha = anba_to_secfem_angles(fiber, plane)
            C_sec = rotate_stiffness_6x6(C, beta, alpha)
            T_anba = _anba_T(plane, fiber)
            C_anba = T_anba @ C @ T_anba.T
            np.testing.assert_allclose(C_sec, C_anba, rtol=1e-10, atol=1e-3)


def test_anba_to_secfem_input_does_not_swap_card():
    mat = _asym()
    placed = anba_to_secfem_input(mat, 90.0, 0.0)
    assert placed.material.E1 == mat.E1
    assert placed.material.E2 == mat.E2
    assert placed.material.E3 == mat.E3
    assert placed.beta_deg == 0.0
    assert placed.alpha_deg == 90.0


def test_secfem_to_anba_input_does_not_swap_card():
    mat = _asym()
    placed = secfem_to_anba_input(mat, 0.0, 90.0)
    assert placed.material.E2 == mat.E2
    assert placed.fiber_deg == 90.0
    assert placed.plane_deg == 0.0


def test_swap23_is_involution():
    mat = _asym()
    twice = swap23(swap23(mat))
    assert twice.E2 == mat.E2 and twice.E3 == mat.E3
    assert twice.G12 == mat.G12 and twice.G13 == mat.G13
