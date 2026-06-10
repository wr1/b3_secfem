"""Pure-numpy material tests."""

from __future__ import annotations

import numpy as np

from b3_secfem.materials import IsotropicMaterial, OrthotropicMaterial


def test_isotropic_C_local_symmetric_pos_def():
    mat = IsotropicMaterial(E=70e9, nu=0.33, rho=2700.0)
    C = mat.C_local()
    np.testing.assert_allclose(C, C.T, atol=1e-3)
    eigs = np.linalg.eigvalsh(C)
    assert eigs.min() > 0


def test_isotropic_recovers_E_and_nu():
    """Compliance of C_iso recovers E along axis 1 and nu_12."""
    mat = IsotropicMaterial(E=70e9, nu=0.33, rho=2700.0)
    S = np.linalg.inv(mat.C_local())
    np.testing.assert_allclose(1.0 / S[0, 0], 70e9, rtol=1e-9)
    np.testing.assert_allclose(-S[0, 1] * 70e9, 0.33, rtol=1e-9)


def test_orthotropic_C_local_symmetric_pos_def():
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
    np.testing.assert_allclose(C, C.T, atol=1e-3)
    eigs = np.linalg.eigvalsh(C)
    assert eigs.min() > 0


def test_orthotropic_recovers_E_and_nu():
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
    S = np.linalg.inv(mat.C_local())
    np.testing.assert_allclose(1.0 / S[0, 0], 140e9, rtol=1e-9)
    np.testing.assert_allclose(1.0 / S[1, 1], 10e9, rtol=1e-9)
    np.testing.assert_allclose(1.0 / S[2, 2], 10e9, rtol=1e-9)
    # nu12 = -S[1,0] * E1
    np.testing.assert_allclose(-S[1, 0] * 140e9, 0.3, rtol=1e-9)
    # Shear blocks
    np.testing.assert_allclose(1.0 / S[3, 3], 3.5e9, rtol=1e-9)  # G23
    np.testing.assert_allclose(1.0 / S[4, 4], 5e9, rtol=1e-9)  # G13
    np.testing.assert_allclose(1.0 / S[5, 5], 5e9, rtol=1e-9)  # G12
