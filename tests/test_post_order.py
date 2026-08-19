"""Pure-numpy stiffness permutations — no FEM."""

from __future__ import annotations

import numpy as np

from b3_secfem.post import from_gxbeam_order, to_anba_order, to_gxbeam_order


def test_gxbeam_roundtrip():
    rng = np.random.default_rng(0)
    K = rng.normal(size=(6, 6))
    K = 0.5 * (K + K.T)
    assert np.allclose(from_gxbeam_order(to_gxbeam_order(K)), K)


def test_anba_perm_is_documented():
    K = np.arange(36, dtype=float).reshape(6, 6)
    out = to_anba_order(K)
    perm = np.array([2, 5, 0, 1, 3, 4])
    assert np.allclose(out, K[np.ix_(perm, perm)])
