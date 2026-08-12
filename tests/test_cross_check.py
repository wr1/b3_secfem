"""Cross-checks against gxbeam_section and ANBA4 on a shared mesh.

Parametrized over the iso ring + ellipse subset of the
`examples/cross_check_b3_gx.py` ladder, at smaller `n_cells` so the suite
stays fast. The full report (raw K, M, centres for all six cases including
the airfoil + UD-composite cases) lives with the example driver.

The airfoil cases are deliberately **not** asserted in CI: a mesh-convergence
study (32x8 → 64x16 → 128x32) showed the GJ and EIy entries asymptote to a
finite ~80% disagreement vs gxbeam_section, while EA + EIx converge to <0.1%.
That is real signal — a v0.1 limitation of either b3_secfem's torsion
formulation or gxbeam's, on slender highly-elongated cross-sections — and is
preserved in the driver so it stays visible. This file only gates on cases
that are known to mesh-converge to agreement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")
gxbeam_section = pytest.importorskip("gxbeam_section")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))
from cross_check_b3_gx import (  # noqa: E402
    STEEL,
    Case,
    gen_ellipse_hollow,
    gen_ellipse_solid,
    gen_ring,
    run_case,
)


TEST_CASES = [
    Case(
        "iso_ring",
        lambda: gen_ring(R=0.05, t=0.005, n_circ=32, n_rad=2),
        STEEL,
        is_iso=True,
    ),
    Case(
        "iso_ellipse_solid",
        lambda: gen_ellipse_solid(a=0.06, b=0.02, n_circ=48, n_rad=10),
        STEEL,
        is_iso=True,
    ),
    Case(
        "iso_ellipse_hollow",
        lambda: gen_ellipse_hollow(a=0.06, b=0.02, t=0.005, n_circ=32, n_rad=2),
        STEEL,
        is_iso=True,
    ),
]


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
def test_vs_gxbeam(tmp_path, case):
    """K diag (including transverse shear via Stage-2 d₂ warping) and centres agree with gxbeam_section."""
    rec = run_case(case, tmp_path)

    K_tol = 0.02 if case.is_iso else 0.05
    c_tol = 1e-4 if case.is_iso else 1e-3
    e = np.asarray(rec["K_diag_rel_err"])
    msg_K = f"{case.name}: K_diag_rel_err = {e.tolist()}"
    assert np.all(e < K_tol), msg_K

    c = rec["centre_l2_m"]
    msg_c = f"{case.name}: centre L2 errors = {c}"
    assert c["sc"] < c_tol, msg_c
    assert c["tc"] < c_tol, msg_c
    assert c["mc"] < c_tol, msg_c


@pytest.mark.skip(reason="ANBA4 cross-check fixture not ported (no anba4 package)")
def test_iso_ring_vs_anba(tmp_path):
    """Reserved for future ANBA4 ring comparison once a fixture is wired."""
    _ = tmp_path
