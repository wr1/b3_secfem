"""Cross-section warping solver: public dispatch to FEM backends.

Coordinate convention (single source of truth):

    Beam axis   = z (out of section plane)
    Section     = (x, y)
    Output K    = 6x6 in [Fx, Fy, Fz, Mx, My, Mz] order

Two-stage chain solve (Morandini 2010, simplified) — see
``backends.fenicsx.solver`` (reference) and ``backends.mfem`` (cross-check).

Mode constants are re-exported from ``backends.common`` for recovery and
callers that historically imported them from this module.
"""

from __future__ import annotations

from .backends.common import ALL_MODES, STAGE1_MODES, STAGE2_MODES
from .config import SectionInput
from .result import SectionResult

# Re-export for existing internal imports (recovery.py etc.)
__all__ = ["ALL_MODES", "STAGE1_MODES", "STAGE2_MODES", "solve"]


def solve(inp: SectionInput) -> SectionResult:
    """Run the cross-section chain solve and return K, M, centres.

    Dispatches to the backend named in ``inp.backend`` (default ``"fenicsx"``).
    """
    from .backends import get_backend
    from .backends.common import get_backend_name_from_inp

    name = get_backend_name_from_inp(inp)
    backend = get_backend(name)
    return backend.solve(inp)
