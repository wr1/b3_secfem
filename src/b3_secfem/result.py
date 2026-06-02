"""SectionResult: solver output container."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class SectionResult(BaseModel):
    """Result of a single cross-section solve.

    K, M, S, R are pure numpy arrays (6, 6) in [Fx, Fy, Fz, Mx, My, Mz] order.
    Centres are (x, y) tuples in section coordinates.
    mass_center is the rho-weighted centroid; elastic_center is its alias for compat.

    The ``backend`` tag records which FEM engine produced the result ("fenicsx"
    or "mfem"). The four optional payload fields (u_solutions, inplane_shear_warping,
    C_func, mesh) contain backend-specific objects (dolfinx Functions/Mesh or
    mfem GridFunctions/Mesh) retained **only** for downstream strain/stress
    recovery and visualisation. They are not validated by Pydantic and are
    unnecessary if you only consume K/M/centres.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    K: np.ndarray
    M: np.ndarray
    S: np.ndarray
    R: np.ndarray
    backend: str = "fenicsx"
    """6x6 basis-resultant matrix: R[a, i] = a-th generalised-force resultant
    of the i-th chain-basis strain field. Together with the energy matrix,
    ``K = R @ inv(S_energy) @ R^T``. Needed by ``recover_unit_load_strains``
    to map applied unit loads back to basis-field amplitudes via
    ``gamma = inv(R) @ e_k`` — the basis amplitudes are NOT generalised
    strains (see solver.py module docstring), so ``inv(K)`` is the wrong
    weighting."""
    shear_center: tuple[float, float]
    tension_center: tuple[float, float]
    elastic_center: tuple[float, float]
    mass_center: tuple[float, float]

    K_section_xy: float | None = None
    """Section-averaged in-plane shear stiffness [Pa * m^2].

    NOT part of the 6x6 beam K (gamma_xy is not a beam DOF). Computed
    by solving an additional cell problem with assumed strain
    ``(0, 0, 0, 0, 0, 1)``. For an isotropic homogeneous section equals
    G * A.
    """

    u_solutions: list[Any] | None = None
    inplane_shear_warping: Any | None = None
    """Backend-specific warping field (dolfinx Function or mfem GridFunction)
    for the 7th in-plane-shear cell problem. Used by viz.plot_warping."""
    C_func: Any | None = None
    mesh: Any | None = None

    @property
    def K_gxbeam_order(self) -> np.ndarray:
        from .post import to_gxbeam_order

        return to_gxbeam_order(self.K)

    @property
    def K_anba_order(self) -> np.ndarray:
        from .post import to_anba_order

        return to_anba_order(self.K)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            "SectionResult("
            f"K[Fz,Fz]={self.K[2, 2]:.3e}, "
            f"K[Mx,Mx]={self.K[3, 3]:.3e}, "
            f"K[My,My]={self.K[4, 4]:.3e}, "
            f"K[Mz,Mz]={self.K[5, 5]:.3e}, "
            f"shear={self.shear_center}, tension={self.tension_center}, mass={self.mass_center})"
        )
