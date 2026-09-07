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

    ``K`` is assembled about the **section origin** (the mesh/pitch-axis frame), so
    its bending AND torsion diagonals are reference-dependent. Bending carries a
    parallel-axis ``EA * offset^2`` term via the axial-bending coupling
    ``K[Fz, M]`` (decouple at the elastic centre: ``K_elastic_center`` /
    ``bending_stiffness_centroidal``); torsion carries a term via the transverse-
    shear coupling ``K[V, Mz]`` (decouple at the shear centre:
    ``torsional_stiffness_shear_center``). The physical, reference-independent
    stiffnesses are EA = ``K[Fz,Fz]``, the centroidal EI_flap/EI_edge, and the
    shear-centre GJ.

    The ``backend`` tag records which FEM engine produced the result ("fenicsx"
    or "mfem"). The optional payload fields (u_solutions, inplane_shear_warping,
    C_func, Cmat_func, Clocal_func, mesh, oci) contain backend-specific objects
    (dolfinx Functions/Mesh or mfem GridFunctions/Mesh) retained **only** for
    downstream strain/stress recovery and visualisation. They are not validated
    by Pydantic and are unnecessary if you only consume K/M/centres.
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
    Cmat_func: Any | None = None
    """Per-cell ``C_local @ T.T`` so ``sigma_mat = Cmat @ eps_global``."""
    Clocal_func: Any | None = None
    """Per-cell unrotated ply stiffness. ``eps_mat = C_local^{-1} @ sigma_mat``."""
    mesh: Any | None = None
    oci: np.ndarray | None = None
    """Map from dolfinx cell index to input-spec cell index, or identity."""

    @property
    def K_gxbeam_order(self) -> np.ndarray:
        from .post import to_gxbeam_order

        return to_gxbeam_order(self.K)

    @property
    def K_anba_order(self) -> np.ndarray:
        from .post import to_anba_order

        return to_anba_order(self.K)

    @property
    def elastic_center_decoupling(self) -> tuple[float, float]:
        """Section ``(x, y)`` where an axial force produces no bending.

        Solved directly from ``K`` so that translating there exactly cancels the
        axial-bending coupling: ``dx = -K[Fz,My]/K[Fz,Fz]``,
        ``dy = K[Fz,Mx]/K[Fz,Fz]``. (This is the K-self-consistent elastic centre;
        it may differ at the ~1% level from the geometry-based ``elastic_center``/
        ``tension_center`` depending on how those are computed.)
        """
        kaa = float(self.K[2, 2])
        if kaa <= 0.0:
            return 0.0, 0.0
        return -float(self.K[2, 4]) / kaa, float(self.K[2, 3]) / kaa

    @property
    def K_elastic_center(self) -> np.ndarray:
        """The full 6x6 ``K`` translated to the elastic centre (axial-bending decoupled).

        ``K`` itself is referenced to the section origin, so its bending diagonals
        carry a parallel-axis ``EA*offset^2`` term and are reference-dependent.
        This re-references ``K`` to the point where axial load causes no bending
        (``elastic_center_decoupling``) via the congruence transform
        ``K' = P K P^T`` with the moment/force parallel-axis matrix ``P``. The
        result has ``K'[Fz,Mx] = K'[Fz,My] = 0`` and its bending diagonals
        ``K'[Mx,Mx]``, ``K'[My,My]`` are the physical centroidal EI_flap, EI_edge
        (reference-independent). Same numbers as the Schur complement
        ``K[M,M] - K[Fz,M]^2/K[Fz,Fz]``, but the whole matrix (shear/torsion
        couplings included) is moved consistently.
        """
        K = np.asarray(self.K, dtype=float)
        dx, dy = self.elastic_center_decoupling
        P = np.eye(6)
        P[3, 2], P[4, 2] = -dy, dx  # Mx' = Mx - dy.Fz ; My' = My + dx.Fz
        P[5, 0], P[5, 1] = dy, -dx  # Mz' = Mz + dy.Fx - dx.Fy
        return P @ K @ P.T

    @property
    def torsional_stiffness_shear_center(self) -> float:
        """St-Venant torsional stiffness ``GJ`` [N.m^2] about the shear centre.

        The torsion analogue of ``bending_stiffness_centroidal``: ``K[Mz,Mz]`` is
        referenced to the section origin and is reference-dependent — it carries a
        term from the transverse-shear coupling ``K[Vx,Mz]``/``K[Vy,Mz]`` (the
        shear-centre offset). The physical pure torsion (shears free, i.e. about
        the shear centre) is the Schur complement condensing the two transverse-
        shear DOFs::

            GJ = K[Mz,Mz] - K[V,Mz]^T inv(K[V,V]) K[V,Mz],   V = (Vx, Vy)

        For an airfoil this can be ~2x below the raw ``K[Mz,Mz]`` (the shear centre
        sits well forward of the section origin). Bending decouples at the elastic
        centre, torsion at the shear centre — different points, so they are
        exposed separately rather than as one fully-decoupled matrix.
        """
        K = np.asarray(self.K, dtype=float)
        kvv = K[np.ix_([0, 1], [0, 1])]
        kvt = K[[0, 1], 5]
        try:
            return float(K[5, 5] - kvt @ np.linalg.solve(kvv, kvt))
        except np.linalg.LinAlgError:
            return float(K[5, 5])

    @property
    def bending_stiffness_centroidal(self) -> tuple[float, float]:
        """Centroidal bending stiffness ``(EI_x flap, EI_y edge)`` [N.m^2].

        The reference-independent bending EI — the diagonal of
        ``K_elastic_center``. Not edgewise-specific: edgewise is always affected
        (airfoils are fore/aft asymmetric, xc != 0) and flapwise is affected
        whenever yc != 0 (cambered foil or asymmetric spar caps); it is a no-op
        only for a top/bottom-symmetric section.
        """
        Kc = self.K_elastic_center
        return float(Kc[3, 3]), float(Kc[4, 4])

    def __repr__(self) -> str:  # pragma: no cover
        return (
            "SectionResult("
            f"K[Fz,Fz]={self.K[2, 2]:.3e}, "
            f"K[Mx,Mx]={self.K[3, 3]:.3e}, "
            f"K[My,My]={self.K[4, 4]:.3e}, "
            f"K[Mz,Mz]={self.K[5, 5]:.3e}, "
            f"shear={self.shear_center}, tension={self.tension_center}, mass={self.mass_center})"
        )
