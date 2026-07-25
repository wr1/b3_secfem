"""Per-cell strain and stress field recovery for the Saint-Venant problem.

This module provides two related but distinct recoveries:

1. Kinematic basis fields (``recover_strains``)
   The 6 fields whose total Voigt strains ``eps_total^(i)`` are the ones
   used internally to build the resultant matrix R and the energy matrix
   that yield K = R @ inv(S_energy) @ R^T.  These correspond to the unit
   generalised-strain assumptions (unit ε_zz, unit κ_x, unit κ_y, unit κ_z,
   plus the bending kinematics that drive the shear chains).

2. Applied unit-load fields (``recover_unit_load_strains``)
   The actual 3D strain/stress distributions that arise under the six
   *applied unit load cases* [Fx=1, Fy=1, Fz=1, Mx=1, My=1, Mz=1] at the
   section origin.  These are obtained by inverting the basis-resultant
   matrix:

       gamma = inv(R) @ e_k
       eps_k = Σ_i gamma[i] * basis_eps[i]
       sig_k = Σ_i gamma[i] * basis_sig[i]

   where R is the 6x6 basis-resultant matrix stored on the
   ``SectionResult``. (``inv(K)`` is the wrong weighting: the chain basis
   amplitudes are not generalised strains — see solver.py.)

The two sets of fields are related but not identical.  Most engineering
use cases (fatigue, damage, sub-modelling) want the unit-load version.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from .forms import d0_kinematic, total_strain_voigt
from .result import SectionResult
from .solver import ALL_MODES, STAGE2_MODES


class StrainField(BaseModel):
    """Per-cell Voigt strain and stress for the 6 *kinematic basis* fields.

    These are the fields whose total Voigt strains ``eps_total^(i)`` are the
    ones used internally by the solver to assemble the resultant matrix R
    (columns) and the energy matrix (see solver._assemble_resultants and
    _assemble_energy_matrix).  They correspond one-to-one with the six
    generalised-strain assumptions (unit axial strain, unit curvatures κ_x/κ_y,
    unit twist rate κ_z, and the two bending kinematics that drive the shear
    chains).

    Most users who want "stress under unit Fx / unit Mz ..." should call
    ``recover_unit_load_strains`` instead.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    epsilon: np.ndarray  # (6, n_cells, 6) Voigt strain (engineering shears)
    sigma: np.ndarray  # (6, n_cells, 6) Voigt stress
    cell_areas: np.ndarray  # (n_cells,)


class UnitLoadStrainField(BaseModel):
    """Per-cell Voigt strain and stress for the 6 *applied unit load cases*.

    Ordering matches the 6x6 K: [Fx, Fy, Fz, Mx, My, Mz].

    These fields are the actual 3D strain/stress distributions that arise when
    a unit force or moment (one of the six standard load cases applied at the
    section origin) is imposed.  They are obtained from the kinematic basis
    fields by inverting the basis-resultant matrix R:

        gamma = inv(R) @ e_k
        eps_k = Sum_i gamma[i] * basis_eps[i]
        sig_k = Sum_i gamma[i] * basis_sig[i]

    where R is the 6x6 basis-resultant matrix stored on the SectionResult.
    (inv(K) is the wrong weighting: the chain basis amplitudes are not
    generalised strains -- see solver.py.)

    The integrated resultants of each of the six recovered fields must
    recover the corresponding unit load vector (the single strongest
    verification of the whole pipeline).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    epsilon: np.ndarray  # (6, n_cells, 6) Voigt strain (engineering shears)
    sigma: np.ndarray  # (6, n_cells, 6) Voigt stress
    cell_areas: np.ndarray  # (n_cells,)


def recover_strains(result: SectionResult) -> StrainField:
    """Recover per-cell Voigt strain / stress for the 6 kinematic basis fields.

    Returns the six fields whose total Voigt strains ``eps_total^(i)`` are the
    ones used internally by the solver to build the resultant matrix R (its
    columns) and the energy matrix that together produce K.  These correspond
    to the unit generalised-strain assumptions listed in theory.md.

    For the actual strain and stress distributions under the six *applied*
    unit load cases (Fx=1, Fy=1, Fz=1, Mx=1, My=1, Mz=1) use
    ``recover_unit_load_strains`` instead.

    Dispatches on ``result.backend``: mfem results route to the mfem
    backend's own recovery (same contract, same cell ordering semantics).
    """
    if getattr(result, "backend", "fenicsx") == "mfem":
        from .backends import mfem as _mfem_backend

        return _mfem_backend.recover_strains(result)

    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_vector

    if result.u_solutions is None or result.C_func is None or result.mesh is None:
        msg = "result lacks dolfinx state (was it constructed manually?)"
        raise ValueError(msg)

    mesh = result.mesh
    Q_func = result.C_func
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cell_areas = _cell_areas(mesh)

    eps = np.zeros((6, n_cells, 6))
    sig = np.zeros((6, n_cells, 6))

    x = ufl.SpatialCoordinate(mesh)
    DG0 = fem.functionspace(mesh, ("DG", 0))
    v = ufl.TestFunction(DG0)
    C_arr = Q_func.x.array.reshape(n_cells, 6, 6)

    primary = result.u_solutions
    for i in ALL_MODES:
        if primary[i] is None:
            continue
        d0 = d0_kinematic(i, x)
        if i in STAGE2_MODES:
            # shear: d_1 is the corresponding bending warping
            bending_idx = 4 if i == 0 else 3
            d1 = primary[bending_idx]
            d2 = primary[i]
        else:
            d1 = primary[i]
            d2 = None
        eps_total = total_strain_voigt(i, d0, d1, d2)

        for k_voigt in range(6):
            f = fem.form(v * eps_total[k_voigt] * ufl.dx)
            b = assemble_vector(f)
            b.assemble()
            eps[i, :, k_voigt] = b.array.copy() / cell_areas

        for k in range(n_cells):
            sig[i, k, :] = C_arr[k] @ eps[i, k, :]

    return StrainField(epsilon=eps, sigma=sig, cell_areas=cell_areas)


def _cell_areas(mesh: Any) -> np.ndarray:
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_vector

    DG0 = fem.functionspace(mesh, ("DG", 0))
    v = ufl.TestFunction(DG0)
    f = fem.form(v * ufl.dx)
    b = assemble_vector(f)
    b.assemble()
    return b.array.copy()


def recover_unit_load_strains(result: SectionResult) -> UnitLoadStrainField:
    """Recover per-cell Voigt strain / stress for the 6 applied unit load cases.

    For each standard load case k (Fx=1, Fy=1, Fz=1, Mx=1, My=1, Mz=1 applied
    at the section origin) the basis-field amplitudes are

        gamma = inv(R) @ e_k

    where R is the 6x6 basis-resultant matrix stored on the SectionResult
    (R[a, i] = a-th generalised-force resultant of the i-th chain-basis
    strain field) and e_k is the k-th unit vector.  The strain and stress
    fields are then the linear combination of the six kinematic basis fields:

        eps_k = Sum_i gamma[i] * basis_eps[i]
        sig_k = Sum_i gamma[i] * basis_sig[i]

    The integrated resultants of each of the six recovered fields must
    reproduce the corresponding unit load vector (within numerical tolerance).
    This is the single strongest verification of the whole pipeline.

    The returned arrays have shape (6, n_cells, 6) with the same Voigt
    convention and cell ordering as ``recover_strains``.
    """
    if (
        result.R is None
        or result.u_solutions is None
        or result.C_func is None
        or result.mesh is None
    ):
        msg = "result lacks dolfinx state or basis-resultant R (was it constructed manually?)"
        raise ValueError(msg)

    basis = recover_strains(result)
    eps_b = basis.epsilon
    sig_b = basis.sigma
    areas = basis.cell_areas

    Gamma = np.linalg.solve(result.R, np.eye(6))  # column k = inv(R) @ e_k
    eps_out = np.einsum("ki,icv->kcv", Gamma.T, eps_b)
    sig_out = np.einsum("ki,icv->kcv", Gamma.T, sig_b)

    return UnitLoadStrainField(epsilon=eps_out, sigma=sig_out, cell_areas=areas)
