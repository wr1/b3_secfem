"""Fenicsx (dolfinx) cross-section chain solve.

Reference implementation of the Morandini two-stage chain. Dispatched via
``backends.get_backend("fenicsx")``. See package ``solver`` module for the
public ``solve`` dispatch and mode constants.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...config import SectionInput
from ...kinematics import voigt_strain
from ...inertia import assemble_mass
from ...mesh import read_xdmf
from ...post import compute_centres
from ...result import SectionResult
from ..common import ALL_MODES, STAGE1_MODES, STAGE2_MODES, per_cell_arrays
from .forms import (
    assumed_inplane_shear_voigt,
    chain_rhs_stage1,
    chain_rhs_stage2,
    d0_kinematic,
    inplane_shear_rhs_linear,
    stiffness_bilinear,
    total_strain_voigt,
)
from .spaces import (
    fill_per_cell_density,
    fill_per_cell_stiffness,
    make_density_space,
    make_displacement_space,
    make_stiffness_space,
)

log = logging.getLogger(__name__)


def solve(inp: SectionInput) -> SectionResult:
    """Run the cross-section chain solve and return K, M, centres."""
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    mesh, cell_tags = _load_mesh(inp)
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local

    C_per_cell, rho_per_cell = per_cell_arrays(inp, n_cells, cell_tags)

    if inp.per_cell_material is not None:
        # Remap per-cell arrays from INPUT (spec) cell ordering to dolfinx
        # internal cell ordering. dolfinx.mesh.create_mesh renumbers cells
        # for performance; without this remap, rho_per_cell[k] (carrying the
        # density of input-tri-k) would land on dolfinx-cell-k which is in
        # general a DIFFERENT geometric cell. Same shuffling on C_per_cell
        # corrupts the stiffness assembly. Verified empirically: on a 261-
        # cell sec_14 mesh, dolfinx renumbers cell 0 -> input-tri 247.
        # Without this remap, secfem's M[0,0] over-counts by 10.5% and K
        # diagonal disagrees with ANBA on multi-material sections.
        oci = np.asarray(mesh.topology.original_cell_index)
        if oci.shape == (n_cells,) and not np.array_equal(oci, np.arange(n_cells)):
            rho_per_cell = rho_per_cell[oci]
            C_per_cell = C_per_cell[oci]

    V = make_displacement_space(mesh, degree=inp.degree)
    Q = make_stiffness_space(mesh)
    R_space = make_density_space(mesh)

    C_func = fem.Function(Q, name="C_bar")
    fill_per_cell_stiffness(C_func, C_per_cell)

    rho_func = fem.Function(R_space, name="rho")
    fill_per_cell_density(rho_func, rho_per_cell)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_form = fem.form(stiffness_bilinear(C_func, u, v))

    A = assemble_matrix(a_form)
    A.assemble()

    nullspace, _ns_basis = _build_nullspace(V)
    A.setNullSpace(nullspace)
    A.setNearNullSpace(nullspace)

    ksp = _make_ksp(A, mesh.comm, inp.linear_solver)

    x = ufl.SpatialCoordinate(mesh)

    d0_fields: dict[int, Any] = {i: d0_kinematic(i, x) for i in ALL_MODES}
    d1_fields: dict[int, Any] = {}
    d2_fields: dict[int, Any] = {}

    # Stage 1: solve d_1 for axial / bending / twist.
    for i in STAGE1_MODES:
        rhs = chain_rhs_stage1(C_func, v, d0_fields[i])
        d1_fields[i] = _solve_with_nullspace(
            V, ksp, fem.form(rhs), nullspace, name=f"d1_{i}"
        )

    # Shear chains share d_1 with the corresponding bending chain
    d1_fields[0] = d1_fields[4]  # Vx chain uses M_y bending warping
    d1_fields[1] = d1_fields[3]  # Vy chain uses M_x bending warping

    # Stage 2: solve d_2 for transverse shear.
    for i in STAGE2_MODES:
        rhs = chain_rhs_stage2(C_func, v, d0_fields[i], d1_fields[i])
        d2_fields[i] = _solve_with_nullspace(
            V, ksp, fem.form(rhs), nullspace, name=f"d2_{i}"
        )

    # 7th cell problem: section-averaged in-plane shear stiffness.
    # NOT a beam-level DOF -- computed and returned separately.
    #
    # The naive cell problem with only the rigid-body nullspace would give
    # K = 0 because the warping can fully absorb the assumed gamma_xy via
    # an in-plane field of the form (alpha*y, alpha*x, 0). To extract a
    # meaningful section-level K_xy we constrain the warping's spatial
    # average of gamma_xy to be zero, leaving only local Poisson-style
    # corrections. Implemented as a post-projection: solve, then subtract
    # the (y, x, 0) component such that mean(gamma_xy(w)) = 0.
    rhs_xy = inplane_shear_rhs_linear(C_func, v)
    w_xy_raw = _solve_with_nullspace(
        V, ksp, fem.form(rhs_xy), nullspace, name="w_inplane_shear"
    )
    A_total = float(fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx)))
    gamma_xy_raw = voigt_strain(w_xy_raw)[5]
    mean_gamma_xy_raw = (
        float(fem.assemble_scalar(fem.form(gamma_xy_raw * ufl.dx))) / A_total
    )
    # phi_field = (y, x, 0) gives gamma_xy = 2 uniformly; subtract beta*phi
    # so that mean(gamma_xy(w_xy_raw - beta*phi)) = 0.
    beta = mean_gamma_xy_raw / 2.0
    w_xy = fem.Function(V, name="w_inplane_shear")
    w_xy.x.array[:] = w_xy_raw.x.array
    coords = V.tabulate_dof_coordinates()
    bs = V.dofmap.index_map_bs
    arr = w_xy.x.array.reshape(-1, bs)
    arr[:, 0] -= beta * coords[:, 1]  # subtract beta * y from u_x
    arr[:, 1] -= beta * coords[:, 0]  # subtract beta * x from u_y
    w_xy.x.scatter_forward()

    eps_xy_total = assumed_inplane_shear_voigt() + voigt_strain(w_xy)
    K_xy_form = ufl.dot(eps_xy_total, ufl.dot(C_func, eps_xy_total)) * ufl.dx
    K_section_xy = float(fem.assemble_scalar(fem.form(K_xy_form)))

    # Build total Voigt strain field for each mode.
    eps_totals = {
        i: total_strain_voigt(i, d0_fields[i], d1_fields[i], d2_fields.get(i))
        for i in ALL_MODES
    }

    # Resultant matrix R[a, i] = a-th generalised-force resultant of mode i.
    R_mat = _assemble_resultants(C_func, mesh, eps_totals)

    # Energy matrix S[i, j].
    S_mat = _assemble_energy_matrix(C_func, eps_totals)

    # K = R · S^{-1} · R^T.
    K = R_mat @ np.linalg.solve(S_mat, R_mat.T)
    K = 0.5 * (K + K.T)

    M = assemble_mass(mesh, rho_func)
    detK = np.linalg.det(K)
    S_compliance = np.linalg.inv(K) if abs(detK) > 1e-30 else np.full_like(K, np.nan)
    centres = compute_centres(mesh, C_func, rho_func, K, M)

    # Pack the warping fields for downstream recovery: u_solutions[i] is the
    # primary "warping" Function for mode i (d_2 for shear, d_1 otherwise).
    u_solutions = []
    for i in ALL_MODES:
        u_solutions.append(d2_fields[i] if i in STAGE2_MODES else d1_fields[i])

    return SectionResult(
        K=K,
        M=M,
        S=S_compliance,
        R=R_mat,
        shear_center=centres["shear"],
        tension_center=centres["tension"],
        elastic_center=centres["elastic"],
        mass_center=centres["mass"],
        K_section_xy=K_section_xy,
        u_solutions=u_solutions,
        inplane_shear_warping=w_xy,
        C_func=C_func,
        mesh=mesh,
    )


def _make_ksp(A: Any, comm: Any, linear_solver: str) -> Any:
    """Build a PETSc KSP for the singular in-plane operator E.

    ``linear_solver`` (from ``SectionInput``):
      - ``gamg``: CG + smoothed-aggregation GAMG (good for large systems;
        setup-heavy on medium invsec sections).
      - ``lu``:   PREONLY + LU with a small nonzero shift so the 4-D rigid
        kernel does not make the factor singular. Fast factor + apply for
        medium 2D sizes; preferred when many independent medium jobs run
        under a wall-clock budget.
      - ``ilu``:  CG + ILU(0). Cheap setup, more iterations than GAMG on
        large meshes; a middle ground for medium sections without a full
        direct factor.
    """
    from petsc4py import PETSc

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    pc = ksp.getPC()
    if linear_solver == "lu":
        # Direct factor. E is SPSD with a 4-D kernel; a tiny diagonal shift
        # makes the factor well-defined while nullspace.remove keeps RHS
        # orthogonal to ker(E), so the solution stays in the physical range.
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc.setType(PETSc.PC.Type.LU)
        try:
            pc.setFactorSolverType("mumps")
        except Exception:  # mumps not built into this PETSc -- use default LU
            pass
        pc.setFactorShift(PETSc.Mat.FactorShiftType.NONZERO, 1e-10)
    elif linear_solver == "ilu":
        ksp.setType(PETSc.KSP.Type.CG)
        pc.setType(PETSc.PC.Type.ILU)
        ksp.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
    else:  # gamg (default)
        ksp.setType(PETSc.KSP.Type.CG)
        pc.setType(PETSc.PC.Type.GAMG)
        ksp.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
    return ksp


def _solve_with_nullspace(
    V: Any, ksp: Any, L_form: Any, nullspace: Any, name: str
) -> Any:
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_vector
    from petsc4py import PETSc

    b = assemble_vector(L_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    nullspace.remove(b)
    f = fem.Function(V, name=name)
    ksp.solve(b, f.x.petsc_vec)
    f.x.scatter_forward()
    return f


def _assemble_resultants(
    C_func: Any, mesh: Any, eps_totals: dict[int, Any]
) -> np.ndarray:
    """R[a, i] = a-th generalised-force resultant of eps_total^(i).

    Resultant ordering matches K: [Vx, Vy, Fz, Mx, My, Mz].
    Sigma = C @ eps_total in Voigt; recall sigma[2] = sigma_zz, sigma[3] = sigma_yz,
    sigma[4] = sigma_xz, sigma[5] = sigma_xy.

    On the section z = 0:
        Vx = int sigma_xz dA
        Vy = int sigma_yz dA
        Fz = int sigma_zz dA
        Mx = int  y * sigma_zz dA
        My = int -x * sigma_zz dA           (right-hand rule about y)
        Mz = int (x * sigma_yz - y * sigma_xz) dA
    """
    import ufl
    from dolfinx import fem

    x = ufl.SpatialCoordinate(mesh)
    R = np.zeros((6, 6))
    for i in range(6):
        sigma = ufl.dot(C_func, eps_totals[i])
        forms = [
            sigma[4],  # Vx
            sigma[3],  # Vy
            sigma[2],  # Fz
            x[1] * sigma[2],  # Mx
            -x[0] * sigma[2],  # My
            x[0] * sigma[3] - x[1] * sigma[4],  # Mz
        ]
        for a, integrand in enumerate(forms):
            R[a, i] = float(fem.assemble_scalar(fem.form(integrand * ufl.dx)))
    return R


def _assemble_energy_matrix(C_func: Any, eps_totals: dict[int, Any]) -> np.ndarray:
    """S[i, j] = int_Omega eps_total^(i)^T C eps_total^(j) dA."""
    import ufl
    from dolfinx import fem

    S = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            integrand = ufl.dot(eps_totals[i], ufl.dot(C_func, eps_totals[j])) * ufl.dx
            S[i, j] = float(fem.assemble_scalar(fem.form(integrand)))
    return 0.5 * (S + S.T)


def _load_mesh(inp: SectionInput):
    path = Path(inp.mesh_path)
    suffix = path.suffix.lower()
    if suffix == ".xdmf":
        return read_xdmf(path)
    if suffix == ".vtu":
        from ...mesh import from_gxbeam_vtu

        info = from_gxbeam_vtu(path)
        return info["mesh"], None
    msg = f"unsupported mesh extension {suffix}; use .xdmf or .vtu"
    raise ValueError(msg)


def _build_nullspace(V: Any):
    """4-D PETSc null space: translations in x/y/z and rotation about z."""
    from dolfinx import fem, la
    from petsc4py import PETSc

    coords = V.tabulate_dof_coordinates()
    bs = V.dofmap.index_map_bs

    funcs = []
    for mode in range(4):
        f = fem.Function(V)
        arr = np.zeros((coords.shape[0], bs))
        if mode == 0:
            arr[:, 0] = 1.0
        elif mode == 1:
            arr[:, 1] = 1.0
        elif mode == 2:
            arr[:, 2] = 1.0
        elif mode == 3:
            arr[:, 0] = -coords[:, 1]
            arr[:, 1] = coords[:, 0]
        f.x.array[:] = arr.reshape(-1)
        f.x.scatter_forward()
        funcs.append(f)

    la.orthonormalize([f.x for f in funcs])
    basis_vecs = [f.x.petsc_vec for f in funcs]
    nullspace = PETSc.NullSpace().create(vectors=basis_vecs, comm=V.mesh.comm)
    return nullspace, funcs
