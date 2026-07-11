"""Fenicsx (dolfinx) backend for b3_secfem.

During the initial backend introduction this is a thin adapter around the
reference implementation that still lives in solver._fenicsx_solve.
Later the full UFL/PETSc forms, spaces, recovery, etc. will live under
this module (or a fenicsx/ subpackage) so that importing b3_secfem
with only the mfem backend installed does not pull in dolfinx symbols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from ..config import SectionInput
    from ..result import SectionResult


def solve(inp: "SectionInput") -> "SectionResult":
    """Run the solve using the fenicsx (dolfinx) engine."""
    # Delegate to the reference implementation (transition phase).
    # After the full move this will contain (or import from .fenicsx_impl) the logic.
    from ..solver import _fenicsx_solve

    return _fenicsx_solve(inp)


def assemble_stiffness_matrix(
    C_per_cell: np.ndarray,
    mesh: Any,
    degree: int = 2,
) -> sp.csr_matrix:
    """Assemble the core in-plane stiffness operator E.

    E(u, v) = ∫ voigt_strain(v)^T C(x,y) voigt_strain(u) dA

    This is the main bilinear form used for all warping solves.

    Parameters
    ----------
    C_per_cell : (n_cells, 6, 6) array
        Rotated 6x6 stiffness in each cell (Voigt ordering).
    mesh : dolfinx.mesh.Mesh
        The dolfinx mesh (2D, triangles or quads).
    degree : int
        Polynomial degree for the vector displacement space (default 2).

    Returns
    -------
    scipy.sparse.csr_matrix
        The assembled global stiffness matrix (symmetric positive semi-definite,
        with 4-dimensional kernel of rigid modes).
    """
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    from ..forms import stiffness_bilinear
    from ..spaces import (
        fill_per_cell_stiffness,
        make_displacement_space,
        make_stiffness_space,
    )

    # Ensure connectivity (sometimes needed when calling helpers directly)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

    Q = make_stiffness_space(mesh)
    C_func = fem.Function(Q, name="C_bar")
    fill_per_cell_stiffness(C_func, C_per_cell)

    V = make_displacement_space(mesh, degree=degree)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_form = fem.form(stiffness_bilinear(C_func, u, v))
    A_petsc = assemble_matrix(a_form)
    A_petsc.assemble()

    # Convert PETSc Mat to scipy CSR
    ai, aj, av = A_petsc.getValuesCSR()
    A = sp.csr_matrix((av, aj, ai), shape=A_petsc.getSize())

    A_petsc.destroy()
    return A
