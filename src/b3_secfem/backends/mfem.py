"""MFEM (PyMFEM serial) backend for b3_secfem.

This provides an independent implementation of the exact same Morandini
two-stage chain formulation using MFEM for mesh, FE spaces, assembly, and
quadrature, with scipy for the singular linear systems (bordered KKT) + the
4-D rigid-body null space.

Status (2026-05): **Full solve implemented.** A single custom integrator
(`_VoigtFormIntegrator`) assembles the three section operators E (in-plane
stiffness), Cmat (xy-z coupling) and Mmat (pure-z); the two-stage chain then
reduces to matrix-vector products, K = R S^{-1} Rᵀ follows from a quadrature
pass, and M/centres/K_xy mirror the fenicsx closed forms. ``solve()`` returns a
complete ``SectionResult``. Validated against the fenicsx backend via
``b3_secfem.bench``: K/M/centres/K_xy agree to ~1e-11 relative on iso and
orthotropic rectangles (see tests/test_backend_comparison.py).

Not yet ported: strain/stress field recovery (``recover_strains``) and the
input-cell-order remap needed for ``per_cell_material`` on dolfinx-renumbered
meshes (region-tagged and uniform sections are fine).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..config import SectionInput
from ..result import SectionResult
from .common import ALL_MODES, STAGE1_MODES, STAGE2_MODES

log = logging.getLogger(__name__)

try:
    import mfem.ser as mfem
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _MFEM_AVAILABLE = True
except ImportError:
    _MFEM_AVAILABLE = False


def _require_mfem():
    if not _MFEM_AVAILABLE:
        msg = (
            "mfem backend requested but PyMFEM (and/or scipy) not installed. "
            "Install with: pip install 'b3_secfem[mfem]' or conda install -c conda-forge mfem scipy"
        )
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Mesh construction (neutral from meshio + mfem.Mesh)
# ---------------------------------------------------------------------------

def _load_mfem_mesh(inp: SectionInput) -> tuple[Any, np.ndarray | None, int]:
    """Load mesh via meshio and build an mfem.Mesh preserving input cell order.

    Returns (mfem_mesh, tags_per_cell or None, n_cells).
    Only quads are fully supported today (the dominant case in examples
    and cross-check ladder). Tri support is straightforward to add.
    """
    _require_mfem()
    import meshio

    path = Path(inp.mesh_path)
    m = meshio.read(str(path))

    # Find first 2D cell block
    cells = None
    ctype = None
    for cb in m.cells:
        if cb.type in ("quad", "quadrilateral"):
            cells = cb.data.astype(np.int64)
            ctype = "quad"
            break
        if cb.type in ("triangle", "tri"):
            cells = cb.data.astype(np.int64)
            ctype = "tri"
            break
    if cells is None:
        raise ValueError(f"no quad or tri cells found in {path}")

    nodes = m.points[:, :2].astype(np.float64)
    n_cells = cells.shape[0]

    # CCW validation + fix (same shoelace as the rest of the package)
    p = nodes[cells]
    if ctype == "quad":
        sa = (
            p[:, 0, 0] * p[:, 1, 1] - p[:, 1, 0] * p[:, 0, 1]
            + p[:, 1, 0] * p[:, 2, 1] - p[:, 2, 0] * p[:, 1, 1]
            + p[:, 2, 0] * p[:, 3, 1] - p[:, 3, 0] * p[:, 2, 1]
            + p[:, 3, 0] * p[:, 0, 1] - p[:, 0, 0] * p[:, 3, 1]
        )
        flip = sa < 0
        if flip.any():
            cells[flip] = cells[flip][:, [0, 3, 2, 1]]
        # MFEM quad node ordering is usually CCW BL-BR-TR-TL or similar;
        # we keep the CCW order from the file.
    else:
        # tri shoelace (simpler)
        sa = (
            p[:, 0, 0] * p[:, 1, 1] - p[:, 1, 0] * p[:, 0, 1]
            + p[:, 1, 0] * p[:, 2, 1] - p[:, 2, 0] * p[:, 1, 1]
            + p[:, 2, 0] * p[:, 0, 1] - p[:, 0, 0] * p[:, 2, 1]
        )
        flip = sa < 0
        if flip.any():
            cells[flip] = cells[flip][:, [0, 2, 1]]

    # Build mfem.Mesh from vertices + elements using the proper AddQuad/AddTriangle API
    nv = nodes.shape[0]
    ne = n_cells
    mesh = mfem.Mesh(2, nv, ne, 0, 2)  # dim, nvert, nelem, nbdr, spaceDim

    for i in range(nv):
        mesh.AddVertex(nodes[i])

    if ctype == "quad":
        for i in range(ne):
            mesh.AddQuad(int(cells[i,0]), int(cells[i,1]), int(cells[i,2]), int(cells[i,3]))
    else:
        for i in range(ne):
            mesh.AddTriangle(int(cells[i,0]), int(cells[i,1]), int(cells[i,2]))

    mesh.FinalizeTopology()
    mesh.Finalize()

    # Per-cell tags (region or default 1)
    tags = None
    cell_data = m.cell_data_dict if hasattr(m, "cell_data_dict") else {}
    # Try common names used by our VTU/XDMF writers
    for key in ("cell_tags", "tags", "region", "mat"):
        if key in cell_data:
            arr = cell_data[key]
            if isinstance(arr, dict):
                arr = arr.get(ctype) or next(iter(arr.values()), None)
            if arr is not None:
                tags = np.asarray(arr, dtype=np.int32)[:n_cells]
                break

    return mesh, tags, n_cells


def _voigt_strain_from_dshape(dshape_np: np.ndarray) -> np.ndarray:
    """Build the (6, 3*ndof) strain-displacement matrix B for one quad point.

    ``dshape_np`` is the (ndof, 2) array of physical shape-function gradients.
    The Voigt strain is [e_xx, e_yy, e_zz, 2 e_yz, 2 e_xz, 2 e_xy] -- i.e. the
    in-plane part ε_xy(u) of the kinematics (e_zz is identically 0 here since
    there is no z-derivative of the 2D field).

    Column layout is **component-major** to match how MFEM lays out element
    vdofs: ``mfem.FiniteElementSpace.GetElementVDofs`` returns all component-0
    dofs, then all component-1, then all component-2, independently of the
    global ``Ordering``. So the column for (node k, component c) is ``c*ndof+k``.
    Using the interleaved ``3*k+c`` layout instead silently scatters every
    element matrix into the wrong global slots: the diagonal (hence the trace)
    survives, but the off-diagonal coupling -- and therefore the spectrum and
    the 4-D rigid-body null space -- is corrupted. This is the single source of
    truth shared by the stiffness and RHS integrators so they cannot drift.
    """
    ndof = dshape_np.shape[0]
    B = np.zeros((6, 3 * ndof))
    ux, uy, uz = 0, ndof, 2 * ndof
    for k in range(ndof):
        B[0, ux + k] = dshape_np[k, 0]   # e_xx   from ux,x
        B[5, ux + k] = dshape_np[k, 1]   # 2 e_xy from ux,y
        B[1, uy + k] = dshape_np[k, 1]   # e_yy   from uy,y
        B[5, uy + k] = dshape_np[k, 0]   # 2 e_xy from uy,x
        B[3, uz + k] = dshape_np[k, 1]   # 2 e_yz from uz,y
        B[4, uz + k] = dshape_np[k, 0]   # 2 e_xz from uz,x
    return B


def _voigt_epsz_from_shape(shape_np: np.ndarray) -> np.ndarray:
    """Build the (6, 3*ndof) operator Bz for the z-derivative strain eps_z.

    For a field expanded as u = ... + z*v(x,y) + ..., the z-independent strain
    contribution is eps_z(v) = [0, 0, v_z, v_y, v_x, 0] (engineering shears) --
    it uses the *values* of v at the point, not its gradient. So Bz maps the
    nodal dofs through the shape-function values ``shape_np`` (length ndof).
    Column layout is component-major to match :func:`_voigt_strain_from_dshape`
    and MFEM's element vdof ordering.
    """
    ndof = shape_np.shape[0]
    Bz = np.zeros((6, 3 * ndof))
    ux, uy, uz = 0, ndof, 2 * ndof
    for k in range(ndof):
        Bz[2, uz + k] = shape_np[k]   # eps_zz   = v_z
        Bz[3, uy + k] = shape_np[k]   # 2 eps_yz = v_y
        Bz[4, ux + k] = shape_np[k]   # 2 eps_xz = v_x
    return Bz


# ---------------------------------------------------------------------------
# Custom integrators (the heart of the mfem backend)
# These are only defined when mfem is actually importable.
# ---------------------------------------------------------------------------

def _quad_order(el) -> int:
    """Integration-rule order used consistently across every mfem-backend form."""
    return 2 * el.GetOrder() + 3


if _MFEM_AVAILABLE:
    class _VoigtFormIntegrator(mfem.PyBilinearFormIntegrator):
        """Generic ∫ B_test(v)^T C(x) B_trial(u) dA over the section.

        ``test_kind`` / ``trial_kind`` select the Voigt strain operator:
          - ``"xy"`` -> in-plane strain eps_xy(.)  (gradient-based, B)
          - ``"z"``  -> z-derivative strain eps_z(.) (value-based, Bz)

        This single integrator yields every operator the chain solve needs:
          E    = ("xy", "xy")   in-plane stiffness
          Cmat = ("xy", "z")    coupling   C(u, v) = ∫ eps_xy(v)^T C eps_z(u)
          Mmat = ("z",  "z")    pure-z     M(u, v) = ∫ eps_z(v)^T  C eps_z(u)

        With these, the Morandini chain RHS reduce to matrix-vector products
        (see solve()): E d1 = -Cmat d0, and E d2 = Mmat d0 - (Cmat - Cmatᵀ) d1.
        """

        def __init__(self, C_per_cell: np.ndarray, test_kind: str, trial_kind: str):
            super().__init__()
            self.C = np.asarray(C_per_cell)
            self.test_kind = test_kind
            self.trial_kind = trial_kind

        def _operator(self, el, trans, ip, kind, dref, dphys, shp):
            if kind == "xy":
                el.CalcDShape(ip, dref)
                mfem.Mult(dref, trans.InverseJacobian(), dphys)
                return _voigt_strain_from_dshape(dphys.GetDataArray().reshape(el.GetDof(), 2))
            el.CalcShape(ip, shp)
            return _voigt_epsz_from_shape(shp.GetDataArray())

        def AssembleElementMatrix(self, el, trans, elmat):
            ndof = el.GetDof()
            vdim = 3
            elmat.SetSize(ndof * vdim, ndof * vdim)
            elmat.Assign(0.0)

            # Material id is on the ElementTransformation (set from the mesh
            # attribute = cell+1), NOT the FiniteElement -- the latter has no
            # GetAttribute, so reading it there would silently use C[0] for
            # every cell (a multi-material bug).
            eid = max(0, trans.Attribute - 1)
            if eid >= len(self.C):
                eid = 0
            C_local = self.C[eid]

            ir = mfem.IntRules.Get(el.GetGeomType(), _quad_order(el))
            dref = mfem.DenseMatrix(ndof, 2)
            dphys = mfem.DenseMatrix(ndof, 2)
            shp = mfem.Vector(ndof)
            acc = np.zeros((vdim * ndof, vdim * ndof))

            for q in range(ir.GetNPoints()):
                ip = ir.IntPoint(q)
                trans.SetIntPoint(ip)
                w = trans.Weight() * ip.weight
                Bt = self._operator(el, trans, ip, self.test_kind, dref, dphys, shp)
                Bu = self._operator(el, trans, ip, self.trial_kind, dref, dphys, shp)
                acc += (Bt.T @ (C_local @ Bu)) * w

            for r in range(vdim * ndof):
                for c in range(vdim * ndof):
                    elmat[r, c] = acc[r, c]

    class StiffnessIntegrator(_VoigtFormIntegrator):
        """In-plane stiffness ∫ eps_xy(v)^T C eps_xy(u) dA (the E operator)."""

        def __init__(self, C_per_cell: np.ndarray):
            super().__init__(C_per_cell, "xy", "xy")

else:
    _VoigtFormIntegrator = None  # type: ignore[assignment]
    StiffnessIntegrator = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# High-level solve (orchestration re-uses the pure-Python parts of the formulation)
# ---------------------------------------------------------------------------

def solve(inp: SectionInput) -> SectionResult:
    """MFEM backend: full Morandini two-stage chain solve (K, M, centres).

    Mirrors the fenicsx reference (``solver._fenicsx_solve``) on the mfem engine.
    A single custom integrator assembles the three section operators

        E    = ∫ eps_xy(v)^T C eps_xy(u) dA   (in-plane stiffness)
        Cmat = ∫ eps_xy(v)^T C eps_z(u)  dA   (z-coupling)
        Mmat = ∫ eps_z(v)^T  C eps_z(u)  dA   (pure-z)

    so the chain reduces to matrix-vector products:

        Stage 1 (Fz, Mx, My, Mz):  E d1 = -Cmat d0      (eps_xy(d0)=0)
        Stage 2 (Vx, Vy):          E d2 = Mmat d0 - (Cmat - Cmatᵀ) d1

    The singular systems are solved with a bordered KKT factorisation (the
    direct analogue of the fenicsx PETSc null-space projection). Then
    ``K = R S^{-1} Rᵀ`` from the resultant/energy matrices. Validated against
    fenicsx via ``b3_secfem.bench`` (K/M/centres agree to ~1e-8 relative).
    """
    _require_mfem()
    from types import SimpleNamespace

    from ..solver import _per_cell_arrays

    mesh, tags, n_cells = _load_mfem_mesh(inp)
    for e in range(n_cells):
        mesh.GetElement(e).SetAttribute(e + 1)
    mesh.Finalize()

    # Per-cell C/rho in mfem element order (== input cell order, since
    # _load_mfem_mesh preserves it -- no dolfinx-style renumbering needed).
    ct = (
        SimpleNamespace(indices=np.arange(n_cells), values=tags)
        if tags is not None else None
    )
    C_per_cell, rho_per_cell = _per_cell_arrays(inp, n_cells, ct)

    fec = mfem.H1_FECollection(inp.degree, mesh.Dimension())
    fes = mfem.FiniteElementSpace(mesh, fec, 3)

    # Section operators (one integrator, three operator pairs).
    E = _assemble_voigt_form(C_per_cell, fes, "xy", "xy").tocsc()
    Cmat = _assemble_voigt_form(C_per_cell, fes, "xy", "z").tocsc()
    Mmat = _assemble_voigt_form(C_per_cell, fes, "z", "z").tocsc()

    # Coordinates, rigid null space, single KKT factorisation reused below.
    _sfes, xs, ys = _scalar_dof_coords(mesh, inp.degree)
    nullvecs = _orthonormal_rigid(fes, xs, ys)
    ksolve = _make_kkt_solver(E, nullvecs)

    d0 = {i: _analytic_field_dofs(fes, xs, ys, _d0_field(i)) for i in ALL_MODES}

    # Stage 1: E d1 = -H d0 = -Cmat d0 (the Cmatᵀ term vanishes for rigid d0).
    d1: dict[int, np.ndarray] = {}
    for i in STAGE1_MODES:
        d1[i] = ksolve(-(Cmat @ d0[i]))
    d1[0] = d1[4]   # Vx chain reuses My bending warping
    d1[1] = d1[3]   # Vy chain reuses Mx bending warping

    # Stage 2: E d2 = Mmat d0 - (Cmat - Cmatᵀ) d1.
    d2: dict[int, np.ndarray] = {}
    for i in STAGE2_MODES:
        d2[i] = ksolve(Mmat @ d0[i] - (Cmat @ d1[i] - Cmat.T @ d1[i]))

    # Total-strain field specs per mode: eps_total = eps_z(a) + eps_xy(b).
    a_dofs = {0: d1[4], 1: d1[3], 2: d0[2], 3: d0[3], 4: d0[4], 5: d0[5]}
    b_dofs = {0: d2[0], 1: d2[1], 2: d1[2], 3: d1[3], 4: d1[4], 5: d1[5]}

    R_mat, S_mat = _assemble_R_S(mesh, fes, C_per_cell, a_dofs, b_dofs)
    K = R_mat @ np.linalg.solve(S_mat, R_mat.T)
    K = 0.5 * (K + K.T)

    mom = _section_integrals(mesh, fes, C_per_cell, rho_per_cell)
    M = _build_mass(mom)
    K_section_xy, w_xy = _inplane_shear(
        mesh, fes, C_per_cell, ksolve, xs, ys, mom["A"]
    )

    detK = np.linalg.det(K)
    S_comp = np.linalg.inv(K) if abs(detK) > 1e-30 else np.full((6, 6), np.nan)
    centres = _compute_centres_mfem(mom, K)

    u_solutions = [d2[i] if i in STAGE2_MODES else d1[i] for i in ALL_MODES]

    return SectionResult(
        K=K,
        M=M,
        S=S_comp,
        R=R_mat,
        shear_center=centres["shear"],
        tension_center=centres["tension"],
        elastic_center=centres["elastic"],
        mass_center=centres["mass"],
        K_section_xy=K_section_xy,
        u_solutions=u_solutions,
        inplane_shear_warping=w_xy,
        C_func=None,
        mesh=mesh,
        backend="mfem",
    )


def recover_strains(result: SectionResult):
    if getattr(result, "backend", "") != "mfem":
        raise ValueError("recover_strains called on non-mfem result")
    raise NotImplementedError(
        "mfem strain/stress field recovery is not ported yet (K/M/centres are). "
        "Use the fenicsx backend for recover_strains / recover_unit_load_strains."
    )


# (identical pattern for recover_unit_load_strains, the two plot helpers, etc.)


def assemble_stiffness_matrix(
    C_per_cell: np.ndarray,
    mesh: Any,
    degree: int = 2,
) -> "sp.csr_matrix":
    """Assemble the core in-plane stiffness operator E using the MFEM backend.

    This uses the custom StiffnessIntegrator and returns a scipy CSR matrix,
    making direct numeric comparison with the fenicsx backend trivial.
    """
    _require_mfem()

    n_cells = C_per_cell.shape[0]

    # Ensure element attributes are set to cell index + 1 (used by integrators)
    for i in range(n_cells):
        mesh.GetElement(i).SetAttribute(i + 1)
    mesh.Finalize()

    fec = mfem.H1_FECollection(degree, mesh.Dimension())
    # The global Ordering (byNODES/byVDIM) only permutes the global matrix; the
    # element-local vdof order MFEM scatters with is always component-major, so
    # the StiffnessIntegrator's B is built component-major to match (see
    # _voigt_strain_from_dshape). Default ordering is fine here.
    fes = mfem.FiniteElementSpace(mesh, fec, 3)

    a = mfem.BilinearForm(fes)
    a.AddDomainIntegrator(StiffnessIntegrator(C_per_cell))
    a.Assemble()
    a.Finalize()

    Asp = _mfem_spmat_to_scipy(a.SpMat())
    return Asp


# =============================================================================
# Internal helper functions for the mfem backend
# =============================================================================

def _mfem_spmat_to_scipy(spmat) -> "sp.csr_matrix":
    """Convert mfem SparseMatrix to scipy CSR (robust to Array wrappers)."""
    import mfem.ser as mfem
    # Get raw arrays - they may come back as mfem.Array
    I = spmat.GetIArray() if hasattr(spmat, "GetIArray") else spmat.GetI()
    J = spmat.GetJArray() if hasattr(spmat, "GetJArray") else spmat.GetJ()
    D = spmat.GetDataArray()

    # Convert mfem.Array / SWIG wrappers to an OWNED numpy 1-D array.
    # The copy is essential: GetIArray/GetJArray/GetDataArray return numpy
    # views into the C++ SparseMatrix buffers. That matrix is owned by the
    # BilinearForm, which is typically a local that gets garbage-collected
    # right after assembly -- freeing the buffers. Without copying, the scipy
    # matrix would alias freed memory and its indptr/indices would silently
    # corrupt (manifesting as a non-zero indptr[0] or garbage row pointers).
    def _to_np(a):
        if hasattr(a, "GetData"):
            return np.array(a.GetData())  # GetData() already returns a copy
        try:
            return np.array(np.asarray(a).ravel())  # np.array(...) forces a copy
        except Exception:
            return np.array(list(a), dtype=float).ravel()

    rows = _to_np(I).astype(np.int32, copy=False)
    cols = _to_np(J).astype(np.int32, copy=False)
    data = _to_np(D).astype(np.float64, copy=False)

    A = sp.csr_matrix((data, cols, rows), shape=(spmat.Height(), spmat.Width()))
    A.sum_duplicates()
    return A


# =============================================================================
# Full chain solve -- helper functions (mfem engine, validated vs fenicsx)
# =============================================================================

def _d0_field(mode: int):
    """Analytic rigid kinematic d0(x, y) -> (ux, uy, uz) for each chain mode.

    Matches forms.d0_kinematic: modes 0/4 bend about y (uz = x), 1/3 about x
    (uz = -y), 2 axial (uz = 1), 5 twist (-y, x, 0).
    """
    if mode in (0, 4):
        return lambda x, y: (0.0, 0.0, x)
    if mode in (1, 3):
        return lambda x, y: (0.0, 0.0, -y)
    if mode == 2:
        return lambda x, y: (0.0, 0.0, 1.0)
    if mode == 5:
        return lambda x, y: (-y, x, 0.0)
    msg = f"_d0_field: mode must be 0..5, got {mode}"
    raise ValueError(msg)


def _scalar_dof_coords(mesh, degree: int):
    """Return (sfes, xs, ys): a scalar H1 space + physical (x, y) of each scalar dof.

    The displacement space is the same scalar space with vdim=3, so scalar dof
    index s maps to displacement dofs via fes.DofToVDof(s, comp).
    """
    fec = mfem.H1_FECollection(degree, mesh.Dimension())
    sfes = mfem.FiniteElementSpace(mesh, fec, 1)
    ns = sfes.GetNDofs()
    xs = np.full(ns, np.nan)
    ys = np.full(ns, np.nan)
    for e in range(mesh.GetNE()):
        el = sfes.GetFE(e)
        tr = mesh.GetElementTransformation(e)
        nodes = el.GetNodes()
        dofs = sfes.GetElementDofs(e)
        for i in range(el.GetDof()):
            ip = nodes.IntPoint(i)
            tr.SetIntPoint(ip)
            pt = tr.Transform(ip)
            xs[dofs[i]] = pt[0]
            ys[dofs[i]] = pt[1]
    return sfes, xs, ys


def _analytic_field_dofs(fes, xs, ys, fn) -> np.ndarray:
    """Global dof vector of an analytic field fn(x, y) -> (ux, uy, uz).

    Exact for fields representable in the FE space (our d0 kinematics and rigid
    modes are degree <= 1, well within CG-2).
    """
    vec = np.zeros(fes.GetVSize())
    for s in range(len(xs)):
        ux, uy, uz = fn(xs[s], ys[s])
        vec[fes.DofToVDof(s, 0)] = ux
        vec[fes.DofToVDof(s, 1)] = uy
        vec[fes.DofToVDof(s, 2)] = uz
    return vec


def _orthonormal_rigid(fes, xs, ys) -> list:
    """The 4 rigid-body modes (tx, ty, tz, rot_z) spanning ker(E), orthonormalised."""
    raw = [
        _analytic_field_dofs(fes, xs, ys, lambda x, y: (1.0, 0.0, 0.0)),
        _analytic_field_dofs(fes, xs, ys, lambda x, y: (0.0, 1.0, 0.0)),
        _analytic_field_dofs(fes, xs, ys, lambda x, y: (0.0, 0.0, 1.0)),
        _analytic_field_dofs(fes, xs, ys, lambda x, y: (-y, x, 0.0)),
    ]
    Q, _ = np.linalg.qr(np.column_stack(raw))
    return [Q[:, k].copy() for k in range(Q.shape[1])]


def _assemble_voigt_form(C_per_cell, fes, test_kind: str, trial_kind: str):
    """Assemble ∫ B_test^T C B_trial dA as a scipy CSR via _VoigtFormIntegrator."""
    a = mfem.BilinearForm(fes)
    a.AddDomainIntegrator(_VoigtFormIntegrator(C_per_cell, test_kind, trial_kind))
    a.Assemble()
    a.Finalize()
    return _mfem_spmat_to_scipy(a.SpMat())


def _make_kkt_solver(E, nullvecs):
    """Factor the bordered KKT system [[E, N], [Nᵀ, 0]] once; return solve(b).

    N spans ker(E) (the rigid modes). The saddle-point system is nonsingular and
    yields the exact min-norm solution of E x = P_range(b) with Nᵀ x = 0 -- the
    direct analogue of the fenicsx PETSc null-space projection. The single
    factorisation is reused for every chain RHS.
    """
    n = E.shape[0]
    N = np.column_stack(nullvecs)
    k = N.shape[1]
    aug = sp.bmat([[E, sp.csr_matrix(N)], [sp.csr_matrix(N.T), None]], format="csc")
    lu = spla.splu(aug)

    def solve(b: np.ndarray) -> np.ndarray:
        rhs = np.concatenate([np.asarray(b, dtype=float), np.zeros(k)])
        return lu.solve(rhs)[:n]

    return solve


def _decode_vdofs(vdofs):
    """Split MFEM (possibly sign-encoded) vdofs into (index, sign) arrays.

    MFEM encodes an orientation sign flip as a negative vdof j -> -1-j. H1 nodal
    spaces on quad/tri meshes do not use this, but decoding is cheap and keeps the
    manual quadrature consistent with BilinearForm's signed scatter.
    """
    v = np.asarray(list(vdofs), dtype=np.int64)
    sign = np.where(v >= 0, 1.0, -1.0)
    idx = np.where(v >= 0, v, -1 - v)
    return idx, sign


def _iter_quad(mesh, fes, e):
    """Yield (w, x, y, dN(nd,2), N(nd), idx, sign) per quad point of element e.

    dN are physical shape gradients, N shape values, idx/sign the decoded global
    vdof map (component-major, matching _voigt_strain_from_dshape's column layout).
    """
    el = fes.GetFE(e)
    tr = mesh.GetElementTransformation(e)
    nd = el.GetDof()
    ir = mfem.IntRules.Get(el.GetGeomType(), _quad_order(el))
    dref = mfem.DenseMatrix(nd, 2)
    dphys = mfem.DenseMatrix(nd, 2)
    shp = mfem.Vector(nd)
    idx, sign = _decode_vdofs(fes.GetElementVDofs(e))
    for q in range(ir.GetNPoints()):
        ip = ir.IntPoint(q)
        tr.SetIntPoint(ip)
        el.CalcDShape(ip, dref)
        mfem.Mult(dref, tr.InverseJacobian(), dphys)
        el.CalcShape(ip, shp)
        pt = tr.Transform(ip)
        yield (
            tr.Weight() * ip.weight,
            pt[0], pt[1],
            dphys.GetDataArray().reshape(nd, 2).copy(),
            shp.GetDataArray().copy(),
            idx, sign,
        )


def _assemble_R_S(mesh, fes, C_per_cell, a_dofs, b_dofs):
    """Resultant R[a, i] and energy S[i, j] matrices by section quadrature.

    eps_total^(i) = eps_z(field a_i) + eps_xy(field b_i). Mirrors
    solver._assemble_resultants / _assemble_energy_matrix exactly (same Voigt
    indices and resultant integrands; resultant order [Vx, Vy, Fz, Mx, My, Mz]).
    """
    R = np.zeros((6, 6))
    S = np.zeros((6, 6))
    for e in range(mesh.GetNE()):
        C_local = C_per_cell[e]
        a_loc = [a_dofs[i] for i in range(6)]
        b_loc = [b_dofs[i] for i in range(6)]
        for w, x, y, dN, N, idx, sign in _iter_quad(mesh, fes, e):
            Bxy = _voigt_strain_from_dshape(dN)
            Bz = _voigt_epsz_from_shape(N)
            eps = np.empty((6, 6))
            for i in range(6):
                eps[:, i] = Bz @ (sign * a_loc[i][idx]) + Bxy @ (sign * b_loc[i][idx])
            sig = C_local @ eps                 # (voigt, mode)
            S += (eps.T @ sig) * w
            for i in range(6):
                si = sig[:, i]
                R[0, i] += si[4] * w
                R[1, i] += si[3] * w
                R[2, i] += si[2] * w
                R[3, i] += y * si[2] * w
                R[4, i] += -x * si[2] * w
                R[5, i] += (x * si[3] - y * si[4]) * w
    return R, 0.5 * (S + S.T)


def _section_integrals(mesh, fes, C_per_cell, rho_per_cell) -> dict:
    """Scalar section integrals for the mass matrix + centres (one quad pass)."""
    keys = ("A", "m", "mx", "my", "Ixx", "Iyy", "Ixy", "iE", "iEx", "iEy")
    acc = dict.fromkeys(keys, 0.0)
    for e in range(mesh.GetNE()):
        rho = float(rho_per_cell[e])
        E33 = float(C_per_cell[e][2, 2])
        for w, x, y, _dN, _N, _idx, _sign in _iter_quad(mesh, fes, e):
            acc["A"] += w
            acc["m"] += rho * w
            acc["mx"] += rho * x * w
            acc["my"] += rho * y * w
            acc["Ixx"] += rho * y * y * w
            acc["Iyy"] += rho * x * x * w
            acc["Ixy"] += rho * x * y * w
            acc["iE"] += E33 * w
            acc["iEx"] += E33 * x * w
            acc["iEy"] += E33 * y * w
    return acc


def _build_mass(mom: dict) -> np.ndarray:
    """6x6 mass matrix from section moments (matches inertia.assemble_mass)."""
    m, mx, my = mom["m"], mom["mx"], mom["my"]
    Ixx, Iyy, Ixy = mom["Ixx"], mom["Iyy"], mom["Ixy"]
    M = np.array([
        [ m,    0,    0,    0,     0,    -my],
        [ 0,    m,    0,    0,     0,     mx],
        [ 0,    0,    m,    my,   -mx,    0],
        [ 0,    0,    my,   Ixx,  -Ixy,   0],
        [ 0,    0,   -mx,  -Ixy,   Iyy,   0],
        [-my,   mx,   0,    0,     0,     Ixx + Iyy],
    ], dtype=float)
    return 0.5 * (M + M.T)


def _compute_centres_mfem(mom: dict, K: np.ndarray) -> dict:
    """Tension / mass / elastic / shear centres (matches post.compute_centres)."""
    iE = mom["iE"]
    tension = (mom["iEx"] / iE, mom["iEy"] / iE) if iE != 0 else (float("nan"),) * 2
    m = mom["m"]
    mass = (mom["mx"] / m, mom["my"] / m) if m > 0 else (float("nan"), float("nan"))
    shear = (float("nan"), float("nan"))
    try:
        S = np.linalg.inv(K)
        if np.isfinite(S[5, 5]) and abs(S[5, 5]) > 1e-30:
            shear = (-S[5, 1] / S[5, 5], S[5, 0] / S[5, 5])
    except np.linalg.LinAlgError:
        pass
    return {"tension": tension, "mass": mass, "elastic": mass, "shear": shear}


def _inplane_shear(mesh, fes, C_per_cell, ksolve, xs, ys, area: float):
    """7th cell problem: section-averaged in-plane shear stiffness K_xy.

    Mirrors solver._fenicsx_solve: solve E w = -∫ eps_xy(v)^T C eps_a, subtract the
    (y, x, 0) component so mean(gamma_xy) = 0, then
    K_xy = ∫ (eps_a + eps_xy(w))^T C (eps_a + eps_xy(w)). eps_a = (0,0,0,0,0,1).
    """
    eps_a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    g = np.zeros(fes.GetVSize())
    for e in range(mesh.GetNE()):
        f = C_per_cell[e] @ eps_a
        for w, _x, _y, dN, _N, idx, sign in _iter_quad(mesh, fes, e):
            contrib = (_voigt_strain_from_dshape(dN).T @ f) * w
            np.add.at(g, idx, sign * contrib)
    w_raw = ksolve(-g)

    int_gamma = 0.0
    for e in range(mesh.GetNE()):
        for w, _x, _y, dN, _N, idx, sign in _iter_quad(mesh, fes, e):
            int_gamma += (_voigt_strain_from_dshape(dN) @ (sign * w_raw[idx]))[5] * w
    beta = (int_gamma / area) / 2.0
    phi = _analytic_field_dofs(fes, xs, ys, lambda x, y: (y, x, 0.0))
    w_xy = w_raw - beta * phi

    K_xy = 0.0
    for e in range(mesh.GetNE()):
        C_local = C_per_cell[e]
        for w, _x, _y, dN, _N, idx, sign in _iter_quad(mesh, fes, e):
            eps_tot = eps_a + _voigt_strain_from_dshape(dN) @ (sign * w_xy[idx])
            K_xy += float(eps_tot @ (C_local @ eps_tot)) * w
    return K_xy, w_xy
