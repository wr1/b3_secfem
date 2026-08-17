"""b3_secfem — composite cross-section property solver.

Supports two backends (selected via SectionInput.backend or solve(..., backend=...)):
- "fenicsx" (default): dolfinx + UFL + PETSc (full featured, the original implementation)
- "mfem": PyMFEM serial (mfem.ser) — independent numeric engine for cross-validation

Pure-numpy surface (no FEM import required):
    IsotropicMaterial, OrthotropicMaterial, Material
    RegionMat, SectionInput, SectionResult
    rotate_stiffness_6x6
    materials_from_b3_mat
    write_quad_xdmf

Backend-aware helpers (import on use; no extra FEM install if payload already present):
    cell_centroids, solver_cell_tags

Backend-dependent surface (import on use; only the chosen backend's deps are required at runtime):
    solve, recover_strains, recover_unit_load_strains,
    StrainField, UnitLoadStrainField
    read_xdmf, write_xdmf, from_gxbeam_vtu
    to_gxbeam_order, from_gxbeam_order, to_anba_order
"""

from .adapters import (
    anba_to_secfem_angles,
    anba_to_secfem_input,
    secfem_from_gxbeam_theta,
    secfem_to_anba_angles,
    secfem_to_anba_input,
)
from .config import RegionMat, SectionInput, materials_from_b3_mat
from .materials import IsotropicMaterial, Material, OrthotropicMaterial
from .post import from_gxbeam_order, to_anba_order, to_gxbeam_order
from .recovery import StrainField, UnitLoadStrainField, assemble_resultants_from_sigma
from .result import SectionResult
from .rotation3d import SEC_1, SEC_2, SEC_3, material_axes, rotate_stiffness_6x6


def solve(inp, backend: str | None = None):
    """Run the 6-unit-load Saint-Venant solve.

    Parameters
    ----------
    inp : SectionInput
        Input specification (mesh + materials).
    backend : str or None
        Override the backend ("fenicsx" or "mfem"). If None, uses inp.backend.
        The kwarg takes precedence and a copy of inp is made internally if needed.
    """
    from .solver import solve as _solve

    if backend is not None:
        inp = inp.model_copy(update={"backend": backend})
    return _solve(inp)


def recover_strains(result):
    """Recover per-cell Voigt strain / stress for the 6 kinematic basis fields."""
    from .recovery import recover_strains as _rec

    return _rec(result)


def recover_unit_load_strains(result):
    """Recover per-cell Voigt strain / stress for the 6 applied unit load cases [Fx, Fy, Fz, Mx, My, Mz]."""
    from .recovery import recover_unit_load_strains as _rec

    return _rec(result)


def read_xdmf(path, comm=None):
    from .mesh import read_xdmf as _r

    return _r(path, comm)


def write_xdmf(path, mesh, cell_tags=None):
    from .mesh import write_xdmf as _w

    return _w(path, mesh, cell_tags)


def write_quad_xdmf(path, coords, quads, cell_tags=None):
    """Write a quad mesh from arrays (meshio; no FEM backend required)."""
    from .mesh import write_quad_xdmf as _w

    return _w(path, coords, quads, cell_tags=cell_tags)


def cell_centroids(mesh):
    """Per-cell centroids in the mesh's own cell order (backend-aware)."""
    from .mesh import cell_centroids as _c

    return _c(mesh)


def solver_cell_tags(result, mesh_path):
    """Per-cell tags in the solver's cell order (backend-aware)."""
    from .mesh import solver_cell_tags as _s

    return _s(result, mesh_path)


def from_gxbeam_vtu(path):
    from .mesh import from_gxbeam_vtu as _f

    return _f(path)


def plot_section(inp, res, out_path, **kwargs):
    """Render mesh + centres (elastic, mass, shear) + neutral axes; see viz.plot_section."""
    from .viz import plot_section as _ps

    return _ps(inp, res, out_path, **kwargs)


def plot_warping(res, mode, out_path, **kwargs):
    """Render the section deformed under one unit-load mode; see viz.plot_warping."""
    from .viz import plot_warping as _pw

    return _pw(res, mode, out_path, **kwargs)


def plot_unit_load_fields(res, fields, out_path, **kwargs):
    """2×3 grid of recovered unit-load stress/strain; see viz.plot_unit_load_fields."""
    from .viz import plot_unit_load_fields as _pf

    return _pf(res, fields, out_path, **kwargs)


def prepare_env(*, threads: int = 1, cache_home=None):
    """Pin BLAS/OpenMP and share FFCx cache; see :mod:`b3_secfem.warm`."""
    from .warm import prepare_env as _pe

    return _pe(threads=threads, cache_home=cache_home)


def warm_up(**kwargs):
    """Dummy fenicsx solve to pay FFCx JIT once per process; see :mod:`b3_secfem.warm`."""
    from .warm import warm_up as _wu

    return _wu(**kwargs)


__all__ = [
    "anba_to_secfem_angles",
    "anba_to_secfem_input",
    "assemble_resultants_from_sigma",
    "from_gxbeam_order",
    "IsotropicMaterial",
    "Material",
    "OrthotropicMaterial",
    "RegionMat",
    "SectionInput",
    "SectionResult",
    "cell_centroids",
    "from_gxbeam_vtu",
    "materials_from_b3_mat",
    "plot_section",
    "plot_unit_load_fields",
    "plot_warping",
    "prepare_env",
    "read_xdmf",
    "recover_strains",
    "recover_unit_load_strains",
    "SEC_1",
    "SEC_2",
    "SEC_3",
    "material_axes",
    "secfem_from_gxbeam_theta",
    "secfem_to_anba_angles",
    "secfem_to_anba_input",
    "rotate_stiffness_6x6",
    "solve",
    "solver_cell_tags",
    "StrainField",
    "UnitLoadStrainField",
    "to_anba_order",
    "to_gxbeam_order",
    "warm_up",
    "write_quad_xdmf",
    "write_xdmf",
]
