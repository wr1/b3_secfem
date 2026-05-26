"""b3_secfem — composite cross-section property solver (FEniCSx).

Pure-numpy surface (no FEniCSx import required):
    IsotropicMaterial, OrthotropicMaterial, Material
    RegionMat, SectionInput
    rotate_stiffness_6x6
    materials_from_b3_mat

FEniCSx-dependent surface (import on use):
    solve, recover_strains, recover_unit_load_strains,
    SectionResult, StrainField, UnitLoadStrainField
    read_xdmf, write_xdmf, from_gxbeam_vtu
    to_gxbeam_order, to_anba_order
"""

from .config import RegionMat, SectionInput, materials_from_b3_mat
from .materials import IsotropicMaterial, Material, OrthotropicMaterial
from .post import to_anba_order, to_gxbeam_order
from .recovery import StrainField, UnitLoadStrainField
from .rotation3d import rotate_stiffness_6x6


def solve(inp):
    """Run the 6-unit-load Saint-Venant solve. See solver.solve."""
    from .solver import solve as _solve
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


__all__ = [
    "IsotropicMaterial",
    "Material",
    "OrthotropicMaterial",
    "RegionMat",
    "SectionInput",
    "from_gxbeam_vtu",
    "materials_from_b3_mat",
    "plot_section",
    "plot_warping",
    "read_xdmf",
    "recover_strains",
    "recover_unit_load_strains",
    "rotate_stiffness_6x6",
    "solve",
    "StrainField",
    "UnitLoadStrainField",
    "to_anba_order",
    "to_gxbeam_order",
    "write_xdmf",
]
