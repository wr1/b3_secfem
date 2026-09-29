"""Deprecated file-writing mesh builders.

Use :mod:`b3_secfem.geometry`. These functions keep the old signature
``(directory, ...) -> (path, info)`` and write a single-grid XDMF.
"""

from __future__ import annotations

from pathlib import Path

from ._deprecation import warn_deprecated
from . import geometry as _geometry

_FILES = {
    "hollow_cylinder": "ring.xdmf",
    "solid_ellipse": "ellipse_solid.xdmf",
    "hollow_ellipse": "ellipse_hollow.xdmf",
    "i_beam": "ibeam.xdmf",
    "airfoil_hollow": "airfoil_hollow.xdmf",
    "airfoil_solid": "airfoil.xdmf",
    "airfoil_with_web": "airfoil_web.xdmf",
}


def _write(name: str, directory, mesh) -> Path:
    warn_deprecated(
        f"b3_secfem._meshlib.{name}",
        f"b3_secfem.geometry.{name}",
        stacklevel=4,
    )
    path = Path(directory) / _FILES[name]
    mesh.write(path)
    return path


def hollow_cylinder(tmp_path, R_mid, t, n_circ, n_rad=4):
    """Write ``ring.xdmf``. See :func:`b3_secfem.geometry.hollow_cylinder`."""
    mesh, info = _geometry.hollow_cylinder(R_mid, t, n_circ, n_rad=n_rad)
    return _write("hollow_cylinder", tmp_path, mesh), info


def solid_ellipse(tmp_path, a, b, n_circ=64, n_rad=12, hole_frac=0.02):
    """Write ``ellipse_solid.xdmf``. See :func:`b3_secfem.geometry.solid_ellipse`."""
    mesh, info = _geometry.solid_ellipse(
        a, b, n_circ=n_circ, n_rad=n_rad, hole_frac=hole_frac
    )
    return _write("solid_ellipse", tmp_path, mesh), info


def hollow_ellipse(tmp_path, a, b, t, n_circ=64, n_rad=4):
    """Write ``ellipse_hollow.xdmf``. See :func:`b3_secfem.geometry.hollow_ellipse`."""
    mesh, info = _geometry.hollow_ellipse(a, b, t, n_circ=n_circ, n_rad=n_rad)
    return _write("hollow_ellipse", tmp_path, mesh), info


def i_beam(tmp_path, b, h, t_w, t_f, n_b=16, n_h=24, n_tw=4, n_tf=4):
    """Write ``ibeam.xdmf``. See :func:`b3_secfem.geometry.i_beam`."""
    mesh, info = _geometry.i_beam(
        b, h, t_w, t_f, n_b=n_b, n_h=n_h, n_tw=n_tw, n_tf=n_tf
    )
    return _write("i_beam", tmp_path, mesh), info


def airfoil_hollow(
    tmp_path,
    naca="0024",
    chord=1.0,
    skin_t=0.005,
    spar_t=0.020,
    web_loc=None,
    web_t=0.005,
    n_chord=80,
    ds=0.04,
    wns=4,
):
    """Write the hollow-airfoil XDMF. See :func:`b3_secfem.geometry.airfoil_hollow`."""
    mesh, info = _geometry.airfoil_hollow(
        naca=naca,
        chord=chord,
        skin_t=skin_t,
        spar_t=spar_t,
        web_loc=web_loc,
        web_t=web_t,
        n_chord=n_chord,
        ds=ds,
        wns=wns,
    )
    filename = (
        "airfoil_hollow_with_web.xdmf" if web_loc is not None else "airfoil_hollow.xdmf"
    )
    warn_deprecated(
        "b3_secfem._meshlib.airfoil_hollow",
        "b3_secfem.geometry.airfoil_hollow",
        stacklevel=2,
    )
    path = Path(tmp_path) / filename
    mesh.write(path)
    return path, info


def airfoil_solid(tmp_path, chord=1.0, thickness=0.12, n_chord=32, n_thick=8):
    """Write ``airfoil.xdmf``. See :func:`b3_secfem.geometry.airfoil_solid`."""
    mesh, info = _geometry.airfoil_solid(
        chord=chord, thickness=thickness, n_chord=n_chord, n_thick=n_thick
    )
    return _write("airfoil_solid", tmp_path, mesh), info


def airfoil_with_web(
    tmp_path,
    chord=1.0,
    thickness=0.12,
    web_x_frac=0.4,
    n_chord=32,
    n_thick=8,
):
    """Write ``airfoil_web.xdmf``. See :func:`b3_secfem.geometry.airfoil_with_web`."""
    mesh, info = _geometry.airfoil_with_web(
        chord=chord,
        thickness=thickness,
        web_x_frac=web_x_frac,
        n_chord=n_chord,
        n_thick=n_thick,
    )
    return _write("airfoil_with_web", tmp_path, mesh), info
