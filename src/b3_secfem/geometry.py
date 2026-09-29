"""Dolfinx-free section meshes.

Builders return ``(SectionMesh, info)``. Call :meth:`SectionMesh.write` to
store a single-grid XDMF. ``b3_secfem._meshlib`` is the old file-writing API.
"""

from __future__ import annotations

import numpy as np

from .mesh_io import SectionMesh

__all__ = [
    "airfoil_hollow",
    "airfoil_solid",
    "airfoil_with_web",
    "hollow_cylinder",
    "hollow_ellipse",
    "i_beam",
    "rectangle",
    "solid_ellipse",
]


def _quads(
    points: np.ndarray,
    cells: np.ndarray,
    tags: np.ndarray | None = None,
) -> SectionMesh:
    pts = np.asarray(points, dtype=np.float64)
    return SectionMesh(
        points=pts[:, :2],
        cells=np.asarray(cells, dtype=np.int64),
        cell_type="quad",
        cell_tags=None if tags is None else np.asarray(tags, dtype=np.int32),
        cell_data={},
        source=None,
    )


def rectangle(a: float, b: float, nx: int, ny: int) -> tuple[SectionMesh, dict]:
    """Rectangle ``[-a/2, a/2] x [-b/2, b/2]`` with ``nx`` by ``ny`` quads.

    ``info`` carries the analytic area and centroidal second moments.
    """
    xs = np.linspace(-a / 2, a / 2, nx + 1)
    ys = np.linspace(-b / 2, b / 2, ny + 1)
    coords = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    quads = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            quads.append([n0, n0 + 1, n0 + 1 + (nx + 1), n0 + (nx + 1)])
    info = {
        "A": a * b,
        "I_xx": a * b**3 / 12.0,
        "I_yy": b * a**3 / 12.0,
        "a": a,
        "b": b,
        "nx": nx,
        "ny": ny,
    }
    return _quads(coords, quads), info


def hollow_cylinder(
    R_mid: float, t: float, n_circ: int, n_rad: int = 4
) -> tuple[SectionMesh, dict]:
    """Annular ring of mean radius ``R_mid`` and wall thickness ``t``."""
    coords = []
    for k in range(n_rad + 1):
        r = R_mid - t / 2 + k * t / n_rad
        for i in range(n_circ):
            ang = 2 * np.pi * i / n_circ
            coords.append([r * np.cos(ang), r * np.sin(ang)])
    quads = []
    for k in range(n_rad):
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            n0 = k * n_circ + i
            n1 = k * n_circ + i_next
            n2 = (k + 1) * n_circ + i_next
            n3 = (k + 1) * n_circ + i
            quads.append([n0, n1, n2, n3])
    R_o = R_mid + t / 2
    R_i = R_mid - t / 2
    A = np.pi * (R_o**2 - R_i**2)
    I_area = np.pi * (R_o**4 - R_i**4) / 4.0
    J = np.pi * (R_o**4 - R_i**4) / 2.0
    info = {"A": A, "I": I_area, "J": J, "R_mid": R_mid, "t": t}
    return _quads(coords, quads), info


def solid_ellipse(
    a: float,
    b: float,
    n_circ: int = 64,
    n_rad: int = 12,
    hole_frac: float = 0.02,
) -> tuple[SectionMesh, dict]:
    """Filled ellipse with semi-axes ``(a, b)`` and a tiny centre hole.

    Analytic ``A``, ``I_xx``, ``I_yy`` include the hole. ``J_solid`` is the
    Saint-Venant torsion constant of the full ellipse (hole ignored).
    """
    coords = []
    for k in range(n_rad + 1):
        r = hole_frac + k * (1.0 - hole_frac) / n_rad
        for i in range(n_circ):
            ang = 2 * np.pi * i / n_circ
            coords.append([r * a * np.cos(ang), r * b * np.sin(ang)])
    quads = []
    for k in range(n_rad):
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            n0 = k * n_circ + i
            n1 = k * n_circ + i_next
            n2 = (k + 1) * n_circ + i_next
            n3 = (k + 1) * n_circ + i
            quads.append([n0, n1, n2, n3])
    A = np.pi * a * b * (1 - hole_frac**2)
    I_xx = (np.pi / 4) * a * b**3 * (1 - hole_frac**4)
    I_yy = (np.pi / 4) * a**3 * b * (1 - hole_frac**4)
    J_solid = np.pi * a**3 * b**3 / (a**2 + b**2)
    info = {
        "A": A,
        "I_xx": I_xx,
        "I_yy": I_yy,
        "J_solid": J_solid,
        "a": a,
        "b": b,
        "hole_frac": hole_frac,
    }
    return _quads(coords, quads), info


def hollow_ellipse(
    a: float,
    b: float,
    t: float,
    n_circ: int = 64,
    n_rad: int = 4,
) -> tuple[SectionMesh, dict]:
    """Annular ellipse, outer ``(a, b)``, inner ``(a - t, b - t)``.

    The wall is linearly interpolated, so it is not a uniform-normal shell.
    """
    if t <= 0 or t >= min(a, b):
        msg = f"need 0 < t < min(a, b); got a={a}, b={b}, t={t}"
        raise ValueError(msg)
    a_i = a - t
    b_i = b - t
    coords = []
    for k in range(n_rad + 1):
        sa = a_i + k * (a - a_i) / n_rad
        sb = b_i + k * (b - b_i) / n_rad
        for i in range(n_circ):
            ang = 2 * np.pi * i / n_circ
            coords.append([sa * np.cos(ang), sb * np.sin(ang)])
    quads = []
    for k in range(n_rad):
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            n0 = k * n_circ + i
            n1 = k * n_circ + i_next
            n2 = (k + 1) * n_circ + i_next
            n3 = (k + 1) * n_circ + i
            quads.append([n0, n1, n2, n3])
    A = np.pi * (a * b - a_i * b_i)
    I_xx = (np.pi / 4) * (a * b**3 - a_i * b_i**3)
    I_yy = (np.pi / 4) * (a**3 * b - a_i**3 * b_i)
    info = {
        "A": A,
        "I_xx": I_xx,
        "I_yy": I_yy,
        "a": a,
        "b": b,
        "t": t,
        "a_i": a_i,
        "b_i": b_i,
    }
    return _quads(coords, quads), info


def i_beam(
    b: float,
    h: float,
    t_w: float,
    t_f: float,
    n_b: int = 16,
    n_h: int = 24,
    n_tw: int = 4,
    n_tf: int = 4,
) -> tuple[SectionMesh, dict]:
    """Symmetric I-beam: two flanges ``b x t_f`` and a web ``t_w x (h - 2 t_f)``."""
    h_w = h - 2 * t_f
    if h_w <= 0:
        msg = f"web height must be > 0; got h={h}, t_f={t_f}"
        raise ValueError(msg)

    coords: list[tuple[float, float]] = []
    quads: list[tuple[int, int, int, int]] = []
    node_index: dict[tuple[int, int], int] = {}
    rnd = 1_000_000

    def get(x: float, y: float) -> int:
        key = (round(x * rnd), round(y * rnd))
        if key in node_index:
            return node_index[key]
        node_index[key] = len(coords)
        coords.append((x, y))
        return node_index[key]

    def panel(x0, x1, y0, y1, nx, ny):
        xs = np.linspace(x0, x1, nx + 1)
        ys = np.linspace(y0, y1, ny + 1)
        for j in range(ny):
            for i in range(nx):
                n00 = get(xs[i], ys[j])
                n10 = get(xs[i + 1], ys[j])
                n11 = get(xs[i + 1], ys[j + 1])
                n01 = get(xs[i], ys[j + 1])
                quads.append((n00, n10, n11, n01))

    n_left = max(1, n_b // 2 - n_tw // 2)
    panel(-b / 2, -t_w / 2, -h / 2, -h_w / 2, n_left, n_tf)
    panel(-t_w / 2, t_w / 2, -h / 2, -h_w / 2, n_tw, n_tf)
    panel(t_w / 2, b / 2, -h / 2, -h_w / 2, n_left, n_tf)
    panel(-b / 2, -t_w / 2, h_w / 2, h / 2, n_left, n_tf)
    panel(-t_w / 2, t_w / 2, h_w / 2, h / 2, n_tw, n_tf)
    panel(t_w / 2, b / 2, h_w / 2, h / 2, n_left, n_tf)
    panel(-t_w / 2, t_w / 2, -h_w / 2, h_w / 2, n_tw, n_h)

    A = 2 * b * t_f + t_w * h_w
    I_xx = (b * h**3) / 12.0 - ((b - t_w) * h_w**3) / 12.0
    I_yy = (2 * t_f * b**3) / 12.0 + (h_w * t_w**3) / 12.0
    info = {
        "A": A,
        "I_xx": I_xx,
        "I_yy": I_yy,
        "b": b,
        "h": h,
        "t_w": t_w,
        "t_f": t_f,
    }
    return _quads(coords, quads), info


def airfoil_hollow(
    naca: str = "0024",
    chord: float = 1.0,
    skin_t: float = 0.005,
    spar_t: float = 0.020,
    web_loc: float | None = None,
    web_t: float = 0.005,
    n_chord: int = 80,
    ds: float = 0.04,
    wns: int = 4,
) -> tuple[SectionMesh, dict]:
    """Hollow composite airfoil via ``airfoilmesh.afmesh``.

    Skips when ``airfoilmesh`` is not installed. Cell tags are one per unique
    ``(material name, theta)`` pair, starting at 1.
    """
    import pytest

    pytest.importorskip("airfoilmesh")
    from airfoilmesh import Layer, Material, afmesh, naca4

    ply_glass = Material("glass_ud")
    ply_carbon = Material("carbon_ud")
    xaf, yaf = naca4(naca, n=n_chord)
    xbreak = np.array([0.0, 0.2, 0.5, 1.0])
    skin_3 = [
        Layer(ply_glass, skin_t, 0.0),
        Layer(ply_glass, skin_t, 0.0),
    ]
    spar_3 = [
        Layer(ply_glass, skin_t, 0.0),
        Layer(ply_carbon, spar_t, 0.0),
    ]
    segments = [skin_3, spar_3, skin_3]
    if web_loc is not None:
        webloc = np.array([web_loc])
        webs = [[Layer(ply_carbon, web_t, 0.0)]]
    else:
        webloc = np.array([])
        webs = []

    nodes, elements, _surf = afmesh(
        xaf,
        yaf,
        chord=chord,
        twist=0.0,
        paxis=0.5,
        xbreak=xbreak,
        webloc=webloc,
        segments=segments,
        webs=webs,
        ds=ds,
        wns=wns,
    )
    coords = np.array([[n.x, n.y] for n in nodes], dtype=np.float64)
    quads = np.array([e.nodenum for e in elements], dtype=np.int64)
    keys = [(e.material.name, round(float(e.theta), 6)) for e in elements]
    unique = sorted(set(keys))
    key_to_tag = {k: i + 1 for i, k in enumerate(unique)}
    cell_tags = np.array([key_to_tag[k] for k in keys], dtype=np.int32)
    info = {
        "materials": [
            (tag, name, np.degrees(theta_rad))
            for (name, theta_rad), tag in key_to_tag.items()
        ],
        "n_cells": quads.shape[0],
        "naca": naca,
        "chord": chord,
        "web_loc": web_loc,
        "has_web": web_loc is not None,
    }
    return _quads(coords, quads, cell_tags), info


def _naca00_half_thickness(
    eta: np.ndarray, chord: float, thickness: float
) -> np.ndarray:
    return (
        (thickness / 0.2)
        * chord
        * (
            0.2969 * np.sqrt(eta)
            - 0.1260 * eta
            - 0.3516 * eta**2
            + 0.2843 * eta**3
            - 0.1036 * eta**4
        )
    )


def airfoil_solid(
    chord: float = 1.0,
    thickness: float = 0.12,
    n_chord: int = 32,
    n_thick: int = 8,
) -> tuple[SectionMesh, dict]:
    """Solid NACA00xx-like section, structured through the thickness."""
    eta = np.linspace(0.0, 1.0, n_chord + 1)
    half_t = _naca00_half_thickness(eta, chord, thickness)
    xi = np.linspace(-1.0, 1.0, n_thick + 1)
    coords = np.zeros(((n_chord + 1) * (n_thick + 1), 2))
    for i, e in enumerate(eta):
        for j, s in enumerate(xi):
            coords[i * (n_thick + 1) + j] = [chord * (e - 0.5), s * half_t[i]]
    quads = []
    for i in range(n_chord):
        for j in range(n_thick):
            n00 = i * (n_thick + 1) + j
            n10 = (i + 1) * (n_thick + 1) + j
            n11 = (i + 1) * (n_thick + 1) + j + 1
            n01 = i * (n_thick + 1) + j + 1
            quads.append([n00, n10, n11, n01])
    # np.trapezoid is numpy>=2; np.trapz remains on the numpy 1.x apt build.
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = 2.0 * trap(half_t, eta * chord)
    info = {"A": area, "chord": chord, "thickness": thickness}
    return _quads(coords, quads), info


def airfoil_with_web(
    chord: float = 1.0,
    thickness: float = 0.12,
    web_x_frac: float = 0.4,
    n_chord: int = 32,
    n_thick: int = 8,
) -> tuple[SectionMesh, dict]:
    """Solid airfoil with tag 2 on a vertical web strip and tag 1 elsewhere."""
    eta = np.linspace(0.0, 1.0, n_chord + 1)
    half_t = _naca00_half_thickness(eta, chord, thickness)
    xi = np.linspace(-1.0, 1.0, n_thick + 1)
    coords = np.zeros(((n_chord + 1) * (n_thick + 1), 2))
    for i, e in enumerate(eta):
        for j, s in enumerate(xi):
            coords[i * (n_thick + 1) + j] = [chord * (e - 0.5), s * half_t[i]]
    quads = []
    cell_tags = []
    web_x = chord * (web_x_frac - 0.5)
    web_band = chord / n_chord
    for i in range(n_chord):
        x_centre = chord * (eta[i] + eta[i + 1]) / 2 - chord / 2
        is_web = abs(x_centre - web_x) < web_band
        for j in range(n_thick):
            n00 = i * (n_thick + 1) + j
            n10 = (i + 1) * (n_thick + 1) + j
            n11 = (i + 1) * (n_thick + 1) + j + 1
            n01 = i * (n_thick + 1) + j + 1
            quads.append([n00, n10, n11, n01])
            cell_tags.append(2 if is_web else 1)
    info = {"chord": chord, "thickness": thickness, "n_cells": len(quads)}
    return _quads(coords, quads, np.asarray(cell_tags, dtype=np.int32)), info
