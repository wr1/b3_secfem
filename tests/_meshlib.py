"""Mesh-builder helpers shared by the geometry-ladder tests.

All helpers return a ``(mesh_path, geom_info)`` pair. The mesh is written
to ``tmp_path / "<name>.xdmf"`` so it can be loaded directly via the
public ``b3_secfem`` API. Cell tags are written when more than one
region is needed.

Geometries (in order of ladder complexity):

  rectangle(a, b, n)              -- already in test_iso_rectangle.py
  hollow_cylinder(R, t, n_circ, n_rad)
  i_beam(b, h, t_w, t_f, n)
  airfoil(chord, thickness, n)    -- NACA-like ellipse / wing-shaped solid
  airfoil_with_web(...)            -- airfoil split into skin + shear-web region

All meshes are 2D quads. Coordinates in metres.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI


def _write_quad_mesh(
    path: Path,
    coords: np.ndarray,
    quads: np.ndarray,
    cell_tags: np.ndarray | None = None,
) -> Any:
    """Build a dolfinx quad mesh from (n_nodes, 2) coords and (n_cells, 4)
    connectivity, validate / fix CCW winding, and write XDMF.

    Optional ``cell_tags`` (length n_cells) is written as a MeshTags entity.
    """
    import basix.ufl
    import ufl
    from dolfinx import io
    from dolfinx import mesh as dmesh
    from dolfinx.mesh import meshtags

    # Validate CCW winding (signed shoelace area > 0).
    p = coords[quads]
    sa = (
        p[:, 0, 0] * p[:, 1, 1] - p[:, 1, 0] * p[:, 0, 1]
        + p[:, 1, 0] * p[:, 2, 1] - p[:, 2, 0] * p[:, 1, 1]
        + p[:, 2, 0] * p[:, 3, 1] - p[:, 3, 0] * p[:, 2, 1]
        + p[:, 3, 0] * p[:, 0, 1] - p[:, 0, 0] * p[:, 3, 1]
    )
    flip = sa < 0
    if flip.any():
        quads[flip] = quads[flip][:, [0, 3, 2, 1]]

    # Convert CCW (BL, BR, TR, TL) -> dolfinx 0.10 tensor-product ordering
    # (BL, TL, BR, TR) by the permutation [0, 3, 1, 2].
    quads_dolfinx = quads[:, [0, 3, 1, 2]]

    e = basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,))
    domain = ufl.Mesh(e)
    mesh = dmesh.create_mesh(MPI.COMM_WORLD, quads_dolfinx, domain, coords)

    if cell_tags is not None:
        cell_dim = mesh.topology.dim
        n_cells = cell_tags.size
        indices = np.arange(n_cells, dtype=np.int32)
        values = cell_tags.astype(np.int32)
        mesh.topology.create_connectivity(cell_dim, cell_dim)
        mt = meshtags(mesh, cell_dim, indices, values)
        mt.name = "cell_tags"
        with io.XDMFFile(mesh.comm, str(path), "w") as xf:
            xf.write_mesh(mesh)
            xf.write_meshtags(mt, mesh.geometry)
    else:
        with io.XDMFFile(mesh.comm, str(path), "w") as xf:
            xf.write_mesh(mesh)
    return mesh


def hollow_cylinder(
    tmp_path: Path, R_mid: float, t: float, n_circ: int, n_rad: int = 4
) -> tuple[Path, dict]:
    """Annular ring mesh of mean radius ``R_mid``, wall thickness ``t``.

    Returns ``(path, info)`` where ``info`` carries analytic section
    properties for validation.
    """
    coords = []
    for k in range(n_rad + 1):
        r = R_mid - t / 2 + k * t / n_rad
        for i in range(n_circ):
            ang = 2 * np.pi * i / n_circ
            coords.append([r * np.cos(ang), r * np.sin(ang)])
    coords = np.asarray(coords, dtype=np.float64)
    quads = []
    for k in range(n_rad):
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            n0 = k * n_circ + i
            n1 = k * n_circ + i_next
            n2 = (k + 1) * n_circ + i_next
            n3 = (k + 1) * n_circ + i
            quads.append([n0, n1, n2, n3])
    quads = np.asarray(quads, dtype=np.int64)

    path = tmp_path / "ring.xdmf"
    _write_quad_mesh(path, coords, quads)

    R_o = R_mid + t / 2
    R_i = R_mid - t / 2
    A = np.pi * (R_o ** 2 - R_i ** 2)
    I = np.pi * (R_o ** 4 - R_i ** 4) / 4.0
    J = np.pi * (R_o ** 4 - R_i ** 4) / 2.0
    return path, {"A": A, "I": I, "J": J, "R_mid": R_mid, "t": t}


def solid_ellipse(
    tmp_path: Path, a: float, b: float, n_circ: int = 64, n_rad: int = 12,
    hole_frac: float = 0.02,
) -> tuple[Path, dict]:
    """Filled ellipse with semi-axes ``(a, b)``.

    Polar mesh with a tiny inner hole at fraction ``hole_frac`` of the
    semi-axes to avoid the centre-point singularity. With the default
    2 % hole the missing area is 4e-4 of the total.

    Analytic info returned (full ellipse, ignoring the tiny hole when
    that's the better approximation):

        A   = pi * a * b * (1 - hole_frac^2)
        I_xx = (pi/4) * a * b^3 * (1 - hole_frac^4)
        I_yy = (pi/4) * a^3 * b * (1 - hole_frac^4)
        J_solid = pi * a^3 * b^3 / (a^2 + b^2)        (Saint-Venant, full ellipse)
    """
    coords = []
    for k in range(n_rad + 1):
        r = hole_frac + k * (1.0 - hole_frac) / n_rad
        for i in range(n_circ):
            ang = 2 * np.pi * i / n_circ
            coords.append([r * a * np.cos(ang), r * b * np.sin(ang)])
    coords = np.asarray(coords, dtype=np.float64)
    quads = []
    for k in range(n_rad):
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            n0 = k * n_circ + i
            n1 = k * n_circ + i_next
            n2 = (k + 1) * n_circ + i_next
            n3 = (k + 1) * n_circ + i
            quads.append([n0, n1, n2, n3])
    quads = np.asarray(quads, dtype=np.int64)

    path = tmp_path / "ellipse_solid.xdmf"
    _write_quad_mesh(path, coords, quads)

    A = np.pi * a * b * (1 - hole_frac ** 2)
    I_xx = (np.pi / 4) * a * b ** 3 * (1 - hole_frac ** 4)
    I_yy = (np.pi / 4) * a ** 3 * b * (1 - hole_frac ** 4)
    J_solid = np.pi * a ** 3 * b ** 3 / (a ** 2 + b ** 2)
    return path, {
        "A": A, "I_xx": I_xx, "I_yy": I_yy, "J_solid": J_solid,
        "a": a, "b": b, "hole_frac": hole_frac,
    }


def hollow_ellipse(
    tmp_path: Path, a: float, b: float, t: float,
    n_circ: int = 64, n_rad: int = 4,
) -> tuple[Path, dict]:
    """Annular ellipse: outer (a, b), inner (a - t, b - t) (concentric, similar).

    The radial interpolation runs linearly between the inner and outer
    elliptical layers, so the wall is slightly thicker along the major
    axis than along the minor (not a true uniform-normal-thickness shell).

    Analytic for the concentric similar pair:

        A   = pi * (a*b - a_i*b_i)
        I_xx = (pi/4) * (a*b^3 - a_i*b_i^3)
        I_yy = (pi/4) * (a^3*b - a_i^3*b_i)
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
    coords = np.asarray(coords, dtype=np.float64)
    quads = []
    for k in range(n_rad):
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            n0 = k * n_circ + i
            n1 = k * n_circ + i_next
            n2 = (k + 1) * n_circ + i_next
            n3 = (k + 1) * n_circ + i
            quads.append([n0, n1, n2, n3])
    quads = np.asarray(quads, dtype=np.int64)

    path = tmp_path / "ellipse_hollow.xdmf"
    _write_quad_mesh(path, coords, quads)

    A = np.pi * (a * b - a_i * b_i)
    I_xx = (np.pi / 4) * (a * b ** 3 - a_i * b_i ** 3)
    I_yy = (np.pi / 4) * (a ** 3 * b - a_i ** 3 * b_i)
    return path, {"A": A, "I_xx": I_xx, "I_yy": I_yy,
                  "a": a, "b": b, "t": t, "a_i": a_i, "b_i": b_i}


def i_beam(
    tmp_path: Path, b: float, h: float, t_w: float, t_f: float,
    n_b: int = 16, n_h: int = 24, n_tw: int = 4, n_tf: int = 4,
) -> tuple[Path, dict]:
    """Symmetric I-beam: top + bottom flanges (b x t_f) plus web (t_w x (h - 2 t_f)).

    Built from three structured rectangular panels stitched at shared
    nodes. Returns analytic A and I_xx (strong-axis) for validation.
    """
    h_w = h - 2 * t_f                         # web height (between flange interiors)
    if h_w <= 0:
        msg = f"web height must be > 0; got h={h}, t_f={t_f}"
        raise ValueError(msg)

    coords: list[tuple[float, float]] = []
    quads: list[tuple[int, int, int, int]] = []
    node_index: dict[tuple[int, int], int] = {}

    # Node sharing is keyed by (rounded x, rounded y) so panels stitched
    # along common edges share their boundary nodes.
    rnd = 1_000_000

    def get(x: float, y: float) -> int:
        key = (int(round(x * rnd)), int(round(y * rnd)))
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
                n00 = get(xs[i],     ys[j])
                n10 = get(xs[i + 1], ys[j])
                n11 = get(xs[i + 1], ys[j + 1])
                n01 = get(xs[i],     ys[j + 1])
                quads.append((n00, n10, n11, n01))

    # The web's grid lines must align with x = -t_w/2 and x = +t_w/2 in the
    # flange panels so the web stitches into the flanges along those columns.
    # Build the bottom flange in three sub-panels: [-b/2, -t_w/2], [-t_w/2, t_w/2],
    # [t_w/2, b/2], each with appropriate nx; similarly for the top flange.
    n_left = max(1, n_b // 2 - n_tw // 2)
    n_right = n_left
    panel(-b / 2,  -t_w / 2, -h / 2, -h_w / 2, n_left,  n_tf)
    panel(-t_w / 2, t_w / 2, -h / 2, -h_w / 2, n_tw,    n_tf)
    panel( t_w / 2, b / 2,   -h / 2, -h_w / 2, n_right, n_tf)
    panel(-b / 2,  -t_w / 2,  h_w / 2, h / 2,  n_left,  n_tf)
    panel(-t_w / 2, t_w / 2,  h_w / 2, h / 2,  n_tw,    n_tf)
    panel( t_w / 2, b / 2,    h_w / 2, h / 2,  n_right, n_tf)
    # Web spans the full inter-flange height (h_w) at width t_w.
    panel(-t_w / 2, t_w / 2, -h_w / 2, h_w / 2, n_tw, n_h)

    coords_arr = np.asarray(coords, dtype=np.float64)
    quads_arr = np.asarray(quads, dtype=np.int64)
    path = tmp_path / "ibeam.xdmf"
    _write_quad_mesh(path, coords_arr, quads_arr)

    A = 2 * b * t_f + t_w * h_w
    I_xx = (b * h ** 3) / 12.0 - ((b - t_w) * h_w ** 3) / 12.0
    I_yy = (2 * t_f * b ** 3) / 12.0 + (h_w * t_w ** 3) / 12.0
    return path, {"A": A, "I_xx": I_xx, "I_yy": I_yy, "b": b, "h": h, "t_w": t_w, "t_f": t_f}


def airfoil_hollow(
    tmp_path: Path,
    naca: str = "0024",
    chord: float = 1.0,
    skin_t: float = 0.005,
    spar_t: float = 0.020,
    web_loc: float | None = None,
    web_t: float = 0.005,
    n_chord: int = 80,
    ds: float = 0.04,
    wns: int = 4,
) -> tuple[Path, dict]:
    """Hollow composite airfoil section via airfoilmesh.afmesh.

    Returns ``(path, info)``. ``info`` carries:

      * ``materials`` -- list of (region_id, name, theta_deg) tuples,
        one per unique (material, theta) combo present in the elements.
      * ``n_cells``    -- total quad count.
      * ``naca``, ``chord``, ``web_loc`` -- geometry context.
      * ``has_web``    -- whether a shear web is present.

    Skin layup: ``[glass UD, carbon UD spar cap, glass UD]`` between
    breaks at 0%, 20%, 50% chord; just glass UD beyond.

    The mesh is written to ``tmp_path / "airfoil_hollow.xdmf"`` with
    integer cell tags = (region_id from materials list).
    """
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
        xaf, yaf,
        chord=chord, twist=0.0, paxis=0.5,
        xbreak=xbreak, webloc=webloc, segments=segments, webs=webs,
        ds=ds, wns=wns,
    )

    coords = np.array([[n.x, n.y] for n in nodes], dtype=np.float64)
    quads = np.array([e.nodenum for e in elements], dtype=np.int64)

    # Build a region tag per unique (material name, theta) pair.
    keys = [(e.material.name, round(float(e.theta), 6)) for e in elements]
    unique = sorted(set(keys))
    key_to_tag = {k: i + 1 for i, k in enumerate(unique)}
    cell_tags = np.array([key_to_tag[k] for k in keys], dtype=np.int32)
    info_materials = [(tag, name, np.degrees(theta_rad))
                      for (name, theta_rad), tag in key_to_tag.items()]

    name = "airfoil_hollow_with_web" if web_loc is not None else "airfoil_hollow"
    path = tmp_path / f"{name}.xdmf"
    _write_quad_mesh(path, coords, quads, cell_tags=cell_tags)

    return path, {
        "materials": info_materials,
        "n_cells": quads.shape[0],
        "naca": naca,
        "chord": chord,
        "web_loc": web_loc,
        "has_web": web_loc is not None,
    }


def airfoil_solid(
    tmp_path: Path, chord: float = 1.0, thickness: float = 0.12,
    n_chord: int = 32, n_thick: int = 8,
) -> tuple[Path, dict]:
    """Solid (filled) airfoil-like section: NACA00xx upper / lower surfaces.

    Mesh is structured (n_chord+1) x (n_thick+1) along the chord and
    through the thickness. Returns geometric info.
    """
    eta = np.linspace(0.0, 1.0, n_chord + 1)
    # NACA 00xx half-thickness (closed-trailing-edge form)
    half_t = (thickness / 0.2) * chord * (
        0.2969 * np.sqrt(eta) - 0.1260 * eta - 0.3516 * eta ** 2
        + 0.2843 * eta ** 3 - 0.1036 * eta ** 4
    )
    # Through-thickness coordinate xi in [-1, 1]
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
    quads = np.asarray(quads, dtype=np.int64)

    path = tmp_path / "airfoil.xdmf"
    _write_quad_mesh(path, coords, quads)

    # Total area (numerical, by trapezoidal rule on half-thickness)
    A = 2.0 * np.trapz(half_t, eta * chord)
    return path, {"A": A, "chord": chord, "thickness": thickness}


def airfoil_with_web(
    tmp_path: Path, chord: float = 1.0, thickness: float = 0.12,
    web_x_frac: float = 0.4, n_chord: int = 32, n_thick: int = 8,
) -> tuple[Path, dict]:
    """Airfoil mesh with a tagged shear-web region.

    Region tag = 2 for cells in a vertical strip of width
    ``2 * chord / n_chord`` centred at ``x = chord * (web_x_frac - 0.5)``;
    tag = 1 elsewhere. Use with ``region_materials = {1: skin, 2: web}``.
    """
    eta = np.linspace(0.0, 1.0, n_chord + 1)
    half_t = (thickness / 0.2) * chord * (
        0.2969 * np.sqrt(eta) - 0.1260 * eta - 0.3516 * eta ** 2
        + 0.2843 * eta ** 3 - 0.1036 * eta ** 4
    )
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
    quads = np.asarray(quads, dtype=np.int64)
    cell_tags = np.asarray(cell_tags, dtype=np.int32)

    path = tmp_path / "airfoil_web.xdmf"
    _write_quad_mesh(path, coords, quads, cell_tags=cell_tags)
    return path, {"chord": chord, "thickness": thickness, "n_cells": quads.shape[0]}
