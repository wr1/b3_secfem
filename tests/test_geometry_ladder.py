"""Geometry ladder: solid -> hollow cyl -> I-beam -> airfoil -> airfoil + web.

Each step adds geometric / topological complexity. We validate against
analytic where available and check qualitative consistency where not.
"""

from __future__ import annotations

import pytest

dolfinx = pytest.importorskip("dolfinx")

import numpy as np

from b3_secfem import (
    IsotropicMaterial,
    OrthotropicMaterial,
    RegionMat,
    SectionInput,
    solve,
)
from tests._meshlib import (
    airfoil_hollow,
    airfoil_solid,
    hollow_cylinder,
    hollow_ellipse,
    i_beam,
    solid_ellipse,
)


# ----------------------------------------------------------------------
# Step 2: hollow circular cylinder (annulus)
# ----------------------------------------------------------------------


def test_hollow_cylinder_axial_bending_torsion(tmp_path):
    """Thin annular ring: K[Fz, Fz] = E A; K[M*, M*] = E I; K[Mz, Mz] = G J."""
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    R, t = 0.05, 0.005       # mean radius 50 mm, wall 5 mm
    path, info = hollow_cylinder(tmp_path, R, t, n_circ=64, n_rad=4)

    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)

    G = iso.E / (2 * (1 + iso.nu))
    EA = iso.E * info["A"]
    EI = iso.E * info["I"]
    GJ = G * info["J"]

    np.testing.assert_allclose(res.K[2, 2], EA, rtol=2e-2)
    np.testing.assert_allclose(res.K[3, 3], EI, rtol=3e-2)
    np.testing.assert_allclose(res.K[4, 4], EI, rtol=3e-2)
    np.testing.assert_allclose(res.K[5, 5], GJ, rtol=3e-2)


def test_hollow_cylinder_centres(tmp_path):
    """Symmetric annulus: tension and elastic centres at the origin."""
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    path, _info = hollow_cylinder(tmp_path, 0.05, 0.005, 64, 4)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    np.testing.assert_allclose(res.tension_center, (0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(res.elastic_center, (0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(res.mass_center, (0.0, 0.0), atol=1e-12)


def test_hollow_cylinder_shear(tmp_path):
    """Thin-wall annulus: K[Fx, Fx] approximates G A / 2 (the thin-wall shear factor)."""
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    path, info = hollow_cylinder(tmp_path, 0.05, 0.005, 64, 4)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    G = iso.E / (2 * (1 + iso.nu))
    K_shear_thin = G * info["A"] / 2.0
    # The Saint-Venant shear stiffness for a thin-wall tube is exactly G A / 2.
    # Mesh discretisation gives ~5 % error on a 64 x 4 mesh.
    np.testing.assert_allclose(res.K[0, 0], K_shear_thin, rtol=8e-2)
    np.testing.assert_allclose(res.K[1, 1], K_shear_thin, rtol=8e-2)


# ----------------------------------------------------------------------
# Step 2b: ellipses (solid + hollow)
# ----------------------------------------------------------------------


def test_solid_ellipse_axial_bending_torsion(tmp_path):
    """Solid ellipse: K[Fz, Fz] = E A; bending matches I_xx, I_yy;
    torsion matches the Saint-Venant exact value
    K[Mz, Mz] = G * pi * a^3 * b^3 / (a^2 + b^2).
    """
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    a, b = 0.06, 0.04
    path, info = solid_ellipse(tmp_path, a, b, n_circ=96, n_rad=12)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)

    G = iso.E / (2 * (1 + iso.nu))
    np.testing.assert_allclose(res.K[2, 2], iso.E * info["A"], rtol=2e-2)
    np.testing.assert_allclose(res.K[3, 3], iso.E * info["I_xx"], rtol=3e-2)
    np.testing.assert_allclose(res.K[4, 4], iso.E * info["I_yy"], rtol=3e-2)
    # Saint-Venant exact for solid ellipse
    np.testing.assert_allclose(res.K[5, 5], G * info["J_solid"], rtol=4e-2)


def test_solid_ellipse_centres(tmp_path):
    """Symmetric ellipse: all three centres at the origin."""
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    path, _ = solid_ellipse(tmp_path, 0.06, 0.04)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    np.testing.assert_allclose(res.tension_center, (0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(res.elastic_center, (0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(res.mass_center, (0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(res.shear_center, (0.0, 0.0), atol=1e-6)


def test_hollow_ellipse_axial_and_bending(tmp_path):
    """Annular ellipse: K[Fz, Fz] = E A and K[M*,M*] = E I*."""
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    a, b, t = 0.06, 0.04, 0.005
    path, info = hollow_ellipse(tmp_path, a, b, t, n_circ=64, n_rad=4)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    np.testing.assert_allclose(res.K[2, 2], iso.E * info["A"], rtol=2e-2)
    np.testing.assert_allclose(res.K[3, 3], iso.E * info["I_xx"], rtol=4e-2)
    np.testing.assert_allclose(res.K[4, 4], iso.E * info["I_yy"], rtol=4e-2)
    # Different bending stiffnesses about the two axes (a > b means EI_yy > EI_xx)
    assert res.K[4, 4] > res.K[3, 3]


# ----------------------------------------------------------------------
# Step 3: I-beam
# ----------------------------------------------------------------------


def test_i_beam_axial_and_bending(tmp_path):
    """I-beam K[Fz, Fz] = E A and K[Mx, Mx] = E I_xx within mesh tolerance."""
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    b, h = 0.10, 0.20            # 100 x 200 mm
    t_f, t_w = 0.012, 0.008
    path, info = i_beam(tmp_path, b, h, t_w, t_f)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    np.testing.assert_allclose(res.K[2, 2], iso.E * info["A"], rtol=2e-2)
    np.testing.assert_allclose(res.K[3, 3], iso.E * info["I_xx"], rtol=4e-2)
    np.testing.assert_allclose(res.K[4, 4], iso.E * info["I_yy"], rtol=4e-2)


def test_i_beam_shear_carriers(tmp_path):
    """For an I-beam, strong-axis shear (Fy) is carried by the web and
    weak-axis shear (Fx) by the flanges. The Saint-Venant K[Fy, Fy] is
    well-approximated by G * A_web; K[Fx, Fx] by ~G * A_flanges.

    For this section A_flanges > A_web, so K[Fx, Fx] > K[Fy, Fy].
    """
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    b, h = 0.10, 0.20
    t_f, t_w = 0.012, 0.008
    path, _info = i_beam(tmp_path, b, h, t_w, t_f)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)

    G = iso.E / (2 * (1 + iso.nu))
    A_web = (h - 2 * t_f) * t_w
    A_flanges = 2 * b * t_f
    # Web carries strong-axis shear within ~10% (G*A_web is the standard
    # thin-wall estimate; FE gives a slightly higher value).
    np.testing.assert_allclose(res.K[1, 1], G * A_web, rtol=0.10)
    # Flanges carry weak-axis shear; rough estimate ~G * A_flanges (5/6 of
    # it in the thin-wall limit, but the flanges are sloppier shear
    # carriers than the web due to their geometry — wide tolerance).
    assert 0.5 * G * A_flanges < res.K[0, 0] < G * A_flanges
    # A_flanges > A_web for this section, so weak-axis shear is stiffer.
    assert res.K[0, 0] > res.K[1, 1]


# ----------------------------------------------------------------------
# Step 4: hollow composite airfoil (skin only)
# ----------------------------------------------------------------------


def _glass_ud():
    return OrthotropicMaterial(
        E1=45e9, E2=12e9, E3=12e9,
        G12=4.5e9, G13=4.5e9, G23=4.0e9,
        nu12=0.3, nu13=0.3, nu23=0.4, rho=2000.0,
        name="glass_ud",
    )


def _carbon_ud():
    return OrthotropicMaterial(
        E1=140e9, E2=10e9, E3=10e9,
        G12=5e9, G13=5e9, G23=3.5e9,
        nu12=0.3, nu13=0.3, nu23=0.4, rho=1600.0,
        name="carbon_ud",
    )


def _region_materials_for_hollow(info, glass, carbon):
    """Map the per-(material_name, theta) tags returned by airfoil_hollow
    to b3_secfem RegionMat with the right OrthotropicMaterial + alpha=0."""
    region_materials = {}
    for tag, mat_name, theta_deg in info["materials"]:
        material = glass if "glass" in mat_name else carbon
        region_materials[tag] = RegionMat(
            material=material,
            beta_deg=0.0,
            alpha_deg=float(theta_deg),
        )
    return region_materials


def test_hollow_airfoil_skin_only(tmp_path):
    """Hollow composite NACA0024 section (skin only, no web).

    Axial stiffness K[Fz, Fz] should be dominated by the spar-cap carbon
    plies. Strong-axis bending K[Mx, Mx] (about chord) >> weak-axis K[My, My]
    is no longer trivially true for a hollow section since the spar caps
    are far above/below the neutral axis -- but the chord is still much
    longer than the thickness, so K[My, My] (about the thin axis,
    distance squared in chord) > K[Mx, Mx] (about the long axis, distance
    squared in thickness).
    """
    glass = _glass_ud()
    carbon = _carbon_ud()
    path, info = airfoil_hollow(
        tmp_path, naca="0024", chord=1.0, skin_t=0.005, spar_t=0.020,
        web_loc=None, ds=0.04,
    )
    inp = SectionInput(
        mesh_path=path,
        region_materials=_region_materials_for_hollow(info, glass, carbon),
    )
    res = solve(inp)

    # Sanity: K diagonal entries finite and positive
    for i in range(6):
        assert np.isfinite(res.K[i, i]) and res.K[i, i] > 0
    # Chord >> thickness so K[My, My] > K[Mx, Mx]
    assert res.K[4, 4] > 5.0 * res.K[3, 3]


def test_shear_centre_symmetric_at_origin(tmp_path):
    """Symmetric sections must have shear centre at the origin (within
    floating-point noise) — geometric symmetry forces it."""
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)

    # Hollow cylinder (centro-symmetric)
    path, _ = hollow_cylinder(tmp_path / "cyl", 0.05, 0.005, 64, 4)
    (tmp_path / "cyl").mkdir(exist_ok=True)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    np.testing.assert_allclose(res.shear_center, (0.0, 0.0), atol=1e-6)

    # I-beam (doubly symmetric)
    path, _ = i_beam(tmp_path / "ibeam", 0.10, 0.20, 0.008, 0.012)
    (tmp_path / "ibeam").mkdir(exist_ok=True)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    np.testing.assert_allclose(res.shear_center, (0.0, 0.0), atol=1e-6)


def test_shear_centre_offset_for_asymmetric_airfoil(tmp_path):
    """A hollow airfoil with a single off-centre web has its shear centre
    distinct from the elastic centre — the asymmetric distribution of
    shear-carrying material moves the no-twist point away from the
    geometric centroid."""
    glass = _glass_ud()
    carbon = _carbon_ud()
    path, info = airfoil_hollow(
        tmp_path, naca="0024", chord=1.0, skin_t=0.005, spar_t=0.020,
        web_loc=0.4, web_t=0.005, ds=0.04, wns=4,
    )
    inp = SectionInput(
        mesh_path=path,
        region_materials=_region_materials_for_hollow(info, glass, carbon),
    )
    res = solve(inp)
    # Shear centre must be inside the chord [0, 1] and clearly different
    # from the elastic centre (otherwise the formula is degenerate).
    assert 0.0 <= res.shear_center[0] <= 1.0
    assert abs(res.shear_center[0] - res.elastic_center[0]) > 0.05


def test_hollow_airfoil_with_web(tmp_path):
    """Same airfoil + a single carbon shear web at 40% chord.

    Adding the web should NOT noticeably change axial K[Fz, Fz] (web has
    small area), but should raise K[Mz, Mz] (torsion via the closed-cell
    shear flow) and K[Mx, Mx] (web contributes to flap-wise bending).
    """
    glass = _glass_ud()
    carbon = _carbon_ud()

    path_no, info_no = airfoil_hollow(
        tmp_path / "no_web", naca="0024", chord=1.0, skin_t=0.005, spar_t=0.020,
        web_loc=None, ds=0.04,
    )
    (tmp_path / "no_web").mkdir(exist_ok=True)
    inp_no = SectionInput(
        mesh_path=path_no,
        region_materials=_region_materials_for_hollow(info_no, glass, carbon),
    )
    res_no = solve(inp_no)

    path_w, info_w = airfoil_hollow(
        tmp_path / "with_web", naca="0024", chord=1.0, skin_t=0.005, spar_t=0.020,
        web_loc=0.4, web_t=0.005, ds=0.04, wns=4,
    )
    (tmp_path / "with_web").mkdir(exist_ok=True)
    inp_w = SectionInput(
        mesh_path=path_w,
        region_materials=_region_materials_for_hollow(info_w, glass, carbon),
    )
    res_w = solve(inp_w)

    # Web should noticeably stiffen torsion and flap-wise bending
    assert res_w.K[5, 5] > 1.10 * res_no.K[5, 5]
    assert res_w.K[3, 3] > 1.02 * res_no.K[3, 3]


# ----------------------------------------------------------------------
# Step 6 (legacy): solid block airfoil and tagged-web variant remain
# available for cross-checks against simpler analytic envelopes.
# ----------------------------------------------------------------------


def test_airfoil_solid_isotropic(tmp_path):
    """Solid (filled) NACA0012-like section, isotropic. Used as a
    reference for the hollow composite cases."""
    iso = IsotropicMaterial(E=70e9, nu=0.3, rho=2700.0)
    path, info = airfoil_solid(tmp_path, chord=1.0, thickness=0.12, n_chord=32, n_thick=8)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    np.testing.assert_allclose(res.K[2, 2], iso.E * info["A"], rtol=2e-2)
    assert res.K[4, 4] > 50.0 * res.K[3, 3]
