"""Analytical reference for the iso steel NACA 0024 airfoil case.

References:
  * EA, EIx, EIy via high-precision quadrature on the analytical NACA half-
    thickness formula. The mesh's coordinate frame (chord centred at x=0,
    y in [-half_t(x+0.5), +half_t(x+0.5)]) is the integration domain;
    no centroid shift — moments of area are reported about the **origin**,
    matching how the engines integrate over the as-given mesh.
  * GJ via an independent Prandtl stress-function Poisson solve on the
    same VTU mesh (``-∇²φ = 2`` in Ω, ``φ = 0`` on ∂Ω, then ``J = 2 ∫ φ dA``).
    Solved with FEniCSx CG-2 directly — no b3_secfem chain machinery
    is involved.

GAx and GAy (Timoshenko shear stiffnesses) are NOT computed: they involve
the shear-warping function and Cowper coefficients, which is more work
than this probe needs.

Run::

    /home/wr1/projects/b3/b3_secfem/.venv/bin/python examples/probe_airfoil_analytical.py
"""

from __future__ import annotations

import dolfinx  # noqa: F401  - load before juliacall in cross_check_b3_gx
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).parent))
from cross_check_b3_gx import (  # noqa: E402
    STEEL,
    _gxbeam_section_from_arrays,
    iso_row,
)
from _anba_runner_host import anba_section_from_arrays  # noqa: E402
from _gx_vtu import write_gx_vtu  # noqa: E402
from b3_secfem import (  # noqa: E402
    SectionInput,
    from_gxbeam_vtu,
    solve,
    to_gxbeam_order,
)
from probe_airfoil_3way import gen_airfoil  # noqa: E402


def naca_halft(eta: float, thickness: float = 0.24, chord: float = 1.0) -> float:
    """NACA 00xx half-thickness at chord fraction ``eta`` (closed-TE form)."""
    return (thickness / 0.2) * chord * (
        0.2969 * np.sqrt(eta) - 0.1260 * eta - 0.3516 * eta ** 2
        + 0.2843 * eta ** 3 - 0.1036 * eta ** 4
    )


def analytical_section_props(
    chord: float = 1.0,
    thickness: float = 0.24,
    eta_clip: float = 0.01,
) -> dict:
    """Compute A, x_c, I_xx, I_yy, I_xy in the chord-centred frame.

    Mesh coords: x ∈ [-chord/2, +chord/2], y ∈ [-half_t, +half_t].
    """
    eta_lo, eta_hi = eta_clip, 1.0 - eta_clip

    def x_of_eta(e):  # mesh x coordinate
        return chord * (e - 0.5)

    def half_t(e):
        return naca_halft(e, thickness=thickness, chord=chord)

    # A = ∫ 2·half_t(eta) · (chord deta) over eta ∈ [eta_clip, 1-eta_clip]
    A, _ = quad(lambda e: 2 * half_t(e) * chord, eta_lo, eta_hi, epsabs=1e-12, epsrel=1e-12)

    # x_c = (∫ x dA) / A
    Sx, _ = quad(
        lambda e: x_of_eta(e) * 2 * half_t(e) * chord,
        eta_lo, eta_hi, epsabs=1e-12, epsrel=1e-12,
    )
    x_c = Sx / A

    # I_xx = ∫ y² dA = ∫ (2/3) half_t(eta)³ · chord deta
    I_xx, _ = quad(
        lambda e: (2.0 / 3.0) * half_t(e) ** 3 * chord,
        eta_lo, eta_hi, epsabs=1e-12, epsrel=1e-12,
    )

    # I_yy_origin = ∫ x² dA = ∫ x_of_eta² · 2·half_t(eta) · chord deta
    I_yy_origin, _ = quad(
        lambda e: x_of_eta(e) ** 2 * 2 * half_t(e) * chord,
        eta_lo, eta_hi, epsabs=1e-12, epsrel=1e-12,
    )
    # I_yy_centroid = parallel-axis shift back from origin
    I_yy_centroid = I_yy_origin - A * x_c ** 2

    # I_xy = 0 by y-symmetry
    return {
        "A": A,
        "x_c": x_c,
        "y_c": 0.0,
        "I_xx": I_xx,
        "I_yy_origin": I_yy_origin,
        "I_yy_centroid": I_yy_centroid,
        "I_xy": 0.0,
    }


def saint_venant_J(vtu_path: Path) -> float:
    """Solve Prandtl stress-function Poisson on the airfoil mesh and return J.

        -∇² φ = 2  in Ω,  φ = 0 on ∂Ω
        J = 2 ∫_Ω φ dA

    Independent of b3_secfem's chain solver: just a plain CG-2 Poisson.
    """
    import ufl
    from dolfinx import default_scalar_type, fem, mesh as dmesh
    from dolfinx.fem import locate_dofs_topological
    from dolfinx.fem.petsc import LinearProblem
    from b3_secfem import from_gxbeam_vtu as _ingest

    info = _ingest(vtu_path)
    msh = info["mesh"]

    V = fem.functionspace(msh, ("Lagrange", 2))
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L = fem.Constant(msh, default_scalar_type(2.0)) * v * ufl.dx

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    boundary_facets = dmesh.exterior_facet_indices(msh.topology)
    boundary_dofs = locate_dofs_topological(V, msh.topology.dim - 1, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-12},
        petsc_options_prefix="anal_sv_",
    )
    sol = problem.solve()

    integral = float(fem.assemble_scalar(fem.form(2.0 * sol * ufl.dx)))
    return integral


def main() -> int:
    chord, thickness = 1.0, 0.24
    eta_clip = 0.01
    n_chord, n_thick = 64, 16

    props = analytical_section_props(chord, thickness, eta_clip)
    print(
        f"NACA 0024 chord={chord} thickness={thickness} eta_clip={eta_clip}\n"
        f"  A             = {props['A']:.6e} m²\n"
        f"  x_centroid    = {props['x_c']:.6e} m\n"
        f"  I_xx          = {props['I_xx']:.6e} m⁴   (y² about chord axis)\n"
        f"  I_yy_origin   = {props['I_yy_origin']:.6e} m⁴   (x² about y=0 line through origin)\n"
        f"  I_yy_centroid = {props['I_yy_centroid']:.6e} m⁴   (x² about centroid)\n"
    )

    E = STEEL.E
    nu = STEEL.nu
    G = E / (2 * (1 + nu))

    coords, quads = gen_airfoil(
        chord=chord, thickness=thickness,
        n_chord=n_chord, n_thick=n_thick, eta_clip=eta_clip,
    )
    n_cells = quads.shape[0]
    print(f"Mesh: n_cells={n_cells}, n_chord={n_chord}, n_thick={n_thick}\n")
    mat_props = np.tile(iso_row(STEEL), (n_cells, 1))
    theta = np.zeros(n_cells)
    out_dir = Path(__file__).parent / "cross_check_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    vtu = out_dir / "af_iso_analytical.vtu"
    write_gx_vtu(vtu, coords, quads, mat_props, theta)

    # Independent Saint-Venant J on the same mesh
    J_sv = saint_venant_J(vtu)

    EA_an = E * props["A"]
    EIx_an = E * props["I_xx"]
    EIy_an_origin = E * props["I_yy_origin"]
    EIy_an_centroid = E * props["I_yy_centroid"]
    GJ_sv = G * J_sv

    # Three engines on the same mesh
    K_gx, *_ = _gxbeam_section_from_arrays(coords, quads, mat_props, theta)
    info = from_gxbeam_vtu(vtu)
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=[STEEL] * info["n_cells"],
        per_cell_alpha_deg=np.degrees(info["theta"]),
    )
    K_us = to_gxbeam_order(solve(inp).K)
    anba = anba_section_from_arrays(
        coords, quads, STEEL,
        fiber_orientation_deg=np.full(n_cells, 90.0),
        plane_orientation_deg=np.full(n_cells, 0.0),
    )
    K_anba = to_gxbeam_order(anba["K"])

    metrics = ["EA", "GJ", "EIx", "EIy"]
    refs = [
        ("analytical (chord-centred)", EA_an, GJ_sv, EIx_an, EIy_an_origin),
        ("analytical (centroidal Iyy)", EA_an, GJ_sv, EIx_an, EIy_an_centroid),
        ("gxbeam_section",
         K_gx[0, 0], K_gx[3, 3], K_gx[4, 4], K_gx[5, 5]),
        ("b3_secfem",
         K_us[0, 0], K_us[3, 3], K_us[4, 4], K_us[5, 5]),
        ("ANBA4",
         K_anba[0, 0], K_anba[3, 3], K_anba[4, 4], K_anba[5, 5]),
    ]

    print(f"  {'engine':<32} | " + " | ".join(f"{m:>14}" for m in metrics))
    print("-" * 96)
    for label, *vals in refs:
        print(f"  {label:<32} | " + " | ".join(f"{v:>14.4e}" for v in vals))

    print()
    print("Relative error vs analytical (centroidal Iyy convention):")
    ref = (EA_an, GJ_sv, EIx_an, EIy_an_centroid)
    for label, *vals in refs[2:]:
        rel = [(v - r) / r for v, r in zip(vals, ref)]
        print(f"  {label:<32} | " + " | ".join(f"{100*r:>13.2f}%" for r in rel))

    print()
    print("Relative error vs analytical (origin Iyy convention):")
    ref = (EA_an, GJ_sv, EIx_an, EIy_an_origin)
    for label, *vals in refs[2:]:
        rel = [(v - r) / r for v, r in zip(vals, ref)]
        print(f"  {label:<32} | " + " | ".join(f"{100*r:>13.2f}%" for r in rel))

    return 0


if __name__ == "__main__":
    sys.exit(main())
