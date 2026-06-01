"""Build a Typst report walking the geometry ladder.

Outputs go to ``examples/report_out/``:

  * one PNG per geometry showing the mesh with centres / principal axes
  * ``report.typ`` -- the Typst source
  * ``report.pdf`` -- compiled output (if ``typst`` is on PATH)

Run::

    .venv/bin/python examples/build_report.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from b3_secfem import (
    IsotropicMaterial,
    OrthotropicMaterial,
    RegionMat,
    SectionInput,
    plot_section,
    solve,
    write_xdmf,
)

from b3_secfem._meshlib import (  # noqa: E402
    airfoil_hollow,
    hollow_cylinder,
    hollow_ellipse,
    i_beam,
    solid_ellipse,
)
ROOT = Path(__file__).parent.resolve()
OUT = ROOT / "report_out"


# ---------------------------------------------------------------------------
# Geometry runs
# ---------------------------------------------------------------------------


def run_solid_rectangle() -> dict:
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    a = b = 0.10
    n = 16
    path = OUT / "mesh_rectangle.xdmf"
    m = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [(-a / 2, -b / 2), (a / 2, b / 2)],
        [n, n],
        cell_type=dmesh.CellType.quadrilateral,
    )
    write_xdmf(path, m)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    return {
        "label": "Solid rectangle",
        "subtitle": f"isotropic steel, {a*1000:.0f} x {b*1000:.0f} mm",
        "inp": inp,
        "res": res,
        "notes": [
            f"E A           = {iso.E * a * b:.3e} (analytic)",
            f"E I           = {iso.E * a * b**3 / 12:.3e} (analytic)",
            f"5/6 G A       = {5/6 * iso.E / (2*(1+iso.nu)) * a * b:.3e}"
            " (Saint-Venant exact for a rectangle)",
        ],
    }


def run_hollow_cylinder() -> dict:
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    R, t = 0.05, 0.005
    path, info = hollow_cylinder(OUT, R, t, n_circ=64, n_rad=4)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    G = iso.E / (2 * (1 + iso.nu))
    return {
        "label": "Hollow cylinder",
        "subtitle": f"R = {R*1000:.0f} mm, wall = {t*1000:.0f} mm",
        "inp": inp,
        "res": res,
        "notes": [
            f"E A   = {iso.E * info['A']:.3e}",
            f"E I   = {iso.E * info['I']:.3e}",
            f"G J   = {G * info['J']:.3e}",
            f"G A/2 = {G * info['A'] / 2:.3e} (thin-wall transverse shear)",
        ],
    }


def run_solid_ellipse() -> dict:
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    a, b = 0.06, 0.04
    path, info = solid_ellipse(OUT, a, b, n_circ=96, n_rad=12)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    G = iso.E / (2 * (1 + iso.nu))
    return {
        "label": "Solid ellipse",
        "subtitle": f"isotropic steel, semi-axes {a*1000:.0f} x {b*1000:.0f} mm",
        "inp": inp,
        "res": res,
        "notes": [
            f"E A         = {iso.E * info['A']:.3e}  (analytic, ignoring 0.04 % central hole)",
            f"E I_xx      = {iso.E * info['I_xx']:.3e}",
            f"E I_yy      = {iso.E * info['I_yy']:.3e}",
            f"G J_SaintVenant = {G * info['J_solid']:.3e}"
            f"  (= G · pi a^3 b^3 / (a^2 + b^2), exact for solid ellipse)",
        ],
    }


def run_hollow_ellipse() -> dict:
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    a, b, t = 0.06, 0.04, 0.005
    path, info = hollow_ellipse(OUT, a, b, t, n_circ=64, n_rad=4)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    return {
        "label": "Hollow elliptical tube",
        "subtitle": f"semi-axes {a*1000:.0f} x {b*1000:.0f} mm, wall {t*1000:.0f} mm",
        "inp": inp,
        "res": res,
        "notes": [
            f"E A    = {iso.E * info['A']:.3e}",
            f"E I_xx = {iso.E * info['I_xx']:.3e}  (about x-axis)",
            f"E I_yy = {iso.E * info['I_yy']:.3e}  (about y-axis; larger because a > b)",
        ],
    }


def run_i_beam() -> dict:
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    b, h = 0.10, 0.20
    t_f, t_w = 0.012, 0.008
    path, info = i_beam(OUT, b, h, t_w, t_f)
    inp = SectionInput(mesh_path=path, region_materials={1: RegionMat(material=iso)})
    res = solve(inp)
    G = iso.E / (2 * (1 + iso.nu))
    A_web = (h - 2 * t_f) * t_w
    A_flanges = 2 * b * t_f
    return {
        "label": "I-beam",
        "subtitle": f"{b*1000:.0f} x {h*1000:.0f} mm, t_w={t_w*1000:.0f}, t_f={t_f*1000:.0f}",
        "inp": inp,
        "res": res,
        "notes": [
            f"E A         = {iso.E * info['A']:.3e}",
            f"E I_xx      = {iso.E * info['I_xx']:.3e}   (strong axis)",
            f"E I_yy      = {iso.E * info['I_yy']:.3e}   (weak axis)",
            f"G A_web     = {G * A_web:.3e}   (estimate for K[Fy, Fy])",
            f"G A_flanges = {G * A_flanges:.3e}   (estimate for K[Fx, Fx])",
        ],
    }


def _glass_ud():
    return OrthotropicMaterial(
        E1=45e9, E2=12e9, E3=12e9,
        G12=4.5e9, G13=4.5e9, G23=4.0e9,
        nu12=0.3, nu13=0.3, nu23=0.4, rho=2000.0, name="glass_ud",
    )


def _carbon_ud():
    return OrthotropicMaterial(
        E1=140e9, E2=10e9, E3=10e9,
        G12=5e9, G13=5e9, G23=3.5e9,
        nu12=0.3, nu13=0.3, nu23=0.4, rho=1600.0, name="carbon_ud",
    )


def _region_materials_for_hollow(info, glass, carbon):
    region_materials = {}
    for tag, mat_name, theta_deg in info["materials"]:
        material = glass if "glass" in mat_name else carbon
        region_materials[tag] = RegionMat(
            material=material, beta_deg=0.0, alpha_deg=float(theta_deg),
        )
    return region_materials


def run_hollow_airfoil_skin_only() -> dict:
    glass = _glass_ud()
    carbon = _carbon_ud()
    path, info = airfoil_hollow(
        OUT, naca="0024", chord=1.0, skin_t=0.005, spar_t=0.020,
        web_loc=None, ds=0.04,
    )
    inp = SectionInput(
        mesh_path=path,
        region_materials=_region_materials_for_hollow(info, glass, carbon),
    )
    res = solve(inp)
    return {
        "label": "Hollow composite airfoil",
        "subtitle": "NACA 0024, glass-UD skin (3+3 mm) with carbon-UD spar caps (20 mm)",
        "inp": inp,
        "res": res,
        "notes": [
            f"n_cells = {info['n_cells']}",
            "Skin = 2 plies glass UD (5 mm each), spar = 2 plies (5 mm glass + 20 mm carbon).",
            "All plies fibre-aligned with beam axis (alpha = 0).",
            "K[My, My] >> K[Mx, Mx] because chord >> thickness.",
        ],
    }


def run_hollow_airfoil_with_web() -> dict:
    glass = _glass_ud()
    carbon = _carbon_ud()
    path, info = airfoil_hollow(
        OUT, naca="0024", chord=1.0, skin_t=0.005, spar_t=0.020,
        web_loc=0.4, web_t=0.005, ds=0.04, wns=4,
    )
    inp = SectionInput(
        mesh_path=path,
        region_materials=_region_materials_for_hollow(info, glass, carbon),
    )
    res = solve(inp)
    return {
        "label": "Hollow composite airfoil with shear web",
        "subtitle": "NACA 0024 with single carbon-UD shear web at 40% chord",
        "inp": inp,
        "res": res,
        "notes": [
            f"n_cells = {info['n_cells']}",
            "Same skin as the no-web case; one carbon-UD web (5 mm) at 40% chord.",
            "Web closes the airfoil cell and raises K[Mz, Mz] (torsion) noticeably.",
        ],
    }


# ---------------------------------------------------------------------------
# Typst report writer
# ---------------------------------------------------------------------------


def _format_K(K: np.ndarray) -> str:
    labels = ["F_x", "F_y", "F_z", "M_x", "M_y", "M_z"]
    rows = []
    for i, lbl in enumerate(labels):
        row = [f"`{lbl}`"]
        for j in range(6):
            v = K[i, j]
            row.append(f"`{v:+.2e}`")
        rows.append(row)
    header = ["``"] + [f"`{lbl}`" for lbl in labels]
    body = ", ".join(", ".join(r) for r in [header, *rows])
    return body


def _typst_for_run(run: dict, png_rel: str) -> str:
    K = run["res"].K
    cell_grid = _format_K(K)
    notes_block = "\n".join(f"- {n}" for n in run["notes"])
    cx, cy = run["res"].tension_center
    sx, sy = run["res"].shear_center
    ex, ey = run["res"].elastic_center
    mx, my = run["res"].mass_center
    return f"""
== {run['label']}

#text(style: "italic")[{run['subtitle']}]

#figure(
  image("{png_rel}", width: 95%),
  caption: [Mesh with centres (red = elastic/tension, orange = mass, green = shear) and neutral axes (from elastic centre)],
)

*Centres:*
- tension/elastic: ({cx:+.3e}, {cy:+.3e})
- mass:            ({mx:+.3e}, {my:+.3e})
- shear:           ({sx:+.3e}, {sy:+.3e})

*Cross-section stiffness $K$ (6x6, $[F_x, F_y, F_z, M_x, M_y, M_z]$, units SI):*

#table(
  columns: 7,
  align: (left, right, right, right, right, right, right),
  stroke: 0.4pt,
  {cell_grid}
)

*Notes:*

{notes_block}

#pagebreak()
"""


def write_report(runs: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = []
    for run in runs:
        slug = run["label"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        png_rel = f"{slug}.png"
        plot_section(
            run["inp"], run["res"],
            OUT / png_rel,
            title=f"{run['label']}: {run['subtitle']}",
        )
        blocks.append(_typst_for_run(run, png_rel))

    typ_path = OUT / "report.typ"
    typ_path.write_text(
        "#set page(paper: \"a4\", margin: 1.6cm)\n"
        "#set text(font: \"DejaVu Sans\", size: 10pt)\n"
        "#set heading(numbering: \"1.\")\n"
        "\n"
        "#align(center, text(size: 18pt, weight: \"bold\")[b3_secfem geometry ladder])\n"
        "#align(center, text(size: 11pt)[\n"
        "  Cross-section properties for a series of test geometries\\\n"
        "  generated by `examples/build_report.py`\n"
        "])\n"
        "\n"
        "#v(0.5cm)\n"
        "= Geometries\n"
        + "\n".join(blocks)
    )
    return typ_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [
        run_solid_rectangle(),
        run_solid_ellipse(),
        run_hollow_cylinder(),
        run_hollow_ellipse(),
        run_i_beam(),
        run_hollow_airfoil_skin_only(),
        run_hollow_airfoil_with_web(),
    ]
    typ_path = write_report(runs)
    print(f"Wrote {typ_path}")
    if shutil.which("typst") is None:
        print("typst not on PATH; skipping PDF compile")
        return 0
    pdf_path = typ_path.with_suffix(".pdf")
    res = subprocess.run(
        ["typst", "compile", str(typ_path), str(pdf_path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("typst stderr:", res.stderr)
        return 1
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
