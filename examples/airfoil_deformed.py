"""Show the deformed state of a hollow composite airfoil under several
unit load cases. In-plane displacement is scaled up so the section
visibly twists / bends / shears; the out-of-plane warping ``u_z`` is
shown as a colormap on top of the deformed mesh.

Run::

    .venv/bin/python examples/airfoil_deformed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from b3_secfem import (
    OrthotropicMaterial,
    RegionMat,
    SectionInput,
    plot_warping,
    solve,
)
from b3_secfem._meshlib import airfoil_hollow  # noqa: E402

ROOT = Path(__file__).parent.resolve()
OUT = ROOT / "deformed_out"


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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    glass = _glass_ud()
    carbon = _carbon_ud()
    path, info = airfoil_hollow(
        OUT, naca="0024", chord=1.0, skin_t=0.005, spar_t=0.020,
        web_loc=0.4, web_t=0.005, ds=0.04, wns=4,
    )
    rmats = {}
    for tag, name, theta_deg in info["materials"]:
        m = glass if "glass" in name else carbon
        rmats[tag] = RegionMat(material=m, beta_deg=0.0, alpha_deg=float(theta_deg))
    inp = SectionInput(mesh_path=path, region_materials=rmats)
    res = solve(inp)

    print(f"K[Fz, Fz]    = {res.K[2, 2]:.3e}    (axial)")
    print(f"K[My, My]    = {res.K[4, 4]:.3e}    (chord-wise bending)")
    print(f"K[Mz, Mz]    = {res.K[5, 5]:.3e}    (torsion)")
    print(f"K_section_xy = {res.K_section_xy:.3e}    (in-plane shear, scalar)")

    cases = [
        ("axial",     2,    "Axial $F_z$ — uniform $u_z$ + Poisson contraction"),
        ("bending_x", 3,    "Bending $M_x$ — $u_z$ linear in $-y$ (flap-wise) + Poisson"),
        ("bending_y", 4,    "Bending $M_y$ — $u_z$ linear in $x$ (chord-wise) + Poisson"),
        ("torsion",   5,    "Torsion $M_z$ — rigid in-plane rotation + Saint-Venant warping"),
        ("xy_shear",  "xy", "In-plane shear $\\gamma_{xy}$ — section sheared by 0.5 (assumed) + correction"),
    ]
    for slug, mode, title in cases:
        out_path = OUT / f"{slug}.png"
        plot_warping(res, mode=mode, out_path=out_path, title=title)
        print(f"wrote {out_path}")

    # Combined panel: 2 rows x 3 cols, last cell blank.
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=140)
    fig.suptitle("Hollow composite airfoil — deformed state under unit load cases",
                 fontsize=12)
    for (slug, _mode, _title), ax in zip(cases, axes.flat):
        img = plt.imread(OUT / f"{slug}.png")
        ax.imshow(img)
        ax.set_axis_off()
    # Hide any unused cells.
    for ax in list(axes.flat)[len(cases):]:
        ax.set_axis_off()
    fig.tight_layout()
    combined = OUT / "combined.png"
    fig.savefig(combined, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {combined}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
