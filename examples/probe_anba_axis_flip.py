"""Test whether the b3_secfem-vs-ANBA gap on the torture case closes
when we resolve the suspected Ez/Ex axis-flip in ANBA's input.

Hypothesis:
  The naive mapping (b3_secfem ``E1, E2, E3`` → ANBA's
  ``[E_xx, E_yy, E_zz]`` row 0 of the 3×3 matrix) places the fibre on
  ANBA's local x-axis, then we rotate it to the global beam axis z via
  the ``fiber_orientation = 90°, plane_orientation = 0°`` mapping.
  But empirically (probe_anba_ud.py) (fiber=0, plane=90) does NOT put
  fibre along z, while (fiber=90, plane=0) does — suggesting an axis
  swap between what ANBA's matrix-row convention says is the principal
  fibre axis and what its rotation convention assumes.

We test three plausible mappings on a single rotated UD ellipse (where
b3_secfem and ANBA both give well-defined answers and gxbeam is the
analytical reference):

  Mapping A (current):    [E1,E2,E3]/[G23,G13,G12]/[ν23,ν13,ν12], fiber=90−α, plane=0
  Mapping B (Ez=fibre):   [E2,E3,E1]/[G23,G12,G13]/[ν23,ν12,ν13], fiber=α,    plane=0
  Mapping C (Ex=fibre, no rot): [E1,E2,E3]/[G23,G13,G12]/[ν23,ν13,ν12], fiber=α,    plane=90

For each mapping, dump K[Fz,Fz] + 1/S[Mz,Mz] (decoupled torsion) and
compare to b3_secfem and gxbeam.

Run::

    /home/wr1/projects/b3/b3_secfem/.venv/bin/python examples/probe_anba_axis_flip.py
"""

from __future__ import annotations

import dolfinx  # noqa: F401  - load before juliacall
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cross_check_b3_gx import (  # noqa: E402
    UD_CARBON,
    _gxbeam_section_from_arrays,
    gen_ellipse_solid,
    ortho_row,
)
from _anba_runner_host import anba_section_from_arrays  # noqa: E402
from _gx_vtu import write_gx_vtu  # noqa: E402
from b3_secfem import (  # noqa: E402
    OrthotropicMaterial,
    SectionInput,
    from_gxbeam_vtu,
    solve,
    to_gxbeam_order,
)


def perm_material(mat: OrthotropicMaterial, perm: tuple[int, int, int]) -> OrthotropicMaterial:
    """Return a *physically equivalent* OrthotropicMaterial with axes 1,2,3 permuted.

    ``perm`` is the permutation: e.g. ``(2, 0, 1)`` puts old axis 3 onto
    new axis 1. The returned material has the same elastic properties
    re-labelled.
    """
    Es = [mat.E1, mat.E2, mat.E3]
    Gs = {(1, 2): mat.G12, (1, 3): mat.G13, (2, 3): mat.G23,
          (2, 1): mat.G12, (3, 1): mat.G13, (3, 2): mat.G23}
    nus = {(1, 2): mat.nu12, (1, 3): mat.nu13, (2, 3): mat.nu23}
    # nu_ij = nu_ji * E_i / E_j (major/minor Poisson convention)
    # We need new nu_12, nu_13, nu_23 in the new labelling.
    inv = [perm.index(i) for i in (0, 1, 2)]   # new index of old axis i

    def G(i, j):
        oi = perm[i - 1] + 1
        oj = perm[j - 1] + 1
        return Gs[(oi, oj)] if (oi, oj) in Gs else Gs[(oj, oi)]

    def nu(i, j):
        oi = perm[i - 1] + 1
        oj = perm[j - 1] + 1
        if (oi, oj) in nus:
            return nus[(oi, oj)]
        # nu_oioj = nu_ojoi * E_oj / E_oi
        return nus[(oj, oi)] * Es[oj - 1] / Es[oi - 1]

    return OrthotropicMaterial(
        E1=Es[perm[0]], E2=Es[perm[1]], E3=Es[perm[2]],
        G12=G(1, 2), G13=G(1, 3), G23=G(2, 3),
        nu12=nu(1, 2), nu13=nu(1, 3), nu23=nu(2, 3),
        rho=mat.rho, name=f"{mat.name}-perm{perm}",
    )


def main() -> int:
    # Single ellipse case at theta = 45° (the discriminating angle)
    coords, quads = gen_ellipse_solid(a=0.06, b=0.02, n_circ=64, n_rad=12)
    n_cells = quads.shape[0]
    alpha_deg = 45.0
    theta_arr = np.full(n_cells, np.radians(alpha_deg))

    out = Path(__file__).parent / "cross_check_out"
    out.mkdir(parents=True, exist_ok=True)
    mat_props = np.tile(ortho_row(UD_CARBON), (n_cells, 1))
    vtu = out / "ud_ellipse_axisflip.vtu"
    write_gx_vtu(vtu, coords, quads, mat_props, theta_arr)

    # Reference: gxbeam
    #K_gx_gxorder, *_ = _gxbeam_section_from_arrays(coords, quads, mat_props, theta_arr)
    #perm_gx_to_nat = np.array([1, 2, 0, 4, 5, 3])
    #K_gx = K_gx_gxorder[np.ix_(perm_gx_to_nat, perm_gx_to_nat)]

    # b3_secfem
    info = from_gxbeam_vtu(vtu)
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=[UD_CARBON] * info["n_cells"],
        per_cell_alpha_deg=np.full(info["n_cells"], alpha_deg),
    )
    res = solve(inp)
    K_b3 = res.K

    print(f"UD-on-ellipse, alpha = {alpha_deg}°, n_cells = {n_cells}")
    print()

    def report(label, K):
        S = np.linalg.inv(K)
        diag = np.diag(K)
        sd = 1.0 / np.diag(S)
        names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        print(f"  {label:<32} | "
              + " | ".join(f"{n}={v:>11.4e}" for n, v in zip(names, diag)))
        print(f"  {'  └─ 1/S diagonal':<32} | "
              + " | ".join(f"{n}={v:>11.4e}" for n, v in zip(names, sd)))

    #report("gxbeam", K_gx)
    report("b3_secfem", K_b3)
    print()

    # === Mapping A (current): pass material as-is, fiber=90-alpha, plane=0 ===
    anba_A = anba_section_from_arrays(
        coords, quads, UD_CARBON,
        fiber_orientation_deg=np.full(n_cells, 90.0 - alpha_deg),
        plane_orientation_deg=np.full(n_cells, 0.0),
    )
    K_A = anba_A["K"]
    report("ANBA A (fiber=90−α, plane=0)", K_A)

    # === Mapping B: permute material so fibre lives on ANBA's z-axis,
    # then rotate via fiber=alpha, plane=0 ===
    UD_perm = perm_material(UD_CARBON, (2, 1, 0))   # (E1,E2,E3)→(E3,E2,E1)
    anba_B = anba_section_from_arrays(
        coords, quads, UD_perm,
        fiber_orientation_deg=np.full(n_cells, alpha_deg),
        plane_orientation_deg=np.full(n_cells, 0.0),
    )
    K_B = anba_B["K"]
    report("ANBA B (Ez=fibre, fiber=α, plane=0)", K_B)

    # === Mapping C: original material, but use the example-style (fiber=alpha, plane=90) ===
    anba_C = anba_section_from_arrays(
        coords, quads, UD_CARBON,
        fiber_orientation_deg=np.full(n_cells, alpha_deg),
        plane_orientation_deg=np.full(n_cells, 90.0),
    )
    K_C = anba_C["K"]
    report("ANBA C (fiber=α, plane=90)", K_C)

    # === Mapping D: permute (E2,E3,E1) -- another candidate ===
    UD_permD = perm_material(UD_CARBON, (1, 2, 0))   # (E1,E2,E3) → (E2,E3,E1)
    anba_D = anba_section_from_arrays(
        coords, quads, UD_permD,
        fiber_orientation_deg=np.full(n_cells, alpha_deg),
        plane_orientation_deg=np.full(n_cells, 0.0),
    )
    K_D = anba_D["K"]
    report("ANBA D (E2,E3,E1 + fiber=α, plane=0)", K_D)

    # === Mapping E: same as A but try negative angle ===
    anba_E = anba_section_from_arrays(
        coords, quads, UD_CARBON,
        fiber_orientation_deg=np.full(n_cells, 90.0 + alpha_deg),
        plane_orientation_deg=np.full(n_cells, 0.0),
    )
    K_E = anba_E["K"]
    report("ANBA E (fiber=90+α, plane=0)", K_E)

    print()
    print("Pairwise relative error vs b3_secfem on full K (max element-wise):")
    eps = 1e-30

    def relmax(A, B):
        gmean = np.sqrt(np.outer(np.maximum(np.abs(np.diag(B)), eps),
                                 np.maximum(np.abs(np.diag(B)), eps)))
        gmean[gmean == 0] = 1.0
        return float(np.max(np.abs(A - B) / gmean))

    for label, K in [("ANBA A", K_A), ("ANBA B", K_B), ("ANBA C", K_C),
                     ("ANBA D", K_D), ("ANBA E", K_E)]:
        print(f"  {label:<10}  vs b3:  {100*relmax(K, K_b3):>7.3f}%   "
              f"vs gxbeam: {100*relmax(K, K_gx):>7.3f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
