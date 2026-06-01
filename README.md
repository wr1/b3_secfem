# b3_secfem

Lightweight FEniCSx-based 2D cross-section property solver for composite
wing / wind-turbine-blade sections.

Computes the 6×6 cross-section stiffness `K` and mass `M` matrices, plus
shear / tension / elastic centres, from a 2D mesh of the cross-section and
per-element composite material orientation.

Sits alongside `b3_gx` (gxbeam_section, Julia) and `anba4_orig` (legacy
FEniCS) and provides an independent cross-check on both.

## Why

- `gxbeam_section` requires a Julia env and bakes in a non-obvious force
  ordering.
- `anba4_orig` runs on legacy FEniCS 2019 (Docker / pinned conda env), is
  triangle-only, and has a `-sin` rotation sign quirk.

`b3_secfem` is built on **FEniCSx (dolfinx ≥ 0.9, pip-installable)**,
supports triangles and quadrilaterals, and is **explicit about fibre and
ply directions** at every API surface.

## Quickstart

```python
from b3_secfem import (
    OrthotropicMaterial, RegionMat, SectionInput, solve,
)

mat = OrthotropicMaterial(
    E1=140e9, E2=10e9, E3=10e9,
    G12=5e9, G13=5e9, G23=3.5e9,
    nu12=0.3, nu13=0.3, nu23=0.4, rho=1600.0,
)

inp = SectionInput(
    mesh_path="section.xdmf",
    region_materials={1: RegionMat(material=mat, beta_deg=45.0)},
)
res = solve(inp)
print(res.K)             # 6x6 stiffness, [Fx,Fy,Fz,Mx,My,Mz]
print(res.shear_center)
```

## Conventions

- Beam axis = **z** (out of section plane).
- Section coordinates = **(x, y)**.
- Generalised force ordering: **`[Fx, Fy, Fz, Mx, My, Mz]`**.
- Rotation sign: standard right-hand rule.
- Material rotation angles `(beta_deg, alpha_deg)`:
  - `(0, 0)` → fibre along beam axis z (axial — typical UD spar plies).
  - `alpha=90` → fibre fully in-plane at angle `beta` from x.

## v0.1 limitations

The four well-posed cross-section modes — axial (Fz), bending (Mx, My),
torsion (Mz) — are computed by solving for the warping field with an
assumed-strain ansatz on each generalised strain. These match analytic
values to machine precision for isotropic sections.

Transverse shear (Fx, Fy) is obtained from the Stage-2 d₂ warping
solution in the Morandini chain (E d₂ = M d₀ − H d₁, with ε_total for the
shear modes assembled from d₂; see `notes/claude.md`). For simple sections
this reproduces the expected Saint-Venant values — e.g. K[Fx, Fx] =
K[Fy, Fy] = (5/6) G A for an isotropic rectangle (verified to mesh
tolerance in `tests/test_iso_rectangle.py`). Complex airfoils exhibit
mesh-convergence differences versus gxbeam_section on the shear terms (as
on torsion); these are documented in the cross-check driver and notes.

The solver and visualization tools are serial-only (single MPI rank). Per-cell material assignment and recovery assume local cell counts match the input arrays. Parallel execution is not yet supported.

See `notes/claude.md` for project context and `notes/theory.md` for the
mathematical formulation.
