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

`b3_secfem` is built on **FEniCSx (dolfinx ≥ 0.9, default)** or the optional
**MFEM** backend (PyMFEM serial). It supports triangles and quadrilaterals
and is **explicit about fibre and ply directions** at every API surface.

The two backends give numerically equivalent results and are intended for
cross-validation of the formulation. See [Backends](#backends) below.

## Quickstart

```python
from b3_secfem import (
    OrthotropicMaterial, RegionMat, SectionInput, SectionResult, solve,
    write_quad_xdmf,
)

mat = OrthotropicMaterial(
    E1=140e9, E2=10e9, E3=10e9,
    G12=5e9, G13=5e9, G23=3.5e9,
    nu12=0.3, nu13=0.3, nu23=0.4, rho=1600.0,
)

# Optional: build an input XDMF from arrays without dolfinx installed
# write_quad_xdmf("section.xdmf", coords, quads, cell_tags=tags)

inp = SectionInput(
    mesh_path="section.xdmf",
    region_materials={1: RegionMat(material=mat, beta_deg=45.0)},
)
res: SectionResult = solve(inp)
print(res.K)             # 6x6 stiffness, [Fx,Fy,Fz,Mx,My,Mz]
print(res.shear_center)
```

CLI: `b3_secfem spec.json` or `b3_secfem spec.json --backend mfem`
(JSON may also set `"backend"`; the CLI flag overrides).

## Conventions

- Beam axis = **z** (out of section plane).
- Section coordinates = **(x, y)**.
- Generalised force ordering: **`[Fx, Fy, Fz, Mx, My, Mz]`**.
- Rotation sign: standard right-hand rule.
- Material rotation angles `(beta_deg, alpha_deg)`:
  - `(0, 0)` → fibre along beam axis z (axial — typical UD spar plies).
  - `alpha=90` → fibre fully in-plane at angle `beta` from x.

## Backends

The solver is pluggable. Select the engine with `SectionInput.backend` or the
`backend=` kwarg to `solve()`:

```python
res = solve(inp, backend="mfem")   # default is "fenicsx"
```

- **`"fenicsx"`** (default) — full dolfinx / UFL / PETSc path. Produces the
  complete `SectionResult` (K, M, centres, and strain/stress recovery). The
  4-D rigid-body null space of the in-plane operator is projected out with a
  PETSc `MatNullSpace`. For **many medium/small jobs** (e.g. surrogate dataset
  sweeps), set `linear_solver="lu"` on `SectionInput` — direct factorisation
  usually beats the default CG+GAMG setup cost on that size class. Agent and
  runner checklist: **[SKILL.md](SKILL.md)**; numbers in
  `examples/profile_speed.py` and `notes/mind/speed.md`.

- **`"mfem"`** — PyMFEM serial path (`mfem.ser`), an independent assembly and
  FE engine for cross-validation. A single custom `_VoigtFormIntegrator`
  assembles the three section operators (`E` in-plane stiffness, `Cmat` xy–z
  coupling, `Mmat` pure-z); the Morandini two-stage chain then reduces to
  matrix-vector products, the singular systems are solved via a bordered KKT
  factorisation (analogue of the PETSc null-space projection), and
  `K = R S⁻¹ Rᵀ`, `M`, centres and `K_xy` follow from a quadrature pass.
  `solve(backend="mfem")` returns a complete `SectionResult`, validated
  against fenicsx to ~1e-11 relative on K / M / centres / K_xy for isotropic
  and orthotropic sections.

  **Not yet ported:** strain/stress field recovery (`recover_strains`) and the
  input-cell-order remap that `per_cell_material` needs on dolfinx-renumbered
  meshes (region-tagged and uniform sections work).

Install the MFEM backend with `pip install mfem` (or conda); it pulls in
`scipy`, used only for the singular-system linear algebra. The MFEM backend
has fewer binary-compatibility issues than dolfinx on some platforms.

### Cross-backend validation and timing

`b3_secfem.bench` compares the two engines:

- `run_comparison` assembles `E` with both engines on the same mesh and
  compares permutation-invariant quantities (the two engines number global
  DOFs differently, so operators match only up to `P E Pᵀ`): sorted spectrum,
  trace, and the 4-D rigid-body null space.
- `run_full_comparison` runs the complete `solve()` per backend and diffs the
  physical outputs directly (K, M, K_xy, centres — no permutation ambiguity).
- `time_backends` gives an assembly-timing table. The MFEM custom integrator
  is a pure-Python per-element loop: faster than fenicsx on tiny meshes (no
  form compilation) but several× slower on large ones.
- `profile_solve` / `profile_matrix` (and `examples/profile_speed.py`) time
  **import + cold/warm full solve + recovery** in fresh subprocesses — the
  metric that matters for spawn-pool surrogate jobs.

Each backend runs in its own subprocess — importing both dolfinx (PETSc/MPI)
and mfem into one interpreter assembles fine but segfaults at teardown. See
`tests/test_backend_comparison.py` and `examples/compare_backends.py`.

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
