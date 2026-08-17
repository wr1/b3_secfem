# b3_secfem

[![CI](https://github.com/wr1/b3_secfem/actions/workflows/ci.yml/badge.svg)](https://github.com/wr1/b3_secfem/actions/workflows/ci.yml)
[![Release](https://github.com/wr1/b3_secfem/actions/workflows/release.yml/badge.svg)](https://github.com/wr1/b3_secfem/actions/workflows/release.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
Machine-readable numeric output: `b3_secfem spec.json --json` (or `-j`) —
K, M, centres, backend on stdout; default remains the Rich human table.

## Conventions

Same zero as ANBA. Details: `docs/conventions/` (`dockb` → `/docs`).

- Section: **`sec_1, sec_2, sec_3` ≡ `x, y, z`**. Geometry in `(x, y)`; `z` is the RH normal / beam axis.
- Material card: **`mat_1` = fibre**, **`mat_2` = in-ply transverse**, **`mat_3` = ply normal**.
- Placement `(beta_deg, alpha_deg)`, `R = Rz(β) · Ry(α)`:
  - `(0, 0)` is the identity: `mat_1 → +x`, `mat_2 → +y`, `mat_3 → +z`.
  - `α = 90` → fibre along `−z` (beam). `α = 0` → fibre in the section plane; `β` is its angle from `+x`.
- Force order: **`[Fx, Fy, Fz, Mx, My, Mz]`**.
- SONATA / ANBA drop-in: same card, `anba_to_secfem_input(card, fiber, plane)` (`β = plane`, `α = fiber`). Do not rewrite E2/E3.

## Backends

```python
res = solve(inp)                 # fenicsx (default)
res = solve(inp, backend="mfem") # cross-check engine
```

| Backend | Role |
|---------|------|
| **`fenicsx`** | Production path: full `SectionResult` (K, M, centres, recovery). PETSc null-space on the in-plane operator. |
| **`mfem`** | Independent PyMFEM serial engine for validation (~1e-11 on K/M/centres vs fenicsx). Bulk/numba assembly by default when numba is installed. Multi-grid XDMF / some recovery paths still limited — see backend module docstring. |

`mfem` is a core dependency (`scipy` for the bordered KKT solve). Prefer fenicsx for real multi-region sections and invsec.

### Performance

Many medium jobs (e.g. surrogate sweeps) care about **warm** wall time and
**not** paying FFCx JIT per design:

- Prefer **`linear_solver="lu"`** on medium meshes (default CG+GAMG is setup-heavy).
- **One long-lived spawn pool**, `prepare_env` / `warm_up` once per worker, shared
  `XDG_CACHE_HOME` (see **[SKILL.md](SKILL.md)**).
- Profile: `examples/profile_speed.py`, `examples/compare_assemble_speed.py`,
  `notes/mind/speed.md`.

### Cross-backend checks

`b3_secfem.bench` — `run_comparison` (E spectrum/nullspace), `run_full_comparison`
(K/M/centres), `time_backends` / `profile_*`. Each backend in its own subprocess
(dolfinx + mfem teardown is unsafe in one process). See
`tests/test_backend_comparison.py`, `examples/compare_backends.py`.

## Develop / CI

```bash
pre-commit install          # ruff lint + format on commit
make lint && make format    # same tools via Makefile
make test-pure              # no dolfinx (matches CI unit job)
make test                   # full suite (needs fenicsx env — see Makefile)
```

Release: tag `v*` (e.g. `git tag v0.1.0 && git push origin v0.1.0`) → GitHub
Actions builds sdist/wheel and creates a Release. PyPI publish is optional
(OIDC stub in `.github/workflows/release.yml`).

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
