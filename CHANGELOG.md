# Changelog

All notable changes to this project are documented here. Format: Keep a Changelog; versions: SemVer (pre-1.0: minor = breaking).

## [0.2.0] - unreleased

### Added
- `b3_secfem.backends.base`: `SectionBackend` protocol, `BackendSpec`, `register_backend(..., replace=False)`, `available_backends()`, `BackendUnavailableError`, `BackendCapabilityError`, `SolverConvergenceError`.
- `b3_secfem.section_math` (pure numpy): `SectionMoments`, `stiffness_from_resultants`, `compliance`, `mass_matrix`, `centres`, `SHEAR_BENDING_PAIR`.
- `b3_secfem.mesh_io`: `SectionMesh`, `read_section_mesh` (meshio single-grid XDMF, dolfinx two-grid XDMF, gxbeam VTU), `write_section_mesh`, `orient_ccw`.
- `b3_secfem.geometry` (public, dolfinx-free section builders incl. `rectangle`).
- `SectionResult.meta` (`SolveMeta`: version, backend, solver requested/used, assembly, dofs, mesh sha256, wall time, warnings), `SectionResult.cell_tags`, `SectionResult.cell_centroids()`, `b3_secfem.__version__`.
- `SectionInput.from_file()` / `to_file()`, `SectionInput.from_gxbeam_vtu()`, `SectionInput.assembly`, JSON-schema support (`b3_secfem schema`), `b3_secfem backends`.
- `recover_inplane_shear_strain(result)` (fenicsx; mfem raises `BackendCapabilityError`).
- Extras: `b3mat`, `viz`. CI: required mfem job, fenicsx (conda-forge) job, coverage gate, `uv lock --check`, Dependabot.

### Changed
- `SectionInput` and materials reject unknown keys (`extra="forbid"`).
- `linear_solver` default is `"auto"` (LU below 20 000 dofs, else GAMG); an explicit `"gamg"` is now honoured on small meshes.
- `SectionResult.elastic_center` returns the tension centre (E33-weighted axial-stiffness centroid). In 0.1.x the same name returned the mass centre.
- CLI: `mesh_path` in a spec file is resolved relative to the spec file; the full `SectionInput` schema is accepted (previously only mesh_path/degree/backend/region_materials); logging is WARNING unless `-v`; errors exit with code 2 without a traceback.
- Both backends read meshes through one reader with one tag rule; tagged meshes written by dolfinx are now readable by the mfem backend.
- `b3_mat` moved from core dependencies to the `b3mat` extra; `h5py` is now a core dependency.
- `viz` no longer switches the global matplotlib backend.

### Deprecated (removed in 0.3.0)
- `SectionResult.elastic_center` (it is the tension centre: use `tension_center`; use `mass_center` for the rho centroid, or `elastic_center_decoupling` for the axial-bending decoupling point).
- `SectionResult` payload kwargs/fields `C_func`, `u_solutions`, `d1_fields`, `mesh`, `original_cell_index`, `mfem_state` (use `state`).
- `cell_centroids(mesh)`, `solver_cell_tags(result, path)`, `mesh.read_xdmf`, `mesh.write_xdmf`, dict-returning `mesh.from_gxbeam_vtu`, `b3_secfem._meshlib`, `post.compute_centres`, `inertia.assemble_mass`, top-level `kinematics`.
- Env vars `B3_SECFEM_FORCE_GAMG`, `B3_SECFEM_MFEM_ASSEMBLE`; material key `type`; `adapters.SecfemPlacement`, `adapters.secfem_from_gxbeam_theta`.

### Fixed
- mfem: out-of-range element attributes raise instead of silently using material 0.
- fenicsx: diverged Krylov solves raise `SolverConvergenceError`; a missing MUMPS is logged and recorded.
- Invalid `B3_SECFEM_MFEM_ASSEMBLE` values raise instead of falling back.
- Compliance `S` uses a condition-number test instead of `|det K| > 1e-30`.
- pre-commit whitespace/EOF hooks now actually run on `src/` and `tests/`.
- Dangling doc references (`notes/…`, `@../../CLAUDE.md`, b3_invsec-only env knobs) removed or re-pointed.

### Removed
- Hand-maintained CI dependency lists and the `continue-on-error` mfem job.
- sdist no longer ships `public/`, `examples/`, `.githooks/`, `uv.lock`.

## [0.1.0] - 2026-09-03
- Initial tagged release (`v0.1.0`, commit 35ab658).
