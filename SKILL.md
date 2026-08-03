---
name: b3-secfem
description: >
  Use when running, batching, or integrating b3_secfem section solves (K/M,
  unit-load strains), especially for b3_invsec dataset sweeps, adaptive refine,
  or end-of-opt / surrogate multi-section validation — always fenicsx + batched
  pool — or when tuning fenicsx performance, JIT/cold-start, linear_solver, or
  process pools.
---

# b3_secfem — production use (fenicsx)

## Principle

**fenicsx is the main path.** Optimize *how* you run it (warm workers, direct
LU, one long-lived pool). Do not reach for C++/GPU unless a profiled warm
solve is still the bottleneck after the checklist below.

## Performance facts (measured)

Rectangle ladder, fresh spawn subprocess per config
(`examples/profile_speed.py`). Warm ≈ reused worker after first solve.

| mesh | path | cold solve | warm solve | e2e warm (+recover) |
|------|------|------------|------------|---------------------|
| 40×28 | fenicsx **lu** | ~2.5 s | **~0.5 s** | **~0.8 s** |
| 40×28 | fenicsx gamg | ~3.3 s | ~1.4 s | ~1.6 s |
| 40×28 | mfem pure-Python | ~4.7 s | ~4.1 s | ~5.9 s |
| small | fenicsx cold vs warm | cold pays **~+2 s** FFCx/first-touch | — | — |

- **Cold start (FFCx JIT)** dominates short sweeps if workers are not warmed.
- **`linear_solver="lu"`** beats default GAMG on medium invsec sections (~2–3× warm solve) with K parity ~1e-11.
- **mfem** avoids JIT but loses badly on medium meshes (Python element loop). Cross-check only; multi-grid XDMF incomplete for real laminates.
- Warm fenicsx+lu is already strong; a future C++ port is only ~1.5–2× e2e, not 10×.

Details: `notes/mind/speed.md`.

## Required runner pattern (batch + avoid JIT)

```text
parent:  prepare_env() / pin OMP=1 / XDG_CACHE_HOME shared
         └─ spawn ProcessPool (ONE pool for entire batch)
              initializer → warm_up(linear_solver="lu")   # pay JIT once
              for design in all_designs:
                  evaluate → solve + recover               # warm path
```

### Do

| Rule | How |
|------|-----|
| One pool for the whole batch | `ProcessPoolExecutor(max_workers=N, mp_context=spawn, initializer=warm)` |
| Reuse workers | Leave `max_tasks_per_child=None` (default) |
| Warm each worker once | `b3_secfem.warm_up(linear_solver="lu", recover=True)` in initializer |
| Share FFCx cache | `prepare_env()` → `XDG_CACHE_HOME` → `…/fenics` |
| Pin threads | `OMP/OPENBLAS/MKL_NUM_THREADS=1` when N_proc > 1 |
| Medium sections | `SectionInput(..., linear_solver="lu")` or `B3_SECFEM_LINEAR_SOLVER=lu` |
| Submit all jobs up front | saturate the pool; no serial “start process per design” |

### Do not

- `python -c` / new interpreter **per design** on the timed path
- `max_tasks_per_child=1` (re-pays JIT every job)
- Default **gamg** for invsec-sized meshes without measuring
- Expect mfem pure-Python to beat warm fenicsx+lu on medium airfoils
- Oversubscribe: N workers × N BLAS threads

### API

```python
from b3_secfem import prepare_env, warm_up, SectionInput, solve

prepare_env(threads=1)          # parent, before spawn pool
# in each worker once:
warm_up(linear_solver="lu", recover=True)

inp = SectionInput(
    mesh_path=...,
    region_materials=...,
    backend="fenicsx",          # production default
    linear_solver="lu",         # invsec medium default
)
res = solve(inp)
```

Env knobs (used by `b3_invsec`):

| Variable | Default (invsec) | Meaning |
|----------|------------------|---------|
| `B3_SECFEM_BACKEND` | `fenicsx` | `fenicsx` \| `mfem` |
| `B3_SECFEM_LINEAR_SOLVER` | `lu` | `lu` \| `gamg` \| `ilu` (fenicsx only) |
| `B3_SECFEM_SKIP_WARMUP` | unset | `1` skips dummy solve (debug) |
| `XDG_CACHE_HOME` | `~/.cache` | FFCx under `$XDG_CACHE_HOME/fenics` |

## b3_invsec integration

`b3_invsec.dataset._evaluate_jobs` already:

1. Calls `_prepare_workers()` (threads + cache + `BACKEND=fenicsx` + `LINEAR_SOLVER=lu`)
2. Uses one spawn `ProcessPoolExecutor` with `initializer=_worker_init` → `warm_up`
3. Submits the full pending job list to that pool
4. Serial path also warms once, then loops designs in-process

### Multi-section = always batch (not only dataset sweeps)

Any path that runs **more than one** section FEM must use the same pool, not a
serial `for design: evaluate_section(...)` loop. That includes:

| Call site | Correct entry |
|-----------|---------------|
| Dataset / DoE / CLI sweep | `build_dataset` / `_evaluate_jobs` |
| Adaptive surrogate refine (infill at optima) | `adaptive.evaluate_designs` → `_evaluate_jobs` |
| End-of-opt / surrogate validation (N optima) | `adaptive.evaluate_designs` (or `_evaluate_jobs`) |
| Training-set builds in examples | `_evaluate_jobs` |

**Single** section (CLI `eval`, one-off smoke, one reference calibrate) may call
`evaluate_section` directly — still inherits `B3_SECFEM_BACKEND=fenicsx` and
`linear_solver=lu` defaults when env is prepared.

### Do not (invsec)

- Serial `for r in optima: evaluate_section(r.design)` for end-of-opt checks
- Spawning a new process / interpreter per station or per NACA
- Switching end-of-opt multi-section runs to `mfem` for “speed” (it is slower
  on medium invsec meshes; fenicsx+lu warm is the production path)

Prefer `b3_invsec dataset …` / `build_dataset` / `adaptive.evaluate_designs` /
`_evaluate_jobs` over ad-hoc loops.

## When to profile

```bash
micromamba run -n b3secfem python examples/profile_speed.py
```

API: `b3_secfem.bench.profile_solve`, `profile_matrix`.

## mfem bulk / numba (optional)

Default mfem assembly uses bulk+numba when installed (`B3_SECFEM_MFEM_ASSEMBLE`):

| value | meaning |
|-------|---------|
| `bulk` / unset | numba if available else numpy bulk |
| `numba` / `numpy` | force engine |
| `python` | original `PyBilinearFormIntegrator` |

Numba kernel ≈ fenicsx on **E assembly alone**; tabulation + chain recovery still
leave full mfem solve slower than warm fenicsx+lu. See `notes/mind/speed.md`
and `examples/compare_assemble_speed.py`.

## Out of scope (for now)

- C++ backend (modest gain vs warm fenicsx+lu; ops win only)
- GPU assembly/LU (assembly possible; sparse LU non-trivial; not the default path)
- MPI domain decomp of one section (wrong granularity — parallelise **jobs**)
