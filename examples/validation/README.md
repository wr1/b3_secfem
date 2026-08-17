# validation — two3 convention measurements

On-demand drivers for the SONATA 2/3-axis question. **Not** wired into
`make test`. They do not change secfem behaviour; they record what each
solver does.

Need the `b3secfem` env (dolfinx). ANBA and SONATA are optional extra
environments — the scripts skip or exit 2 when those are missing.

```text
micromamba run -n b3secfem python examples/validation/three_way.py
micromamba run -n b3secfem python examples/validation/plot_three_way.py
micromamba run -n b3secfem python examples/validation/same_problem.py
micromamba run -n b3secfem python examples/validation/two3_probe.py
micromamba run -n b3secfem python examples/validation/sonata_compare.py --synthetic
micromamba run -n b3secfem python examples/validation/gxbeam_ladder.py
micromamba run -n b3secfem python examples/validation/gxbeam_ladder.py --gx-swap23
micromamba run -n b3secfem python examples/validation/afmesh_naca.py
micromamba run -n b3secfem python examples/validation/plot_compare.py
```

| script | what |
|---|---|
| `three_way.py` | Campaign: mapped secfem / ANBA / gxbeam on the ladder + NACA 0018. |
| `plot_three_way.py` | Figures → `public/figures/compare/`. |
| `same_problem.py` | Gate: ANBA `(card, fiber, plane)` → `anba_to_secfem_input` → compare K. |
| `two3_probe.py` | Rectangle + E1≠E2≠E3. Analytic 1/2/3 axes. secfem fenicsx + mfem (separate processes). ANBA if `anba4:latest` Docker is present. |
| `sonata_dump.py` | **SONATA `b3_fix` env.** Export test meshes + solver `TS`/`MM`. Branch: [NLRWindSystems/SONATA@b3_fix](https://github.com/NLRWindSystems/SONATA/tree/b3_fix). |
| `sonata_compare.py` | Replay a dump, or `--synthetic` IEA / Smith–Chopra cards on a rectangle (1:1 vs 2↔3 swap). |
| `gxbeam_ladder.py` | Existing gxbeam cross-check plus E2≠E3 rings. `--anba`, `--mfem`, `--full`. |
| `afmesh_naca.py` | NACA 0018 hollow: glass-biax skin, carbon-UD spar caps, one biax web. 1:1 vs swap23. |
| `plot_compare.py` | Rebuild all comparison figures from the JSON above. |

JSON and PNG land in `out/` (gitignored). `plot_compare.py` also copies the
figures into `notes/mind/two3/` so the report images load. Narrative:
`notes/mind/two3-report.md`.

Force order in every secfem JSON: `[Fx, Fy, Fz, Mx, My, Mz]`.

Figures (colour = mismatch *source*, not solver name): `mismatch_map.png`,
`mismatch_sources.png`, `source_23.png`, `source_gxbeam.png`, `two3_axes.png`,
`two3_ea.png`, `afmesh_naca.png`.

Findings: `notes/mind/two3-report.md`.
