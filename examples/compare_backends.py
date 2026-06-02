#!/usr/bin/env python3
"""Cross-check and time the fenicsx vs mfem backends.

Three sections:
  1. E-operator agreement (permutation-invariant: sorted spectrum / trace / null
     space), since dolfinx and mfem number DOFs differently.
  2. Full-solve agreement: the complete 6x6 K, mass M, K_xy and section centres
     from the two independent chain-solve implementations.
  3. An E-assembly timing table across mesh sizes.

Run:
    uv run --no-sync python examples/compare_backends.py

Needs both dolfinx and mfem importable in the same environment.
"""

from __future__ import annotations

from b3_secfem.bench import (
    format_timing_table,
    run_comparison,
    run_full_comparison,
    time_backends,
)


def main() -> None:
    print("=== 1. E-operator agreement: fenicsx vs mfem ===\n")
    for mat_key in ("iso", "ortho"):
        res = run_comparison(mat_key, nx=16, ny=10)
        c = res["compare"]
        print(f"[{mat_key}] mesh {res['fenicsx']['n_cells']} cells, "
              f"ndof {res['fenicsx']['shape'][0]}")
        print(f"    sorted-spectrum max rel diff : {c['spectrum_max_reldiff']:.2e}")
        print(f"    trace rel diff               : {c['trace_reldiff']:.2e}")
        print(f"    null-space dim (fenicsx/mfem): {c['nulldim_a']} / {c['nulldim_b']}")
        print()

    print("=== 2. Full-solve agreement (K, M, K_xy, centres) ===\n")
    for mat_key in ("iso", "ortho"):
        res = run_full_comparison(mat_key, nx=12, ny=8)
        c = res["compare"]
        print(f"[{mat_key}] solve time fenicsx/mfem: "
              f"{res['fenicsx']['solve_s']:.3f}s / {res['mfem']['solve_s']:.3f}s")
        print(f"    K   max rel diff : {c['K_rel']:.2e}")
        print(f"    M   max rel diff : {c['M_rel']:.2e}")
        print(f"    K_xy    rel diff : {c['K_section_xy_rel']:.2e}")
        print(f"    shear-centre Δ   : {c['shear_center_absdiff']:.2e}")
        print()

    print("=== 3. E-assembly timing (min of 3 runs, assembly call only) ===\n")
    rows = time_backends(repeats=3)
    print(format_timing_table(rows))


if __name__ == "__main__":
    main()
