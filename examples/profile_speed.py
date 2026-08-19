#!/usr/bin/env python3
"""Invsec-oriented cold/warm solve profile for fenicsx (gamg/lu/ilu) vs mfem.

Each configuration runs in a fresh subprocess so *import* and first-solve
(FFCx JIT / factor setup) are visible — that matches spawn workers' first job.
Warm numbers inside the same process approximate later jobs on a reused worker.

Run:
    micromamba run -n b3secfem python examples/profile_speed.py
"""

from __future__ import annotations

from b3_secfem.bench import format_profile_table, profile_matrix


def main() -> None:
    print(
        "=== Cold/warm full solve (+ recovery) — rectangle ladder ===\n"
        "import / cold = first job after spawn; warm = min of subsequent solves\n"
    )
    rows = profile_matrix(
        sizes=[(12, 8), (24, 16), (40, 28)],
        linear_solvers=["gamg", "lu", "ilu"],
        recover=True,
        n_warm=2,
    )
    print(format_profile_table(rows))
    print(
        "\nColumns: import [s], solve cold/warm [s], recovery warm [s], "
        "end-to-end warm = solve_warm + recover_warm."
    )


if __name__ == "__main__":
    main()
