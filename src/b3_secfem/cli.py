"""Command-line entry point.

Usage:
    b3_secfem path/to/spec.json
    b3_secfem path/to/spec.json --backend mfem
    b3_secfem path/to/spec.json --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from .config import RegionMat, SectionInput
from .materials import IsotropicMaterial, OrthotropicMaterial


def _build_input(spec: dict) -> SectionInput:
    """Parse a JSON spec into a SectionInput.

    JSON schema (region-based)::
        {
          "mesh_path": "section.xdmf",
          "degree": 2,
          "backend": "fenicsx",
          "region_materials": {
            "1": {
              "material": {"type": "isotropic", "E": 1e9, "nu": 0.3, "rho": 1000},
              "beta_deg": 0.0,
              "alpha_deg": 0.0
            }
          }
        }
    """
    rm: dict[int, RegionMat] = {}
    for k, entry in spec.get("region_materials", {}).items():
        m = entry["material"]
        if m.get("type") == "orthotropic":
            mat = OrthotropicMaterial(**{kk: v for kk, v in m.items() if kk != "type"})
        else:
            mat = IsotropicMaterial(**{kk: v for kk, v in m.items() if kk != "type"})
        rm[int(k)] = RegionMat(
            material=mat,
            beta_deg=entry.get("beta_deg", 0.0),
            alpha_deg=entry.get("alpha_deg", 0.0),
        )
    inp = SectionInput(
        mesh_path=Path(spec["mesh_path"]),
        degree=spec.get("degree", 2),
        region_materials=rm,
    )
    if "backend" in spec:
        # pydantic will validate the Literal
        inp = inp.model_copy(update={"backend": spec["backend"]})
    return inp


def _print_K(K: np.ndarray, console: Console) -> None:
    table = Table(title="K (6x6) [Fx, Fy, Fz, Mx, My, Mz]")
    for label in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
        table.add_column(label, justify="right")
    for row in K:
        table.add_row(*[f"{v:.3e}" for v in row])
    console.print(table)


def _as_xy(pair: Any) -> list[float]:
    return [float(pair[0]), float(pair[1])]


def _result_to_json(res: Any) -> dict[str, Any]:
    """Numeric-only payload for scripts/CI — no FEM mesh/function objects."""
    payload: dict[str, Any] = {
        "K": np.asarray(res.K, dtype=float).tolist(),
        "M": np.asarray(res.M, dtype=float).tolist(),
        "shear_center": _as_xy(res.shear_center),
        "tension_center": _as_xy(res.tension_center),
        "elastic_center": _as_xy(res.elastic_center),
        "mass_center": _as_xy(res.mass_center),
        "backend": getattr(res, "backend", "fenicsx"),
    }
    k_xy = getattr(res, "K_section_xy", None)
    if k_xy is not None:
        payload["K_section_xy"] = float(k_xy)
    return payload


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(
        description="b3_secfem cross-section solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "JSON may set \"backend\" (fenicsx|mfem). "
            "CLI --backend overrides the JSON value.\n"
            "\n"
            "Minimal JSON spec:\n"
            "  {\n"
            '    "mesh_path": "section.xdmf",\n'
            '    "region_materials": {\n'
            '      "1": {"material": {"type": "isotropic", "E": 1e9, "nu": 0.3, "rho": 1000}}\n'
            "    }\n"
            "  }\n"
            "\n"
            "Machine output: --json / -j writes K, M, centres, backend to stdout\n"
            "(stable contract for scripts; default remains the Rich human table)."
        ),
    )
    parser.add_argument("spec", type=Path, help="JSON input spec")
    parser.add_argument(
        "--backend",
        choices=("fenicsx", "mfem"),
        default=None,
        help="FEM backend (overrides JSON \"backend\" if set)",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Emit numeric K/M/centres/backend as JSON on stdout (no Rich)",
    )
    args = parser.parse_args(argv)

    if not args.spec.is_file():
        print(f"error: spec not found: {args.spec}", file=sys.stderr)
        return 1

    with args.spec.open() as f:
        spec = json.load(f)
    inp = _build_input(spec)

    from . import solve

    res = solve(inp, backend=args.backend)

    if args.json:
        json.dump(_result_to_json(res), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    console = Console()
    _print_K(res.K, console)
    console.print(f"Shear centre:   {res.shear_center}")
    console.print(f"Tension centre: {res.tension_center}")
    console.print(f"Elastic centre: {res.elastic_center}")
    console.print(f"Mass centre:    {res.mass_center}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
