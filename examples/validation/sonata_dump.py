#!/usr/bin/env python3
"""Export SONATA test-section meshes + ANBA K/M for later secfem compare.

The live SONATA+secfem adapter is
`https://github.com/NLRWindSystems/SONATA/tree/b3_fix`
(not the stock ANBA `cbm_run_anbax`). Dump from a checkout of that branch::

    git clone -b b3_fix https://github.com/NLRWindSystems/SONATA.git
    # env with pythonocc + dolfinx + editable b3_secfem
    python examples/validation/sonata_dump.py

Writes ``examples/validation/out/sonata/<case>.json`` with nodes, cells,
MatID, theta_1, theta_3, material cards, and solver TS/MM when the solve
succeeds. Does not patch SONATA.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out" / "sonata"
# NLR b3_fix checkout if present; else local clone. Override with SONATA_ROOT.
SONATA_ROOT = Path(os.environ.get("SONATA_ROOT", "/home/wr1/software/SONATA"))

CASES = {
    "beam0": (
        SONATA_ROOT
        / "tests/regression/0_beams"
        / "0_box_beam_HT_antisym_layup_15_6_SI_SmithChopra91.yaml"
    ),
    "iea15": SONATA_ROOT / "tests/regression/1_iea15mw/iea_15_240_rwt.yaml",
    "iea22": SONATA_ROOT / "tests/regression/2_iea22mw/iea_22_280_rwt.yaml",
    "rotated": SONATA_ROOT / "tests/regression/6_box_beam/rotated_beam.yaml",
}


def _mat_card(m) -> dict:
    card = {
        "id": int(m.id),
        "name": getattr(m, "name", None),
        "orth": int(m.orth),
        "rho": float(m.rho),
    }
    if m.orth == 0:
        card["E"] = float(m.E)
        card["nu"] = float(m.nu)
    elif m.orth == 1:
        card["E"] = [float(x) for x in m.E]
        card["G"] = [float(x) for x in m.G]
        card["nu"] = [float(x) for x in m.nu]
        e = card["E"]
        card["e2_eq_e3"] = bool(abs(e[1] - e[2]) / max(abs(e[1]), 1e-30) < 1e-6)
    return card


def _dump_section(x, cs, materials) -> dict:
    mesh, nodes = cs.mesh, None
    try:
        from SONATA.cbm.cbm_utl import sort_and_reassignID

        mesh, nodes = sort_and_reassignID(mesh)
    except Exception:  # noqa: BLE001
        nodes = []
        seen = {}
        for c in mesh:
            for n in c.nodes:
                if n.id not in seen:
                    seen[n.id] = n
        nodes = [seen[k] for k in sorted(seen)]

    node_xy = []
    id_map = {}
    for i, n in enumerate(nodes):
        id_map[n.id] = i
        xy = n.coordinates if hasattr(n, "coordinates") else [n.Pnt2d.X(), n.Pnt2d.Y()]
        node_xy.append([float(xy[0]), float(xy[1])])

    cells, mat_id, th1, th3 = [], [], [], []
    for c in mesh:
        cells.append([id_map[n.id] for n in c.nodes])
        mat_id.append(int(c.MatID))
        t1 = c.theta_1[0] if hasattr(c.theta_1, "__len__") else c.theta_1
        th1.append(float(t1))
        th3.append(float(getattr(c, "theta_3", 0.0) or 0.0))

    out: dict = {
        "grid": float(x),
        "n_nodes": len(node_xy),
        "n_cells": len(cells),
        "node_xy": node_xy,
        "cells": cells,
        "MatID": mat_id,
        "theta_1": th1,
        "theta_3": th3,
        "materials": {str(k): _mat_card(v) for k, v in materials.items()},
    }
    bp = getattr(cs, "BeamProperties", None)
    if bp is not None and getattr(bp, "TS", None) is not None:
        out["ANBA_TS"] = np.asarray(bp.TS, dtype=float).tolist()
        out["ANBA_MM"] = np.asarray(bp.MM, dtype=float).tolist()
        out["ANBA_order"] = ["SONATA/VABS after trsf_sixbysix"]
    return out


def dump_case(
    name: str, yaml_path: Path, *, stations: list[float] | None, run_anba: bool
) -> dict:
    from SONATA.classBlade import Blade

    flags = {
        "flag_wt_ontology": name.startswith("iea") or name == "rotated",
        "flag_ref_axes_wt": name.startswith("iea") or name == "rotated",
        "attribute_str": "MatID",
        "flag_plotDisplacement": False,
        "flag_plotTheta11": False,
        "flag_wf": False,
        "flag_lft": False,
        "flag_topo": False,
        "flag_recovery": False,
        "mesh_resolution": 80 if name.startswith("iea") else 100,
    }
    job = Blade(name=name, filename=str(yaml_path), flags=flags, stations=stations)
    job.blade_gen_section(topo_flag=True, mesh_flag=True)
    if run_anba:
        job.blade_run_anbax()

    sections = []
    for x, cs in job.sections:
        sections.append(_dump_section(x, cs, job.materials))
    return {
        "case": name,
        "yaml": str(yaml_path),
        "n_sections": len(sections),
        "ran_anba": run_anba,
        "sections": sections,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "cases",
        nargs="*",
        default=["beam0"],
        help=f"subset of {list(CASES)} (default: beam0)",
    )
    p.add_argument("--all", action="store_true")
    p.add_argument(
        "--no-anba", action="store_true", help="mesh only (no blade_run_anbax)"
    )
    p.add_argument(
        "--stations",
        type=float,
        nargs="*",
        default=None,
        help="IEA radial stations; default is the yaml / one station",
    )
    args = p.parse_args()

    try:
        import SONATA  # noqa: F401
    except ImportError:
        print("SONATA is not importable in this interpreter.", file=sys.stderr)
        print("Activate the SONATA+ANBA env and re-run.", file=sys.stderr)
        return 2

    names = list(CASES) if args.all else args.cases
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        yaml_path = CASES[name]
        stations = args.stations
        if stations is None and name.startswith("iea"):
            stations = [0.5]
        print(f"dumping {name} from {yaml_path}")
        data = dump_case(name, yaml_path, stations=stations, run_anba=not args.no_anba)
        dest = OUT / f"{name}.json"
        dest.write_text(json.dumps(data))
        print(f"  {data['n_sections']} section(s) -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
