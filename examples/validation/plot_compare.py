#!/usr/bin/env python3
"""Rebuild comparison figures from the JSON the solvers already wrote.

    micromamba run -n b3secfem python examples/validation/plot_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent.parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from validation._common import OUT  # noqa: E402
from validation._viz import (  # noqa: E402
    plot_gxbeam,
    plot_overview,
    plot_sonata,
    plot_two3,
    plot_afmesh_k,
)


def _load(name: str):
    p = OUT / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    two3 = _load("two3_probe.json")
    sonata = _load("sonata_compare.json")
    gx_raw = _load("gxbeam_ladder.json")
    gx_swap = _load("gxbeam_ladder_swap23.json")
    afmesh = _load("afmesh_naca.json")

    written: list[Path] = []
    if two3:
        written.extend(plot_two3(two3))
        print("two3 figures ok")
    else:
        print("skip two3 (no JSON)")

    if sonata:
        p = plot_sonata(sonata)
        if p:
            written.append(p)
        print("sonata figure ok")
    else:
        print("skip sonata (no JSON)")

    if gx_raw:
        p = plot_gxbeam(gx_raw, gx_swap)
        if p:
            written.append(p)
        print("gxbeam figure ok")
    else:
        print("skip gxbeam (no JSON)")

    if afmesh:
        p = plot_afmesh_k(afmesh)
        if p:
            written.append(p)
        print("afmesh K figure ok")
    else:
        print("skip afmesh (no JSON)")

    written.append(plot_overview(two3, sonata, gx_raw, gx_swap, afmesh))
    print("overview ok")

    notes_fig = OUT.parents[2] / "notes" / "mind" / "two3"
    if notes_fig.parent.is_dir():
        import shutil

        notes_fig.mkdir(parents=True, exist_ok=True)
        names = {
            "two3_axes.png",
            "two3_ea.png",
            "mismatch_map.png",
            "mismatch_sources.png",
            "source_23.png",
            "source_anba.png",
            "source_gxbeam.png",
            "afmesh_naca.png",
        }
        for name in sorted(names):
            src = OUT / name
            if src.exists():
                dest = notes_fig / name
                shutil.copy2(src, dest)
                print(f"  copied {dest}")

    for p in written:
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
