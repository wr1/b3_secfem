"""Freeze MFEM K, M, and centres before the section-math refactor.

The meshes are written with ``write_quad_xdmf`` so the test does not need
dolfinx. Regenerate the fixture with::

    python tests/test_golden_mfem.py --write
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_GOLDEN = Path(__file__).resolve().parent / "data" / "golden_mfem.json"

_STEEL = dict(E=210e9, nu=0.3, rho=7850.0)
_ALU = dict(E=70e9, nu=0.33, rho=2700.0)


def _quads(a: float, b: float, nx: int, ny: int, *, split: bool):
    xs = np.linspace(-a / 2, a / 2, nx + 1)
    ys = np.linspace(-b / 2, b / 2, ny + 1)
    coords = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    quads = []
    tags = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            quads.append([n0, n1, n2, n3])
            tags.append(2 if split and i >= nx // 2 else 1)
    return coords, np.asarray(quads, dtype=np.int64), np.asarray(tags, dtype=np.int32)


def _cases():
    return {
        "rectangle_12x8": dict(
            a=0.2, b=0.1, nx=12, ny=8, split=False, materials={"1": _STEEL}
        ),
        "strip_two_region": dict(
            a=0.2,
            b=0.04,
            nx=12,
            ny=4,
            split=True,
            materials={"1": _STEEL, "2": _ALU},
        ),
    }


def _solve_case(name: str, spec: dict, tmp: Path) -> dict:
    from b3_secfem import IsotropicMaterial, RegionMat, SectionInput, solve
    from b3_secfem.mesh import write_quad_xdmf

    coords, quads, tags = _quads(
        spec["a"], spec["b"], spec["nx"], spec["ny"], split=spec["split"]
    )
    path = tmp / f"{name}.xdmf"
    write_quad_xdmf(path, coords, quads, cell_tags=tags)
    regions = {
        int(tag): RegionMat(material=IsotropicMaterial(**mat))
        for tag, mat in spec["materials"].items()
    }
    res = solve(
        SectionInput(
            mesh_path=path,
            region_materials=regions,
            backend="mfem",
            degree=2,
        )
    )
    return {
        "K": np.asarray(res.K, dtype=float).tolist(),
        "M": np.asarray(res.M, dtype=float).tolist(),
        "tension_center": [float(x) for x in res.tension_center],
        "mass_center": [float(x) for x in res.mass_center],
        "shear_center": [float(x) for x in res.shear_center],
    }


def _write(dest: Path) -> None:
    import tempfile

    payload = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, spec in _cases().items():
            payload[name] = _solve_case(name, spec, tmp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")


def test_golden_mfem_k_m_centres(tmp_path):
    pytest.importorskip("mfem")
    if not _GOLDEN.is_file():
        pytest.skip(f"missing {_GOLDEN}; regenerate with --write")
    golden = json.loads(_GOLDEN.read_text())
    for name, spec in _cases().items():
        got = _solve_case(name, spec, tmp_path)
        ref = golden[name]
        np.testing.assert_allclose(got["K"], ref["K"], rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(got["M"], ref["M"], rtol=1e-10, atol=0.0)
        for key in ("tension_center", "mass_center", "shear_center"):
            np.testing.assert_allclose(got[key], ref[key], atol=1e-10)


if __name__ == "__main__":
    if "--write" not in sys.argv:
        print("usage: python tests/test_golden_mfem.py --write", file=sys.stderr)
        raise SystemExit(2)
    _write(_GOLDEN)
    print(f"wrote {_GOLDEN}")
