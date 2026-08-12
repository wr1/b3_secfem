"""CLI flags and machine-readable output (no FEM required)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import b3_secfem
from b3_secfem.cli import _result_to_json, main


def _fake_result(**overrides):
    base = dict(
        K=np.eye(6) * 2.0,
        M=np.eye(6) * 0.5,
        shear_center=(0.1, 0.2),
        tension_center=(0.0, 0.0),
        elastic_center=(0.05, 0.05),
        mass_center=(0.03, 0.04),
        backend="fenicsx",
        K_section_xy=1.23e6,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _write_spec(tmp_path: Path) -> Path:
    spec = {
        "mesh_path": "x.xdmf",
        "region_materials": {
            "1": {"material": {"type": "isotropic", "E": 1e9, "nu": 0.3, "rho": 1000}}
        },
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    return p


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--json" in out or "-j" in out
    assert "--backend" in out


def test_cli_missing_spec_nonzero():
    rc = main(["/nonexistent/path/spec.json"])
    assert rc == 1


def test_cli_json_output(monkeypatch, tmp_path: Path, capsys):
    p = _write_spec(tmp_path)
    monkeypatch.setattr(b3_secfem, "solve", lambda inp, backend=None: _fake_result())
    rc = main([str(p), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["K"]) == 6
    assert len(data["K"][0]) == 6
    assert data["K"][0][0] == 2.0
    assert data["M"][1][1] == 0.5
    assert data["shear_center"] == [0.1, 0.2]
    assert data["backend"] == "fenicsx"
    assert data["K_section_xy"] == 1.23e6


def test_cli_json_short_flag(monkeypatch, tmp_path: Path, capsys):
    p = _write_spec(tmp_path)
    monkeypatch.setattr(
        b3_secfem, "solve", lambda inp, backend=None: _fake_result(backend="mfem")
    )
    rc = main([str(p), "-j", "--backend", "mfem"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["backend"] == "mfem"


def test_result_to_json_numeric_only():
    res = _fake_result()
    res.mesh = object()  # must not appear in payload
    payload = _result_to_json(res)
    assert "mesh" not in payload
    assert "u_solutions" not in payload
    assert set(payload) >= {
        "K",
        "M",
        "shear_center",
        "tension_center",
        "elastic_center",
        "mass_center",
        "backend",
    }
