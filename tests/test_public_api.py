"""Public package surface — exports that callers rely on without FEM installs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import b3_secfem
from b3_secfem import SectionResult, materials_from_b3_mat, write_quad_xdmf
from b3_secfem.cli import main
from b3_secfem.materials import IsotropicMaterial as SecIso
from b3_secfem.materials import OrthotropicMaterial as SecOrtho


def test_section_result_in_all_and_importable():
    assert "SectionResult" in b3_secfem.__all__
    assert b3_secfem.SectionResult is SectionResult


def test_mesh_utils_in_all():
    for name in ("write_quad_xdmf", "cell_centroids", "solver_cell_tags"):
        assert name in b3_secfem.__all__
        assert callable(getattr(b3_secfem, name))


def test_write_quad_xdmf_from_package_root(tmp_path: Path):
    """Public write_quad_xdmf works without importing dolfinx."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    quads = np.array([[0, 1, 2, 3]], dtype=np.int32)
    tags = np.array([1], dtype=np.int32)
    out = tmp_path / "sec.xdmf"
    fixed = write_quad_xdmf(out, coords, quads, cell_tags=tags)
    assert out.is_file()
    assert fixed.shape == (1, 4)


def test_materials_from_b3_mat_name_fallback_iso():
    class IsotropicMaterial:  # noqa: N801 — mirrors b3_mat type name
        pass

    mat = IsotropicMaterial()
    mat.E = 70e9
    mat.nu = 0.33
    mat.rho = 2700.0
    mat.name = "alu"
    out = materials_from_b3_mat(mat)
    assert isinstance(out, SecIso)
    assert out.E == 70e9
    assert out.nu == 0.33


def test_materials_from_b3_mat_name_fallback_ortho():
    class OrthotropicMaterial:  # noqa: N801
        pass

    mat = OrthotropicMaterial()
    mat.Ex = 140e9
    mat.Ey = 10e9
    mat.Ez = 10e9
    mat.Gxy = 5e9
    mat.Gxz = 5e9
    mat.Gyz = 3.5e9
    mat.nuxy = 0.3
    mat.nuxz = 0.3
    mat.nuyz = 0.4
    mat.rho = 1600.0
    mat.name = "ud"
    out = materials_from_b3_mat(mat)
    assert isinstance(out, SecOrtho)
    assert out.E1 == 140e9
    assert out.G12 == 5e9


def test_materials_from_b3_mat_rejects_unknown():
    class NotAMat:
        pass

    with pytest.raises(TypeError, match="unsupported"):
        materials_from_b3_mat(NotAMat())


def test_cli_backend_flag_in_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--backend" in help_text
    assert "fenicsx" in help_text
    assert "mfem" in help_text


def test_cli_parses_backend(monkeypatch, tmp_path: Path):
    """--backend is accepted and passed through to solve without running FEM."""
    spec = {
        "mesh_path": "x.xdmf",
        "region_materials": {
            "1": {"material": {"type": "isotropic", "E": 1e9, "nu": 0.3, "rho": 1000}}
        },
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))

    captured: dict = {}

    def fake_solve(inp, backend=None):
        captured["backend_kw"] = backend
        captured["inp_backend"] = inp.backend
        return SimpleNamespace(
            K=np.eye(6),
            shear_center=(0.0, 0.0),
            tension_center=(0.0, 0.0),
            elastic_center=(0.0, 0.0),
            mass_center=(0.0, 0.0),
        )

    # main does `from . import solve` — look up package attribute at call time
    monkeypatch.setattr(b3_secfem, "solve", fake_solve)

    rc = main([str(p), "--backend", "mfem"])
    assert rc == 0
    assert captured.get("backend_kw") == "mfem"


def test_material_and_strain_fields_import_from_root():
    from b3_secfem import Material, StrainField, UnitLoadStrainField
    from b3_secfem.materials import IsotropicMaterial, OrthotropicMaterial

    assert "Material" in b3_secfem.__all__
    assert "StrainField" in b3_secfem.__all__
    assert "UnitLoadStrainField" in b3_secfem.__all__
    assert b3_secfem.Material is Material
    assert Material == IsotropicMaterial | OrthotropicMaterial
    assert StrainField is b3_secfem.StrainField
    assert UnitLoadStrainField is b3_secfem.UnitLoadStrainField


def test_strain_fields_reject_non_ndarray():
    """Pydantic type-check: list is not np.ndarray (arbitrary_types_allowed)."""
    from pydantic import ValidationError
    from b3_secfem import StrainField, UnitLoadStrainField

    with pytest.raises(ValidationError):
        StrainField(epsilon=[0], sigma=[0], cell_areas=[0])
    with pytest.raises(ValidationError):
        UnitLoadStrainField(epsilon=[0], sigma=[0], cell_areas=[0])

