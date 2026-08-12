"""Pydantic input schemas for b3_secfem."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .materials import IsotropicMaterial, Material, OrthotropicMaterial


class RegionMat(BaseModel):
    """Material + orientation assigned to a tagged region of the mesh."""

    material: Material
    beta_deg: float = Field(0.0, description="Fibre angle about beam axis z [deg]")
    alpha_deg: float = Field(0.0, description="Ply tilt about new x' axis [deg]")


class SectionInput(BaseModel):
    """Top-level input for a single cross-section solve.

    The ``region_materials`` and ``per_cell_*`` specification modes are
    mutually exclusive: supply exactly one of them (not both). Supplying
    both raises ``ValueError`` at model construction time:
    "SectionInput: provide either region_materials OR per_cell_material, not both".

    1. Region-based (the common case): ``region_materials`` maps integer
       region tag -> RegionMat. The mesh is expected to carry per-cell
       region tags.
    2. Per-cell arrays (gxbeam-style): pass lists/arrays of length n_cells
       in the ``per_cell_*`` fields.

    Mode is chosen at solve time by which (exactly one) set of fields is
    populated.

    The ``backend`` field selects the FEM engine (default "fenicsx" for
    dolfinx/UFL/PETSc; "mfem" for PyMFEM serial). Both produce numerically
    equivalent K/M/centres/recovery fields on the same input.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mesh_path: Path = Field(
        ..., description="XDMF mesh file (or VTU via mesh.from_gxbeam_vtu)"
    )
    degree: int = Field(2, ge=1, le=3, description="CG polynomial degree")
    backend: Literal["fenicsx", "mfem"] = Field(
        "fenicsx",
        description="FEM backend: 'fenicsx' (default, requires dolfinx) or 'mfem' (PyMFEM)",
    )
    # fenicsx-only: how to invert the singular in-plane operator E.
    # Medium/small invsec sections often pay more for GAMG *setup* than for the
    # few CG iterations; "lu" (direct, with a tiny shift for the 4-D kernel) and
    # "ilu" (cheap preconditioner) are better throughput defaults for that size.
    # mfem always uses a bordered KKT factorisation and ignores this field.
    linear_solver: Literal["gamg", "lu", "ilu"] = Field(
        "gamg",
        description=(
            "fenicsx linear solver for E: 'gamg' (CG+GAMG; auto-switches to "
            "'lu' when ndof < 20000 unless B3_SECFEM_FORCE_GAMG=1), "
            "'lu' (direct, shifted), or 'ilu' (CG+ILU). Ignored by mfem."
        ),
    )

    region_materials: dict[int, RegionMat] | None = None

    per_cell_material: list[Material] | None = None
    per_cell_beta_deg: np.ndarray | None = None
    per_cell_alpha_deg: np.ndarray | None = None

    @model_validator(mode="after")
    def _check_inputs(self) -> SectionInput:
        a = self.region_materials is not None
        b = self.per_cell_material is not None
        if a and b:
            msg = "SectionInput: provide either region_materials OR per_cell_material, not both"
            raise ValueError(msg)
        if not (a or b):
            msg = "must supply either region_materials or per_cell_material"
            raise ValueError(msg)
        if b:
            n = len(self.per_cell_material)
            for arr_name in ("per_cell_beta_deg", "per_cell_alpha_deg"):
                arr = getattr(self, arr_name)
                if arr is None:
                    continue
                if arr.shape != (n,):
                    msg = f"{arr_name} shape {arr.shape} does not match n_cells={n}"
                    raise ValueError(msg)
        return self


def materials_from_b3_mat(mat: Any) -> Material:
    """Convert a b3_mat material to its b3_secfem twin.

    Prefer ``isinstance`` against real ``b3_mat`` classes when the package is
    importable; fall back to class-name matching so ``b3_mat`` stays optional.
    """
    try:
        from b3_mat import IsotropicMaterial as B3Iso
        from b3_mat import OrthotropicMaterial as B3Ortho
    except ImportError:
        B3Iso = B3Ortho = ()  # type: ignore[misc, assignment]

    if B3Iso and isinstance(mat, B3Iso):
        return IsotropicMaterial(E=mat.E, nu=mat.nu, rho=mat.rho, name=mat.name)
    if B3Ortho and isinstance(mat, B3Ortho):
        return OrthotropicMaterial.from_b3_mat(mat)

    # Name fallback for stubs / when b3_mat is not installed
    cls_name = type(mat).__name__
    if cls_name == "IsotropicMaterial":
        return IsotropicMaterial(E=mat.E, nu=mat.nu, rho=mat.rho, name=mat.name)
    if cls_name == "OrthotropicMaterial":
        return OrthotropicMaterial.from_b3_mat(mat)
    msg = f"unsupported b3_mat material type: {cls_name}"
    raise TypeError(msg)
