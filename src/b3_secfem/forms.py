"""UFL forms for the fenicsx chain solve.

Implementation lives under ``backends.fenicsx.forms``. This module re-exports
the public form helpers for historical imports (e.g. recovery, b3_invsec).
"""

from __future__ import annotations

from .backends.fenicsx.forms import (
    assumed_inplane_shear_voigt,
    chain_rhs_stage1,
    chain_rhs_stage2,
    d0_kinematic,
    inplane_shear_rhs_linear,
    stiffness_bilinear,
    total_strain_voigt,
)
from .kinematics import epsilon_z_voigt, voigt_strain

__all__ = [
    "assumed_inplane_shear_voigt",
    "chain_rhs_stage1",
    "chain_rhs_stage2",
    "d0_kinematic",
    "epsilon_z_voigt",
    "inplane_shear_rhs_linear",
    "stiffness_bilinear",
    "total_strain_voigt",
    "voigt_strain",
]
