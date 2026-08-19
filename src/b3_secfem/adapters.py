"""Convention maps between secfem, ANBA, and gxbeam.

ANBA ``transformation_matrix(plane, fiber)`` (``sn = −sin``) equals
secfem ``_bond_T(Rz(β) · Ry(α))`` at ``β = plane``, ``α = fiber``.
The card is not rewritten. ``swap23`` remains available as a tool.

SONATA drop-in: take the card and ``(fiber, plane)`` ANBA would get,
then ``anba_to_secfem_input``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import Material, OrthotropicMaterial
from .post import from_gxbeam_order, to_anba_order, to_gxbeam_order

__all__ = [
    "AnbaPlacement",
    "SecfemPlacement",
    "anba_to_secfem_angles",
    "anba_to_secfem_input",
    "secfem_from_gxbeam_theta",
    "secfem_to_anba_angles",
    "secfem_to_anba_input",
    "swap23",
    "from_gxbeam_order",
    "to_anba_order",
    "to_gxbeam_order",
]


@dataclass(frozen=True)
class AnbaPlacement:
    """Card + ANBA (fiber, plane) for one region."""

    material: Material
    fiber_deg: float
    plane_deg: float


@dataclass(frozen=True)
class SecfemPlacement:
    """Card + secfem (β, α) for one region."""

    material: Material
    beta_deg: float
    alpha_deg: float


def secfem_to_anba_angles(beta_deg: float, alpha_deg: float) -> tuple[float, float]:
    """secfem ``(β, α)`` → ANBA ``(fiber, plane)``.

    ``fiber = α``, ``plane = β``. No 90° offset; the 90 that puts the
    fibre on the beam lives in the input (ANBA ``fiber = 90``).
    """
    return float(alpha_deg), float(beta_deg)


def anba_to_secfem_angles(fiber_deg: float, plane_deg: float) -> tuple[float, float]:
    """ANBA ``(fiber, plane)`` → secfem ``(β, α)``."""
    return float(plane_deg), float(fiber_deg)


def secfem_from_gxbeam_theta(theta_rad: float) -> tuple[float, float]:
    """gxbeam per-element ``theta`` → secfem ``(beta_deg, alpha_deg)``.

    gxbeam ``theta`` is the out-of-plane fibre tilt only (``α``). ``β`` is 0.
    """
    return 0.0, float(np.degrees(theta_rad))


def swap23(mat: OrthotropicMaterial) -> OrthotropicMaterial:
    """Relabel principal 2 ↔ 3 (including G and major Poisson)."""
    nu32 = mat.nu23 * mat.E3 / mat.E2
    return OrthotropicMaterial(
        E1=mat.E1,
        E2=mat.E3,
        E3=mat.E2,
        G12=mat.G13,
        G13=mat.G12,
        G23=mat.G23,
        nu12=mat.nu13,
        nu13=mat.nu12,
        nu23=nu32,
        rho=mat.rho,
        name=None if mat.name is None else f"{mat.name}_swap23",
    )


def anba_to_secfem_input(
    mat: Material, fiber_deg: float, plane_deg: float
) -> SecfemPlacement:
    """SONATA / ANBA ``(card, fiber, plane)`` → secfem placement.

    Orthotropic card is forwarded unchanged.
    """
    beta, alpha = anba_to_secfem_angles(fiber_deg, plane_deg)
    return SecfemPlacement(material=mat, beta_deg=beta, alpha_deg=alpha)


def secfem_to_anba_input(
    mat: Material, beta_deg: float, alpha_deg: float
) -> AnbaPlacement:
    """secfem ``(card, β, α)`` → ANBA placement. Inverse of the drop-in."""
    fiber, plane = secfem_to_anba_angles(beta_deg, alpha_deg)
    return AnbaPlacement(material=mat, fiber_deg=fiber, plane_deg=plane)
