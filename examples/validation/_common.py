"""Shared materials, meshes, axis helpers, and mapping variants for two3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from b3_secfem import OrthotropicMaterial
from b3_secfem.adapters import secfem_to_anba_angles, swap23
from b3_secfem.rotation3d import _rotation_matrix_3x3, fibre_direction

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
OUT = Path(__file__).resolve().parent / "out"

# Asymmetric ortho: no 2/3 degeneracy. Same numbers as b3_section user-contract.
ASYM = OrthotropicMaterial(
    E1=160e9,
    E2=80e9,
    E3=20e9,
    G12=8e9,
    G13=6e9,
    G23=4e9,
    nu12=0.25,
    nu13=0.30,
    nu23=0.35,
    rho=1500.0,
    name="ortho-asym",
)

# IEA-15 glass_uni / glass_biax (SONATA yaml), SI.
GLASS_UNI = OrthotropicMaterial(
    E1=4.46e10,
    E2=1.7e10,
    E3=1.67e10,
    G12=3.27e9,
    G13=3.48e9,
    G23=3.5e9,
    nu12=0.262,
    nu13=0.35,
    nu23=0.264,
    rho=1940.0,
    name="glass_uni",
)
GLASS_BIAX = OrthotropicMaterial(
    E1=1.11e10,
    E2=1.11e10,
    E3=1.67e10,
    G12=3.27e9,
    G13=3.48e9,
    G23=3.5e9,
    nu12=0.262,
    nu13=0.35,
    nu23=0.264,
    rho=1940.0,
    name="glass_biax",
)

CARBON_UD = OrthotropicMaterial(
    E1=140e9,
    E2=10e9,
    E3=10e9,
    G12=5e9,
    G13=5e9,
    G23=3.5e9,
    nu12=0.30,
    nu13=0.30,
    nu23=0.40,
    rho=1600.0,
    name="carbon_ud",
)

# Smith–Chopra box ply (E2 = E3). N/mm² in the yaml; SI here.
SMITH_CHOPRA = OrthotropicMaterial(
    E1=0.142e12,
    E2=0.979e10,
    E3=0.979e10,
    G12=0.60e10,
    G13=0.60e10,
    G23=0.48e10,
    nu12=0.42,
    nu13=0.42,
    nu23=0.42,
    rho=1000.0,
    name="AS4_3501_6",
)

FORCE_ORDER_SECFEM = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def anba_angles(beta_deg: float, alpha_deg: float) -> tuple[float, float]:
    """secfem (β, α) → ANBA (fiber, plane). See adapters.secfem_to_anba_angles."""
    return secfem_to_anba_angles(beta_deg, alpha_deg)

# gxbeam: [F1=Fz, F2=Fx, F3=Fy, M1=Mz, M2=Mx, M3=My] → secfem order.
_GX_TO_SECFEM = np.array([1, 2, 0, 4, 5, 3])


def from_gxbeam_order(K: np.ndarray) -> np.ndarray:
    """Permute gxbeam [F1, F2, F3, M1, M2, M3] → [Fx, Fy, Fz, Mx, My, Mz]."""
    p = _GX_TO_SECFEM
    K = np.asarray(K, dtype=float)
    return K[np.ix_(p, p)]


def rel_per_dof(a, b) -> np.ndarray:
    """|a−b| / max(|a|, |b|) per entry, floor 1e-30."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)


def sonata_anba_as_secfem(mat: OrthotropicMaterial) -> OrthotropicMaterial:
    """SONATA ``build_mat_library`` permutation, read as secfem (E1,E2,E3).

    SONATA card ``E = [along beam, along perimeter, through thickness]`` is
    sent to ANBA as ``(Exx, Eyy, Ezz) = (E2, E3, E1)``. Feeding that triple
    to secfem as ``(E1, E2, E3)`` is one adapter hypothesis — not a
    recommended user mapping.
    """
    return OrthotropicMaterial(
        E1=mat.E2,
        E2=mat.E3,
        E3=mat.E1,
        G12=mat.G23,
        G13=mat.G13,
        G23=mat.G12,
        nu12=mat.nu23,
        nu13=mat.nu13,
        nu23=mat.nu12,
        rho=mat.rho,
        name=f"{mat.name}_sonata_anba_perm",
    )


def principal_axes(beta_deg: float, alpha_deg: float) -> dict[str, list[float]]:
    """Material 1/2/3 unit vectors in global (x, y, z)."""
    R = _rotation_matrix_3x3(beta_deg, alpha_deg)
    return {
        "axis1_fibre": (R @ np.array([1.0, 0.0, 0.0])).tolist(),
        "axis2": (R @ np.array([0.0, 1.0, 0.0])).tolist(),
        "axis3": (R @ np.array([0.0, 0.0, 1.0])).tolist(),
        "fibre": fibre_direction(beta_deg, alpha_deg).tolist(),
    }


def describe_inplane(axis: np.ndarray, *, tol: float = 1e-8) -> str:
    """One-line role of a unit vector relative to a horizontal (xz) ply."""
    a = np.asarray(axis, dtype=float)
    ax, ay, az = a
    bits = []
    if abs(az) > 1.0 - tol:
        bits.append("beam/z")
    if abs(ax) > 1.0 - tol:
        bits.append("section-x (tangent of a y=const flange)")
    if abs(ay) > 1.0 - tol:
        bits.append("section-y (through-thickness of a y=const flange)")
    if not bits:
        bits.append(f"mixed ({ax:+.3f}, {ay:+.3f}, {az:+.3f})")
    return "; ".join(bits)


def closest_modulus(value: float, mat: OrthotropicMaterial) -> dict[str, Any]:
    """Which of E1, E2, E3 is closest to ``value`` (e.g. EA / A)."""
    cands = {"E1": mat.E1, "E2": mat.E2, "E3": mat.E3}
    name = min(cands, key=lambda k: abs(cands[k] - value))
    ref = cands[name]
    rel = abs(value - ref) / max(abs(ref), 1e-30)
    return {
        "closest": name,
        "value": float(value),
        "E_closest": float(ref),
        "rel_err": float(rel),
        "rel_vs_E1": float(abs(value - mat.E1) / max(abs(mat.E1), 1e-30)),
        "rel_vs_E2": float(abs(value - mat.E2) / max(abs(mat.E2), 1e-30)),
        "rel_vs_E3": float(abs(value - mat.E3) / max(abs(mat.E3), 1e-30)),
    }


def rectangle(width: float, height: float, n_w: int, n_h: int):
    """Axis-aligned rectangle centred at origin. Returns coords, quads, area."""
    xs = np.linspace(-width / 2, width / 2, n_w + 1)
    ys = np.linspace(-height / 2, height / 2, n_h + 1)
    coords = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    quads = []
    for j in range(n_h):
        for i in range(n_w):
            n00 = j * (n_w + 1) + i
            n10 = n00 + 1
            n11 = n10 + (n_w + 1)
            n01 = n00 + (n_w + 1)
            quads.append([n00, n10, n11, n01])
    return coords, np.asarray(quads, dtype=np.int64), float(width * height)


def mat_to_dict(mat: OrthotropicMaterial) -> dict[str, Any]:
    return {
        "name": mat.name,
        "E1": mat.E1,
        "E2": mat.E2,
        "E3": mat.E3,
        "G12": mat.G12,
        "G13": mat.G13,
        "G23": mat.G23,
        "nu12": mat.nu12,
        "nu13": mat.nu13,
        "nu23": mat.nu23,
        "rho": mat.rho,
        "e2_eq_e3": bool(np.isclose(mat.E2, mat.E3)),
    }


def mat_from_dict(d: dict[str, Any]) -> OrthotropicMaterial:
    return OrthotropicMaterial(
        E1=d["E1"],
        E2=d["E2"],
        E3=d["E3"],
        G12=d["G12"],
        G13=d["G13"],
        G23=d["G23"],
        nu12=d["nu12"],
        nu13=d["nu13"],
        nu23=d["nu23"],
        rho=d.get("rho", 0.0),
        name=d.get("name"),
    )


@dataclass(frozen=True)
class Mapping:
    """One way of handing a user material to a solver. Measurement only."""

    name: str
    material: OrthotropicMaterial
    beta_deg: float
    alpha_deg: float
    # ANBA angles if this mapping is also sent to ANBA (None = skip).
    anba_fiber_deg: float | None = None
    anba_plane_deg: float | None = None
    note: str = ""


def default_mappings(user: OrthotropicMaterial) -> list[Mapping]:
    """Variants that distinguish the ranked hypotheses. Do not prefer one."""
    swapped = swap23(user)
    f0, p0 = anba_angles(0.0, 0.0)
    f90, p90 = anba_angles(0.0, 90.0)
    fb90, pb90 = anba_angles(90.0, 0.0)
    return [
        Mapping(
            "secfem_1to1_a0",
            user,
            0.0,
            0.0,
            anba_fiber_deg=f0,
            anba_plane_deg=p0,
            note="1:1 card; (0,0)=I; fibre along +x; EA≈E3·A",
        ),
        Mapping(
            "secfem_1to1_a90",
            user,
            0.0,
            90.0,
            anba_fiber_deg=f90,
            anba_plane_deg=p90,
            note="1:1 card; α=90 fibre on −z; EA≈E1·A",
        ),
        Mapping(
            "secfem_swap23_a0",
            swapped,
            0.0,
            0.0,
            anba_fiber_deg=f0,
            anba_plane_deg=p0,
            note="E2↔E3 then α=0 — measurement only, not the drop-in",
        ),
        Mapping(
            "secfem_swap23_a90",
            swapped,
            0.0,
            90.0,
            anba_fiber_deg=f90,
            anba_plane_deg=p90,
            note="E2↔E3 then α=90 — measurement only",
        ),
        Mapping(
            "secfem_1to1_b90_a0",
            user,
            90.0,
            0.0,
            anba_fiber_deg=fb90,
            anba_plane_deg=pb90,
            note="1:1 card; β=90, α=0 — fibre along +y",
        ),
    ]
