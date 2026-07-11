"""Material data models and 6x6 stiffness builders.

Voigt convention used throughout the package:
    sigma = (s11, s22, s33, s23, s13, s12)
    epsilon = (e11, e22, e33, 2*e23, 2*e13, 2*e12)   (engineering shears)

Material principal frame: axis 1 = fibre direction. With (beta, alpha) = (0, 0),
axis 1 lines up with the beam axis z.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class IsotropicMaterial(BaseModel):
    """Isotropic 3D material."""

    E: float = Field(..., gt=0, description="Young's modulus [Pa]")
    nu: float = Field(..., ge=0.0, lt=0.5, description="Poisson's ratio")
    rho: float = Field(0.0, ge=0.0, description="Density [kg/m^3]")
    name: str | None = None

    def C_local(self) -> np.ndarray:
        """6x6 stiffness in the (unrotated) principal frame, Voigt order."""
        E, nu = self.E, self.nu
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        return C


class OrthotropicMaterial(BaseModel):
    """Orthotropic 3D material in (1, 2, 3) principal frame.

    Axis 1 = fibre. With the default (beta, alpha) = (0, 0), axis 1 points
    along the beam axis z.
    """

    E1: float = Field(..., gt=0, description="Young's modulus, fibre [Pa]")
    E2: float = Field(..., gt=0, description="Young's modulus, transverse-2 [Pa]")
    E3: float = Field(..., gt=0, description="Young's modulus, transverse-3 [Pa]")
    G12: float = Field(..., gt=0, description="Shear modulus 1-2 [Pa]")
    G13: float = Field(..., gt=0, description="Shear modulus 1-3 [Pa]")
    G23: float = Field(..., gt=0, description="Shear modulus 2-3 [Pa]")
    nu12: float = Field(..., description="Major Poisson 1-2")
    nu13: float = Field(..., description="Major Poisson 1-3")
    nu23: float = Field(..., description="Major Poisson 2-3")
    rho: float = Field(0.0, ge=0.0, description="Density [kg/m^3]")
    name: str | None = None

    def C_local(self) -> np.ndarray:
        """6x6 orthotropic stiffness in (1,2,3) frame, Voigt order.

        Built by inverting the 3x3 compliance for normal stresses then
        appending shear blocks.
        """
        E1, E2, E3 = self.E1, self.E2, self.E3
        nu12, nu13, nu23 = self.nu12, self.nu13, self.nu23
        nu21 = nu12 * E2 / E1
        nu31 = nu13 * E3 / E1
        nu32 = nu23 * E3 / E2

        S_n = np.array(
            [
                [1.0 / E1, -nu21 / E2, -nu31 / E3],
                [-nu12 / E1, 1.0 / E2, -nu32 / E3],
                [-nu13 / E1, -nu23 / E2, 1.0 / E3],
            ]
        )
        C_n = np.linalg.inv(S_n)

        C = np.zeros((6, 6))
        C[:3, :3] = C_n
        C[3, 3] = self.G23
        C[4, 4] = self.G13
        C[5, 5] = self.G12
        return C

    @classmethod
    def from_b3_mat(cls, mat) -> OrthotropicMaterial:
        """Adapt a b3_mat.OrthotropicMaterial.

        b3_mat uses (x, y, z) names with z out-of-plane. We map x->1, y->2,
        z->3 so the principal frame aligns with how the user authored it.
        Caller is responsible for the (beta, alpha) needed to bring axis 1
        onto the beam axis.
        """
        return cls(
            E1=mat.Ex,
            E2=mat.Ey,
            E3=mat.Ez,
            G12=mat.Gxy,
            G13=mat.Gxz,
            G23=mat.Gyz,
            nu12=mat.nuxy,
            nu13=mat.nuxz,
            nu23=mat.nuyz,
            rho=mat.rho,
            name=mat.name,
        )


Material = IsotropicMaterial | OrthotropicMaterial
