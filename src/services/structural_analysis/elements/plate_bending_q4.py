"""4-düğümlü Mindlin-Reissner plate bending elemanı.

Selective reduced integration ile:
    - Bending kısmı  K_b: 2×2 Gauss (tam integrasyon)
    - Shear kısmı    K_s: 1×1 Gauss (azaltılmış → shear locking'i önler)

Her düğümde 3 DOF: (w, θ_x, θ_y). Lokal 2D düzlemde, eleman normali
boyunca w pozitif. Sınıf 12×12 lokal K üretir.

Konvansiyon (Reddy):
    Kirchhoff limit:    θ_x = -∂w/∂y,   θ_y = ∂w/∂x
    Eğrilikler:         κ_xx = ∂θ_y/∂x
                        κ_yy = -∂θ_x/∂y
                        2κ_xy = ∂θ_y/∂y - ∂θ_x/∂x
    Kayma genlemesi:    γ_xz = ∂w/∂x + θ_y
                        γ_yz = ∂w/∂y - θ_x

Bünye:
    D_b = Eh³/[12(1-ν²)] × [[1,ν,0],[ν,1,0],[0,0,(1-ν)/2]]
    D_s = κ × G × h × I₂        (κ=5/6 shear correction)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..model.dto import MaterialDTO, NodeDTO, ShellSectionDTO

INV = np.linalg.inv
DET = np.linalg.det

_GAUSS_2 = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
_WEIGHTS_2 = [1.0, 1.0]
# Reduced 1x1 integration: single point at origin, weight 2×2=4


def _SF(r: float, s: float) -> np.ndarray:
    return 0.25 * np.asarray([
        (1 - r) * (1 - s),   # tensor n_ll
        (1 + r) * (1 - s),   # n_lr
        (1 - r) * (1 + s),   # n_ul
        (1 + r) * (1 + s),   # n_ur
    ])


def _dSF_dr(r: float, s: float) -> np.ndarray:
    return 0.25 * np.asarray([
        [-1 + s, -1 + r],
        [1 - s, -1 - r],
        [-1 - s, 1 - r],
        [1 + s, 1 + r],
    ])


@dataclass(frozen=True)
class PlateBendingQ4:
    """Mindlin-Reissner plate bending — lokal 2D düzlemde 12×12 K.

    DOF sırası: [w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3, w4, θx4, θy4]
    Düğümler tensor sırası: n_ll, n_lr, n_ul, n_ur.
    """

    nodes_local_xy: np.ndarray       # (4, 2)
    E: float
    nu: float
    thickness: float
    shear_correction: float = 5.0 / 6.0

    @classmethod
    def from_cyclic(
        cls,
        cyclic_xy: np.ndarray,
        E: float,
        nu: float,
        thickness: float,
    ) -> "PlateBendingQ4":
        tensor = np.asarray([cyclic_xy[0], cyclic_xy[1],
                             cyclic_xy[3], cyclic_xy[2]])
        return cls(nodes_local_xy=tensor, E=E, nu=nu, thickness=thickness)

    # ------------------------------------------------------------- bünye
    @property
    def shear_modulus(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    def D_b(self) -> np.ndarray:
        """3×3 bending bünye matrisi."""
        h = self.thickness
        coef = self.E * h ** 3 / (12.0 * (1.0 - self.nu ** 2))
        nu = self.nu
        return coef * np.asarray([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, 0.5 * (1 - nu)],
        ])

    def D_s(self) -> np.ndarray:
        """2×2 shear bünye matrisi."""
        k = self.shear_correction * self.shear_modulus * self.thickness
        return k * np.eye(2)

    # ------------------------------------------------ B matrisleri
    def _XM(self) -> np.ndarray:
        return self.nodes_local_xy.T

    def _jacobian(self, r: float, s: float) -> np.ndarray:
        return self._XM() @ _dSF_dr(r, s)

    def _dN_dxy(self, r: float, s: float) -> np.ndarray:
        """Şekil fonksiyonlarının gerçek koordinatlara türevleri, 4×2."""
        J_inv = INV(self._jacobian(r, s))
        return _dSF_dr(r, s) @ J_inv

    def B_bending(self, r: float, s: float) -> np.ndarray:
        """3×12 bending strain-displacement matrisi.

        κ = [κ_xx, κ_yy, 2κ_xy] = B_b × [w1,θx1,θy1, ..., w4,θx4,θy4]
            κ_xx = ∂θ_y/∂x   → coefficient +dN/dx on θ_y DOFs
            κ_yy = -∂θ_x/∂y  → coefficient -dN/dy on θ_x DOFs
            2κ_xy = ∂θ_y/∂y - ∂θ_x/∂x
        """
        dN = self._dN_dxy(r, s)   # (4, 2)
        B = np.zeros((3, 12))
        for i in range(4):
            dNx = dN[i, 0]
            dNy = dN[i, 1]
            # Node i DOF'ları: w@3i, θx@3i+1, θy@3i+2
            # κ_xx = ∂θ_y/∂x
            B[0, 3 * i + 2] = dNx
            # κ_yy = -∂θ_x/∂y
            B[1, 3 * i + 1] = -dNy
            # 2κ_xy = ∂θ_y/∂y - ∂θ_x/∂x
            B[2, 3 * i + 2] = dNy
            B[2, 3 * i + 1] = -dNx
        return B

    def B_shear(self, r: float, s: float) -> np.ndarray:
        """2×12 shear strain-displacement matrisi.

        γ = [γ_xz, γ_yz] = B_s × u
            γ_xz = ∂w/∂x + θ_y    → +dN/dx on w,  +N on θ_y
            γ_yz = ∂w/∂y - θ_x    → +dN/dy on w,  -N on θ_x
        """
        dN = self._dN_dxy(r, s)
        N = _SF(r, s)
        B = np.zeros((2, 12))
        for i in range(4):
            B[0, 3 * i] = dN[i, 0]       # ∂w/∂x
            B[0, 3 * i + 2] = N[i]       # +θ_y
            B[1, 3 * i] = dN[i, 1]       # ∂w/∂y
            B[1, 3 * i + 1] = -N[i]      # -θ_x
        return B

    # ------------------------------------------------------- rijitlik
    def local_stiffness(self) -> np.ndarray:
        """12×12 plate bending rijitliği (bending + shear)."""
        D_b = self.D_b()
        D_s = self.D_s()

        K = np.zeros((12, 12))
        # Bending: 2×2 tam integrasyon
        for i in range(2):
            for j in range(2):
                r, s = _GAUSS_2[i], _GAUSS_2[j]
                w = _WEIGHTS_2[i] * _WEIGHTS_2[j]
                B_b = self.B_bending(r, s)
                detJ = DET(self._jacobian(r, s))
                K = K + w * B_b.T @ D_b @ B_b * detJ

        # Shear: 1×1 reduced integration (tek merkez noktası, ağırlık 4)
        r, s = 0.0, 0.0
        B_s = self.B_shear(r, s)
        detJ = DET(self._jacobian(r, s))
        K = K + 4.0 * B_s.T @ D_s @ B_s * detJ
        return K


# --------------------------------------------------------------- helper
def build_from_dto(
    nodes_3d: list[NodeDTO],
    section: ShellSectionDTO,
    material: MaterialDTO,
    local_frame: np.ndarray,
    origin: np.ndarray,
) -> PlateBendingQ4:
    """3D düğümleri lokal düzleme iz düşür."""
    local_xy = []
    for n in nodes_3d:
        p = np.asarray([n.x - origin[0], n.y - origin[1], n.z - origin[2]])
        local_xy.append([local_frame[0] @ p, local_frame[1] @ p])
    return PlateBendingQ4.from_cyclic(
        np.asarray(local_xy),
        E=material.E, nu=material.nu, thickness=section.thickness,
    )
