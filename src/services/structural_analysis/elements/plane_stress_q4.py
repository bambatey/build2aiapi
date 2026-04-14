"""4-düğümlü plane stress (membrane) elemanı — SEA_Book Gauss 2×2 portu.

Kaynak: ``sonlu-elemanlar-analizi/SEA_Book/sec4_plane_stress_rectangle_gauss.py``.

Matematik orijinaline sadıktır — characterization testi altın çıktı
(``tests/benchmarks/plane_stress_q4_golden.json``) ile doğrulanır.

API farkı (orijinale göre):
    - Durum: sınıf stateless; DTO'lar parametre olarak verilir.
    - Düğüm sırası: SAP cyclic (n1→n2→n3→n4 köşe köşe dolanır).
      SEA_Book tensor-ordering'e iç dönüşüm yapılır (``_reorder_cyclic``).

Kullanım: lokal 2D eksende 8×8 membrane rijitliği. Üst katman (shell
elemanı) bu K'yı 3D'ye çevirir ve plate bending ile birleştirir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..model.dto import MaterialDTO, NodeDTO, ShellSectionDTO

INV = np.linalg.inv
DET = np.linalg.det

# Gauss 2×2 integrasyon noktaları ve ağırlıkları
_GAUSS = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
_WEIGHTS = [1.0, 1.0]


def _SF(r: float, s: float) -> np.ndarray:
    """Şekil fonksiyonları vektörü. Tensor sırası:
        N1 = (1-r)(1-s)/4    (ξ=-1, η=-1)
        N2 = (1+r)(1-s)/4    (ξ=+1, η=-1)
        N3 = (1-r)(1+s)/4    (ξ=-1, η=+1)
        N4 = (1+r)(1+s)/4    (ξ=+1, η=+1)
    """
    return 0.25 * np.asarray([
        (1 - r) * (1 - s),
        (1 + r) * (1 - s),
        (1 - r) * (1 + s),
        (1 + r) * (1 + s),
    ])


def _dSF_dr(r: float, s: float) -> np.ndarray:
    """Şekil fonksiyonlarının doğal koordinata türev matrisi (4×2).
    Sütunlar: [dN/dr, dN/ds]."""
    return 0.25 * np.asarray([
        [-1 + s, -1 + r],
        [1 - s, -1 - r],
        [-1 - s, 1 - r],
        [1 + s, 1 + r],
    ])


@dataclass(frozen=True)
class PlaneStressQ4:
    """Tek bir Q4 plane stress elemanı — lokal 2D düzlemde membrane rijitliği.

    ``nodes_local_xy``: 4 düğümün lokal 2D koordinatları (4×2).
    Sıra **tensor ordering**: (n_ll, n_lr, n_ul, n_ur) — yani
        n_ll = bottom-left (doğal ξ=-1, η=-1)
        n_lr = bottom-right (ξ=+1, η=-1)
        n_ul = top-left     (ξ=-1, η=+1)
        n_ur = top-right    (ξ=+1, η=+1)

    SAP cyclic'ten dönüştürmek için ``from_cyclic`` yardımcı kurucusunu
    kullan.
    """

    nodes_local_xy: np.ndarray   # (4, 2)
    E: float
    nu: float
    thickness: float

    @classmethod
    def from_cyclic(
        cls,
        cyclic_xy: np.ndarray,
        E: float,
        nu: float,
        thickness: float,
    ) -> "PlaneStressQ4":
        """SAP cyclic sırasından (n1→n2→n3→n4 köşe köşe) kuruluş.

        Cyclic → tensor haritalaması:
            SEA n_ll = cyclic[0]
            SEA n_lr = cyclic[1]
            SEA n_ul = cyclic[3]   (cyclic son köşe, üst-sol)
            SEA n_ur = cyclic[2]
        """
        tensor_xy = np.asarray([
            cyclic_xy[0],
            cyclic_xy[1],
            cyclic_xy[3],
            cyclic_xy[2],
        ])
        return cls(
            nodes_local_xy=tensor_xy,
            E=E, nu=nu, thickness=thickness,
        )

    # ---------------------------------------------------- temel matrisler
    def constitutive_matrix(self) -> np.ndarray:
        """3×3 plane stress bünye matrisi C."""
        E, nu = self.E, self.nu
        return E / (1 - nu ** 2) * np.asarray([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, 0.5 * (1 - nu)],
        ])

    def _XM(self) -> np.ndarray:
        """2×4 düğüm koordinat matrisi (tensor sırası)."""
        return self.nodes_local_xy.T

    def _jacobian(self, r: float, s: float) -> np.ndarray:
        return self._XM() @ _dSF_dr(r, s)

    def _dSF_dx_T(self, r: float, s: float) -> np.ndarray:
        return INV(self._jacobian(r, s)).T @ _dSF_dr(r, s).T

    def strain_displacement(self, r: float, s: float) -> np.ndarray:
        """3×8 genleme-yer değiştirme matrisi B. Sıra:
            [ε_xx, ε_yy, γ_xy]^T = B × [u1,u2,u3,u4, v1,v2,v3,v4]^T
        """
        B = np.zeros((3, 8))
        mat = self._dSF_dx_T(r, s)
        B[0, 0:4] = mat[0]       # du/dx
        B[1, 4:8] = mat[1]       # dv/dy
        B[2, 0:4] = mat[1]       # du/dy
        B[2, 4:8] = mat[0]       # dv/dx
        return B

    # ------------------------------------------------ membrane stiffness
    def local_stiffness(self) -> np.ndarray:
        """8×8 lokal membrane rijitliği. Gauss 2×2 integrasyon."""
        C = self.constitutive_matrix()
        h = self.thickness
        K = np.zeros((8, 8))
        for i in range(2):
            for j in range(2):
                r, s = _GAUSS[i], _GAUSS[j]
                w = _WEIGHTS[i] * _WEIGHTS[j]
                B = self.strain_displacement(r, s)
                detJ = DET(self._jacobian(r, s))
                K = K + w * h * B.T @ C @ B * detJ
        return K


# --------------------------------------------------------------- helpers
def build_from_dto(
    nodes_3d: list[NodeDTO],
    section: ShellSectionDTO,
    material: MaterialDTO,
    local_frame: np.ndarray,
    origin: np.ndarray,
) -> PlaneStressQ4:
    """3D düğüm koordinatlarını lokal düzleme iz düşür, PlaneStressQ4 kur.

    ``local_frame``: 3×3 matris — satırları [x', y', z'] lokal eksenler.
    ``origin``: lokal koordinat sisteminin orijini (genelde n1).
    Sıra: cyclic SAP sırası.
    """
    local_xy = []
    for n in nodes_3d:
        p = np.asarray([n.x - origin[0], n.y - origin[1], n.z - origin[2]])
        # x' ve y' eksenlerine iz düşür
        local_xy.append([local_frame[0] @ p, local_frame[1] @ p])
    cyclic_xy = np.asarray(local_xy)
    return PlaneStressQ4.from_cyclic(
        cyclic_xy, E=material.E, nu=material.nu, thickness=section.thickness,
    )
