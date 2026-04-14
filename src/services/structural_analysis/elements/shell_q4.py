"""4-düğümlü tam shell elemanı — membran + plate bending + drilling.

Her düğümde 6 DOF: (ux, uy, uz, rx, ry, rz) global frame'de. Eleman
lokal frame'i düğüm konumlarından türetilir:

    x̂ = (n1 → n2 yönü normalize)
    n_hat = ((n1→n2) × (n1→n4)) normalize   → lokal z'
    ŷ = n_hat × x̂                            → lokal y'
    T_e = [x̂, ŷ, n_hat]  (3×3 row matrix → satırlar lokal eksenler)

Lokal DOF'lar (5 per node: u', v' membran + w' plate + θ'x, θ'y plate
rotasyonu). 6. DOF rz' drilling — membran/plate hiçbiri bunu kullanmaz,
stabilizasyon için küçük rijitlik verilir.

24×24 global K = T^T × K_local × T

Lokal K_local (24×24):
    - Membran (in-plane): DOF'lar u', v' her 4 düğümde → 8 DOF
    - Plate bending: DOF'lar w', θ'x, θ'y → 12 DOF
    - Drilling: DOF θ'z → 4 DOF (küçük rijitlik)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.dto import MaterialDTO, NodeDTO, ShellSectionDTO
from .plane_stress_q4 import PlaneStressQ4
from .plate_bending_q4 import PlateBendingQ4


@dataclass(frozen=True)
class ShellQ4:
    """Tam 4-düğümlü shell — membran + plate bending.

    Düğüm sırası: SAP cyclic (n1 → n2 → n3 → n4 köşe köşe).
    """

    nodes: list[NodeDTO]         # 4 düğüm, cyclic sırada
    E: float
    nu: float
    thickness: float

    # ------------------------------------------------ eleman lokal frame
    def local_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """3×3 dönüşüm matrisi ve orijin (n1) döner.

        Satırlar [x̂, ŷ, n̂] lokal eksenler (global koordinatlarda).
        """
        p = [np.asarray([n.x, n.y, n.z]) for n in self.nodes]
        x_vec = p[1] - p[0]
        x_hat = x_vec / np.linalg.norm(x_vec)
        diag = p[3] - p[0]
        n_vec = np.cross(x_vec, diag)
        n_hat = n_vec / np.linalg.norm(n_vec)
        y_hat = np.cross(n_hat, x_hat)
        T = np.vstack([x_hat, y_hat, n_hat])
        return T, p[0]

    def local_xy(self) -> np.ndarray:
        """Her 4 düğümün lokal düzlemdeki 2D koordinatları (cyclic sırada)."""
        T, origin = self.local_frame()
        out = []
        for n in self.nodes:
            p = np.asarray([n.x - origin[0], n.y - origin[1], n.z - origin[2]])
            out.append([T[0] @ p, T[1] @ p])
        return np.asarray(out)

    # --------------------------------------------- lokal 24×24 rijitlik
    def local_stiffness(self) -> np.ndarray:
        cyclic_xy = self.local_xy()
        membrane = PlaneStressQ4.from_cyclic(
            cyclic_xy, E=self.E, nu=self.nu, thickness=self.thickness,
        )
        bending = PlateBendingQ4.from_cyclic(
            cyclic_xy, E=self.E, nu=self.nu, thickness=self.thickness,
        )

        K_m = membrane.local_stiffness()      # 8×8: [u1..u4, v1..v4] (tensor)
        K_b = bending.local_stiffness()       # 12×12: [w,θx,θy × 4] (tensor)

        # Tensor → cyclic düğüm sırasına çevir (u1,v1,u2,v2,... vs. taşıma)
        # Aslında K_m ve K_b zaten tensor sırasında. Node-cyclic → tensor
        # haritası aynı: from_cyclic içeride reorder yapmıştı.
        # Tensor-to-cyclic node index: tensor i=0→cyclic 0, 1→1, 2→3, 3→2
        tensor_to_cyclic = [0, 1, 3, 2]

        K_local = np.zeros((24, 24))

        # Membran yerleşimi: tensor i-node'un u DOF'u → lokal 24'te
        # cyclic_i=tensor_to_cyclic[i], DOF offset 0 (u')
        # Membran K_m indekslemesi: u1..u4 (0..3), v1..v4 (4..7) tensor
        for ti in range(4):
            ci = tensor_to_cyclic[ti]
            for tj in range(4):
                cj = tensor_to_cyclic[tj]
                # u-u: K_m[ti, tj] → 24'te (6*ci + 0, 6*cj + 0)
                K_local[6 * ci + 0, 6 * cj + 0] += K_m[ti, tj]
                # u-v
                K_local[6 * ci + 0, 6 * cj + 1] += K_m[ti, 4 + tj]
                # v-u
                K_local[6 * ci + 1, 6 * cj + 0] += K_m[4 + ti, tj]
                # v-v
                K_local[6 * ci + 1, 6 * cj + 1] += K_m[4 + ti, 4 + tj]

        # Plate bending yerleşimi: tensor i-node'un w, θx, θy (3 DOF)
        # K_b indekslemesi: tensor sıra, node başına 3 DOF (w@3i, θx@3i+1, θy@3i+2)
        # Shell lokal: w' @ 6*ci+2, θx' @ 6*ci+3, θy' @ 6*ci+4
        bend_offset = [2, 3, 4]
        for ti in range(4):
            ci = tensor_to_cyclic[ti]
            for tj in range(4):
                cj = tensor_to_cyclic[tj]
                for a in range(3):
                    for b in range(3):
                        K_local[6 * ci + bend_offset[a],
                                6 * cj + bend_offset[b]] += \
                            K_b[3 * ti + a, 3 * tj + b]

        # Drilling stabilizasyon: θ'z (DOF 5) için küçük rijitlik.
        # Değer: membran'ın max diyagonal × 1e-4 — pratik bir konvansiyon.
        max_m_diag = float(np.max(np.abs(K_m.diagonal())))
        drill = max_m_diag * 1e-4 if max_m_diag > 0 else 1.0
        for ci in range(4):
            K_local[6 * ci + 5, 6 * ci + 5] += drill

        return K_local

    # ---------------------------------------------- global 24×24 rijitlik
    def global_stiffness(self) -> np.ndarray:
        """Global K — her düğümün 3 translasyon + 3 rotasyon DOF'unu
        lokal eksenden globale dönüştürür."""
        T_e, _ = self.local_frame()
        # Her düğüm için 6×6 blok dönüşümü: translasyon + rotasyon
        block = np.zeros((6, 6))
        block[0:3, 0:3] = T_e
        block[3:6, 3:6] = T_e
        # 24×24 büyük transform: 4 kez diyagonalde block
        big = np.zeros((24, 24))
        for i in range(4):
            big[6 * i:6 * i + 6, 6 * i:6 * i + 6] = block
        K_local = self.local_stiffness()
        # K_global = T^T × K_local × T (big orthogonal → T^-1 = T^T)
        return big.T @ K_local @ big


# ------------------------------------------------------------- helper
def build_shell(
    nodes: list[NodeDTO],
    section: ShellSectionDTO,
    material: MaterialDTO,
) -> ShellQ4:
    return ShellQ4(
        nodes=nodes,
        E=material.E, nu=material.nu, thickness=section.thickness,
    )
