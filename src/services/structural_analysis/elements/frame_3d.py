"""3D Frame (kiriş-kolon) elemanı.

Port kaynağı: ``sonlu-elemanlar-analizi/SEA_Book/sec5_frame_3d.py``.

Bu sınıfın matematiği orijinal ``ElementFrame3D`` ile **bire bir** aynıdır.
Characterization testi (``tests/test_frame_3d.py``) altın çıktı
(``tests/benchmarks/frame3d_golden.json``) ile her rilease öncesi doğrular.

API farkı:
    - Orijinal kod modül düzeyi global ``nodes/materials/sections`` sözlüklerine
      bağımlıdır. Burada sınıf stateless; DTO nesneleri parametre olarak verilir.
    - Yayılı yük vektörü ``q`` artık yapıcıya değil ``local_load_vector(q)`` /
      ``global_load_vector(q)`` metotlarına geçer. Böylece aynı eleman farklı
      yük durumları için yeniden kullanılabilir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..model.dto import FrameElementDTO, FrameSectionDTO, MaterialDTO, NodeDTO


@dataclass(frozen=True)
class FrameElement3D:
    """Tek bir 3B çerçeve elemanının lokal/global rijitlik matematiği.

    ``omega_deg`` verilmezse ``element.local_axis_angle`` kullanılır.
    """

    element: FrameElementDTO
    node_i: NodeDTO
    node_j: NodeDTO
    section: FrameSectionDTO
    material: MaterialDTO
    omega_deg: float | None = None

    # --------------------------------------------------------- temel özellikler
    @property
    def omega(self) -> float:
        """Kesit duruş açısı (radyan). Orijinaldeki ``elm.omega`` ile özdeş."""
        deg = self.omega_deg if self.omega_deg is not None else self.element.local_axis_angle
        return math.radians(deg)

    @property
    def shear_modulus(self) -> float:
        """G = E / (2(1+ν))."""
        return 0.5 * self.material.E / (1.0 + self.material.nu)

    def direction_cosines_and_length(self) -> tuple[float, float, float, float]:
        """Orijinal ``nx_ny_nz_L`` ile özdeş."""
        n1, n2 = self.node_i, self.node_j
        Lx, Ly, Lz = n2.x - n1.x, n2.y - n1.y, n2.z - n1.z
        L = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
        return Lx / L, Ly / L, Lz / L, L

    @property
    def length(self) -> float:
        return self.direction_cosines_and_length()[3]

    def element_axes_transform(self) -> np.ndarray:
        """Elemanın 3×3 global→lokal dönüşüm matrisi (TOMG @ TALFA).

        Düğüm yerel eksenlerini (TBETA) HARİÇ tutar — yalnızca elemanın
        kendi eksen takımı. Element üzerindeki yayılı yüklerin global→lokal
        dönüşümünde kullanılır.
        """
        omg = self.omega
        TOMG = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(omg), np.sin(omg)],
                [0.0, -np.sin(omg), np.cos(omg)],
            ]
        )
        nx, ny, nz, _ = self.direction_cosines_and_length()
        if np.abs(1 - nz * nz) < 0.001:
            TALFA = np.asarray([[nx, ny, nz], [1.0, 0.0, 0.0], [0.0, nz, -ny]])
        else:
            a = 1.0 / (1.0 - nz * nz)
            TALFA = np.asarray(
                [
                    [nx, ny, nz],
                    [-a * nx * nz, -a * ny * nz, 1.0],
                    [a * ny, -a * nx, 0.0],
                ]
            )
        return TOMG @ TALFA

    # -------------------------------------------------------- dönüşüm matrisi
    def local_to_global_transform(self) -> np.ndarray:
        """12×12 dönüşüm matrisi T. Orijinal ``TLG`` ile özdeş."""
        TE = self.element_axes_transform()
        TBETA1 = _node_euler_transform(self.node_i)
        TBETA2 = _node_euler_transform(self.node_j)
        T = np.identity(12)
        T[0:3, 0:3] = TE @ TBETA1
        T[3:6, 3:6] = TE @ TBETA1
        T[6:9, 6:9] = TE @ TBETA2
        T[9:12, 9:12] = TE @ TBETA2
        return T

    # ------------------------------------------------------ rijitlik matrisi
    def local_stiffness(self) -> np.ndarray:
        """12×12 lokal rijitlik matrisi. Orijinal ``K_Local`` ile özdeş."""
        E, G = self.material.E, self.shear_modulus
        A = self.section.A
        # SEA_Book notasyonu: Ix = burulma (J), Iy = 2-2, Iz = 3-3, Iyz = çarpım
        Ix, Iy, Iz, Iyz = self.section.J, self.section.Iy, self.section.Iz, self.section.Iyz
        _, _, _, L = self.direction_cosines_and_length()

        EA, GIx, EIy, EIz, EIyz = E * A, G * Ix, E * Iy, E * Iz, E * Iyz
        L2, L3 = L * L, L * L * L
        ku = EA / L
        ktx = GIx / L
        k1z, k1y, k1yz = 2 * EIz / L, 2 * EIy / L, 2 * EIyz / L
        k2z, k2y, k2yz = 6 * EIz / L2, 6 * EIy / L2, 6 * EIyz / L2
        k3z, k3y, k3yz = 12 * EIz / L3, 12 * EIy / L3, 12 * EIyz / L3
        return np.asarray(
            [
                [ku, 0, 0, 0, 0, 0, -ku, 0, 0, 0, 0, 0],
                [0, k3z, k3yz, 0, -k2yz, k2z, 0, -k3z, -k3yz, 0, -k2yz, k2z],
                [0, k3yz, k3y, 0, -k2y, k2yz, 0, -k3yz, -k3y, 0, -k2y, k2yz],
                [0, 0, 0, ktx, 0, 0, 0, 0, 0, -ktx, 0, 0],
                [0, -k2yz, -k2y, 0, 2 * k1y, -2 * k1yz, 0, k2yz, k2y, 0, k1y, -k1yz],
                [0, k2z, k2yz, 0, -2 * k1yz, 2 * k1z, 0, -k2z, -k2yz, 0, -k1yz, k1z],
                [-ku, 0, 0, 0, 0, 0, ku, 0, 0, 0, 0, 0],
                [0, -k3z, -k3yz, 0, k2yz, -k2z, 0, k3z, k3yz, 0, k2yz, -k2z],
                [0, -k3yz, -k3y, 0, k2y, -k2yz, 0, k3yz, k3y, 0, k2y, -k2yz],
                [0, 0, 0, -ktx, 0, 0, 0, 0, 0, ktx, 0, 0],
                [0, -k2yz, -k2y, 0, k1y, -k1yz, 0, k2yz, k2y, 0, 2 * k1y, -2 * k1yz],
                [0, k2z, k2yz, 0, -k1yz, k1z, 0, -k2z, -k2yz, 0, -2 * k1yz, 2 * k1z],
            ]
        )

    def global_stiffness(self) -> np.ndarray:
        """Global rijitlik: ``inv(T) @ K_local @ T``. Orijinal ``K`` ile özdeş."""
        T = self.local_to_global_transform()
        return np.linalg.inv(T) @ self.local_stiffness() @ T

    # ---------------------------------------------------- yayılı yük vektörü
    def local_load_vector(self, q: list[float] | np.ndarray) -> np.ndarray:
        """Yayılı yükün eleman uçlarındaki lokal eş-nodal vektörü.

        ``q = [qx, qy, qz, mx, my, mz]`` — orijinal ``q_Local`` ile özdeş.
        """
        qx, qy, qz, mx, my, mz = q
        _, _, _, L = self.direction_cosines_and_length()
        L2 = L * L
        return np.asarray(
            [
                0.5 * qx * L,
                0.5 * qy * L - mz,
                0.5 * qz * L + my,
                0.5 * mx * L,
                -qz * L2 / 12,
                qy * L2 / 12,
                0.5 * qx * L,
                0.5 * qy * L + mz,
                0.5 * qz * L - my,
                0.5 * mx * L,
                qz * L2 / 12,
                -qy * L2 / 12,
            ]
        )

    def global_load_vector(self, q: list[float] | np.ndarray) -> np.ndarray:
        """Global eş-nodal yük vektörü: ``inv(T) @ q_local``. Orijinal ``B``."""
        T = self.local_to_global_transform()
        return np.linalg.inv(T) @ self.local_load_vector(q)


# --------------------------------------------------------------------- helpers
def _node_euler_transform(node: NodeDTO) -> np.ndarray:
    """Düğüm yerel eksen ZYX Euler dönüşümü. Orijinal
    ``getNodeLokalAxesTransformation`` ile özdeş.
    """
    ezd, eyd, exd = node.euler_zyx
    rx, ry, rz = math.radians(exd), math.radians(eyd), math.radians(ezd)
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)
    return np.asarray(
        [
            [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy],
        ]
    )
