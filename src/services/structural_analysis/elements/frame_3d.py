"""3D Frame (kiriş-kolon) elemanı.

Port kaynağı: ``sonlu-elemanlar-analizi/SEA_Book/sec5_frame_3d.py``.

Bu modülün matematiği orijinal ``ElementFrame3D`` ile **bire bir** aynıdır.
Characterization testi (``tests/test_frame_3d.py``) altın çıktı
(``tests/benchmarks/frame3d_golden.json``) ile her rilease öncesi doğrular.

Kod yapısı (docs/architecture/01-performance.md §1.5 Rust-ready refactor):
    - Tüm hot kernel'ler module düzeyinde **pure fonksiyon** olarak yazılır
      ve ``# RUST_KERNEL_CANDIDATE`` etiketiyle işaretlenir. Yalnızca
      ``float`` / ``np.ndarray`` argümanları alırlar, side-effect yok.
    - ``FrameElement3D`` dataclass'ı bu pure fonksiyonları DTO'lardan
      parametre çıkaran bir sarmalayıcıdır — geriye uyumluluk için korunur
      ama performance path'i artık ``FrameKernel`` cache + pure fonksiyonlar
      üzerinden işler.
    - Tagli fonksiyonlar Faz 2'de ``build2ai_kernels`` Rust crate'ine
      ``frame_local_stiffness``, ``element_axes_transform`` vb. isimlerle
      port edilir (docs/architecture/03-rust-wasm-hybrid.md Tier 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..model.dto import FrameElementDTO, FrameSectionDTO, MaterialDTO, NodeDTO


# ============================================================
# PURE HOT KERNEL'LER (Rust port adayları)
# ============================================================
# RUST_KERNEL_CANDIDATE — Tier 1
def frame_local_stiffness(
    E: float, G: float, A: float,
    Iy: float, Iz: float, J: float, Iyz: float,
    L: float,
) -> np.ndarray:
    """12×12 lokal rijitlik matrisi. SEA_Book ``K_Local`` ile özdeş.

    Argümanlar skaler float; dönüş (12, 12) float64. Pure.
    """
    EA, GIx, EIy, EIz, EIyz = E * A, G * J, E * Iy, E * Iz, E * Iyz
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


# RUST_KERNEL_CANDIDATE — Tier 1
def frame_element_axes_transform(
    nx: float, ny: float, nz: float, omega_rad: float,
) -> np.ndarray:
    """3×3 global→lokal element ekseni dönüşümü (TOMG @ TALFA). Pure."""
    co = math.cos(omega_rad)
    so = math.sin(omega_rad)
    TOMG = np.asarray([[1.0, 0.0, 0.0], [0.0, co, so], [0.0, -so, co]])
    if abs(1 - nz * nz) < 0.001:
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


# RUST_KERNEL_CANDIDATE — Tier 1
def node_euler_transform(
    ez_deg: float, ey_deg: float, ex_deg: float,
) -> np.ndarray:
    """Düğüm yerel eksen ZYX Euler 3×3 dönüşümü. Pure."""
    rx, ry, rz = math.radians(ex_deg), math.radians(ey_deg), math.radians(ez_deg)
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)
    return np.asarray(
        [
            [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy],
        ]
    )


# RUST_KERNEL_CANDIDATE — Tier 1
def frame_local_to_global_transform(
    TE: np.ndarray, TBETA_i: np.ndarray, TBETA_j: np.ndarray,
) -> np.ndarray:
    """12×12 global→lokal dönüşüm. TE (3×3) elemana, TBETA (3×3) düğümlere ait.

    Orijinal ``TLG = diag([TE@Ti, TE@Ti, TE@Tj, TE@Tj])``.
    """
    T = np.identity(12)
    TEi = TE @ TBETA_i
    TEj = TE @ TBETA_j
    T[0:3, 0:3] = TEi
    T[3:6, 3:6] = TEi
    T[6:9, 6:9] = TEj
    T[9:12, 9:12] = TEj
    return T


# RUST_KERNEL_CANDIDATE — Tier 1 helper
def direction_cosines(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
) -> tuple[float, float, float, float]:
    """(nx, ny, nz, L) — iki nokta arası birim vektör + boy. Pure."""
    Lx, Ly, Lz = x2 - x1, y2 - y1, z2 - z1
    L = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    return Lx / L, Ly / L, Lz / L, L


# RUST_KERNEL_CANDIDATE — Tier 1 helper
def condense_released_dofs(K: np.ndarray, released: list[int]) -> np.ndarray:
    """Static condensation: released DOF'ları elimine et, K'yı 12×12 tut.

        K_r = K_rr - K_rc @ inv(K_cc) @ K_cr    (retained subset)
        Sonra released pozisyonlara 0 doldur.

    Pure fonksiyon — çağıranın sorumluluk alanı yan etki.
    """
    released = sorted(set(released))
    if not released:
        return K.copy()
    all_idx = list(range(K.shape[0]))
    retained = [i for i in all_idx if i not in released]

    K_rr = K[np.ix_(retained, retained)]
    K_rc = K[np.ix_(retained, released)]
    K_cr = K[np.ix_(released, retained)]
    K_cc = K[np.ix_(released, released)]

    try:
        reduced = K_rr - K_rc @ np.linalg.solve(K_cc, K_cr)
    except np.linalg.LinAlgError:
        reduced = K_rr - K_rc @ np.linalg.pinv(K_cc) @ K_cr

    K_new = np.zeros_like(K)
    for i, ri in enumerate(retained):
        for j, rj in enumerate(retained):
            K_new[ri, rj] = reduced[i, j]
    return K_new


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
        """Orijinal ``nx_ny_nz_L`` ile özdeş — pure ``direction_cosines``'e delege."""
        n1, n2 = self.node_i, self.node_j
        return direction_cosines(n1.x, n1.y, n1.z, n2.x, n2.y, n2.z)

    @property
    def length(self) -> float:
        return self.direction_cosines_and_length()[3]

    def element_axes_transform(self) -> np.ndarray:
        """3×3 global→lokal (sadece element ekseni)."""
        nx, ny, nz, _ = self.direction_cosines_and_length()
        return frame_element_axes_transform(nx, ny, nz, self.omega)

    # -------------------------------------------------------- dönüşüm matrisi
    def local_to_global_transform(self) -> np.ndarray:
        """12×12 dönüşüm matrisi T. Orijinal ``TLG`` ile özdeş."""
        TE = self.element_axes_transform()
        TBETA1 = node_euler_transform(*self.node_i.euler_zyx)
        TBETA2 = node_euler_transform(*self.node_j.euler_zyx)
        return frame_local_to_global_transform(TE, TBETA1, TBETA2)

    # ------------------------------------------------------ rijitlik matrisi
    def local_stiffness(self) -> np.ndarray:
        """12×12 lokal rijitlik matrisi. Orijinal ``K_Local`` ile özdeş."""
        # SEA_Book notasyonu: Ix = burulma (J), Iy = 2-2, Iz = 3-3, Iyz = çarpım
        _, _, _, L = self.direction_cosines_and_length()
        return frame_local_stiffness(
            E=self.material.E, G=self.shear_modulus, A=self.section.A,
            Iy=self.section.Iy, Iz=self.section.Iz,
            J=self.section.J, Iyz=self.section.Iyz,
            L=L,
        )

    def local_stiffness_with_releases(self) -> np.ndarray:
        """Mafsallar (hinges) varsa static condensation uygulanmış lokal K.

        Released DOF'lar kondanse edilir; geri kalan retained DOF'ların
        rijitliği korunur, released satır/kolonlar sıfırlanır. Böylece
        global birleştirmede o DOF'a eleman katkısı olmaz.
        """
        K = self.local_stiffness()
        if not self.element.hinges:
            return K
        released = _release_indices(self.element.hinges)
        if not released:
            return K
        return condense_released_dofs(K, released)

    def global_stiffness(self) -> np.ndarray:
        """Global rijitlik — releases varsa kondanse edilmiş halden dönüşür."""
        T = self.local_to_global_transform()
        return np.linalg.inv(T) @ self.local_stiffness_with_releases() @ T

    # ---------------------------------------------------- yayılı yük vektörü
    def local_load_vector(self, q: list[float] | np.ndarray) -> np.ndarray:
        """Yayılı yükün eleman uçlarındaki lokal eş-nodal vektörü.

        ``q = [qx, qy, qz, mx, my, mz]`` — orijinal ``q_Local`` ile özdeş.
        Pure helper ``_local_load_vector`` (``assembly.load_assembler``)
        ile bire bir aynı matematik; bu method sarmalayıcıdır.
        """
        # NOT: Geriye uyumlu shim — yeni kod load_assembler._local_load_vector
        # kullanmalı. (circular import olmasın diye lokal)
        from ..assembly.load_assembler import _local_load_vector
        _, _, _, L = self.direction_cosines_and_length()
        q_arr = np.asarray(q, dtype=float)
        return _local_load_vector(q_arr, L)

    def global_load_vector(self, q: list[float] | np.ndarray) -> np.ndarray:
        """Global eş-nodal yük vektörü: ``inv(T) @ q_local``. Orijinal ``B``."""
        T = self.local_to_global_transform()
        return np.linalg.inv(T) @ self.local_load_vector(q)


# --------------------------------------------------------------------- helpers
# SAP release tag → 12-DOF lokal K indeksi
# Lokal K sırası: [u, v, w, θx, θy, θz] at I (0-5), sonra J (6-11)
# SAP konvansiyonu: P=axial (u), V2=shear in 2-dir (v), V3=shear in 3-dir (w),
#                   T=torsion (θx), M2=moment about 2 (θy), M3=moment about 3 (θz)
_RELEASE_DOF_AT_I = {"p": 0, "v2": 1, "v3": 2, "t": 3, "m2": 4, "m3": 5}
_RELEASE_DOF_AT_J = {k: v + 6 for k, v in _RELEASE_DOF_AT_I.items()}


def _release_indices(hinges: dict[str, list[str]]) -> list[int]:
    out: list[int] = []
    for tag in hinges.get("start", []):
        idx = _RELEASE_DOF_AT_I.get(tag.lower())
        if idx is not None:
            out.append(idx)
    for tag in hinges.get("end", []):
        idx = _RELEASE_DOF_AT_J.get(tag.lower())
        if idx is not None:
            out.append(idx)
    return out
