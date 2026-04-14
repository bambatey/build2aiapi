"""Response Spectrum Analysis (RSA) — modal kombinasyon.

Her mod için:

    Modal yer değiştirme:  u_n = Γ_n × Sa(T_n) / ω_n² × φ_n

burada:
    Γ_n = φ_n^T M r / (φ_n^T M φ_n)   — modal katılım çarpanı
    r   = yön vektörü (X için UX DOF'larında 1, diğerleri 0)
    Sa  = tasarım ivme spektrumu (m/s²)

Modal kombinasyon: SRSS (Square Root of Sum of Squares)
    U_rs = sqrt(Σ u_n²)

Not: CQC (Complete Quadratic Combination) sonraki iterasyon; SRSS yakın
periyotlu modlar olmadığında yeterlidir.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse as sp

from ..exceptions import SolverError

if TYPE_CHECKING:
    from ..assembly.dof_numbering import DofMap
    from ..pipeline import ModeResult
    from ..spectra.tbdy_2018 import TBDY2018Spectrum

logger = logging.getLogger(__name__)

Direction = Literal["x", "y", "z"]

_DIR_TO_DOF_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class ResponseSpectrumResult:
    """Tek bir yön için RS sonucu."""

    direction: Direction
    U: np.ndarray          # SRSS birleştirilmiş sistem yer değiştirme vektörü
    per_mode_max: list[float]       # her modun peak |disp| katkısı (diagnostics)
    participation_factors: list[float]
    spectral_accels: list[float]    # her moda karşılık gelen Sa (m/s²)


def solve_response_spectrum(
    K: sp.csc_matrix,
    M: sp.csc_matrix,
    modes: "list[ModeResult]",
    dof_map: "DofMap",
    spectrum: "TBDY2018Spectrum",
    direction: Direction,
) -> ResponseSpectrumResult:
    """Response spectrum — tek yönde modal katılım + SRSS."""
    if not modes:
        raise SolverError("RS için modal sonuç yok.")

    N = dof_map.n_free
    M11 = M[0:N, 0:N]

    # Yön vektörü r: her düğümün d-DOF'una 1 yerleştir, gerisi 0
    dof_idx = _DIR_TO_DOF_INDEX[direction]
    r = np.zeros(dof_map.n_total)
    for code in dof_map.codes.values():
        r[code[dof_idx]] = 1.0
    r_free = r[0:N]

    # Her modun φ_n'ini free-DOF vektörüne çevir (pipeline'da shape düğüm sözlüğü tutuyor;
    # re-inşa için dof_map kullan)
    U_sq = np.zeros(dof_map.n_total)     # SRSS birikim (DOF başına)
    per_mode_max: list[float] = []
    participations: list[float] = []
    spectral_accels: list[float] = []

    for mode in modes:
        phi = _shape_to_vector(mode.shape, dof_map)
        phi_free = phi[0:N]

        # ω² = Mφ = Kφ ⇒ ω_n already known from modal
        omega2 = mode.angular_frequency ** 2
        if omega2 <= 0:
            continue

        # Modal katılım: Γ = φ^T M r / (φ^T M φ)
        M_phi = M11 @ phi_free
        gen_mass = float(phi_free @ M_phi)     # modal genel kütle m_n*
        if gen_mass <= 0:
            continue
        gamma = float(phi_free @ (M11 @ r_free)) / gen_mass

        # Spektral ivme (m/s²)
        Sa = spectrum.Sa_design_ms2(mode.period)

        # Modal yer değiştirme katkısı
        # u_n = Γ × Sa / ω² × φ
        factor = gamma * Sa / omega2
        u_n = factor * phi

        U_sq += u_n ** 2

        per_mode_max.append(float(np.max(np.abs(u_n))))
        participations.append(gamma)
        spectral_accels.append(Sa)

    U_rs = np.sqrt(U_sq)
    # SRSS her zaman pozitif döner — işaret kaybı normal, kullanıcı yönü seçerken bilir
    return ResponseSpectrumResult(
        direction=direction,
        U=U_rs,
        per_mode_max=per_mode_max,
        participation_factors=participations,
        spectral_accels=spectral_accels,
    )


def _shape_to_vector(
    shape: dict[int, dict[str, float]], dof_map: "DofMap"
) -> np.ndarray:
    """Mod şekli dict'ini full DOF vektörüne çevir."""
    keys = ("ux", "uy", "uz", "rx", "ry", "rz")
    v = np.zeros(dof_map.n_total)
    for nid, disp in shape.items():
        code = dof_map.codes.get(nid)
        if code is None:
            continue
        for i, k in enumerate(keys):
            v[code[i]] = disp.get(k, 0.0)
    return v
