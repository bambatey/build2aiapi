"""Modal analiz — ``K·φ = ω² M·φ`` genelleştirilmiş öz-değer problemi.

Yalnızca serbest DOF'lar üzerinde çözülür. ``scipy.sparse.linalg.eigsh``
ile en küçük ``n_modes`` öz-değer bulunur.

Limitler (MVP):
- Diagonal (lumped) mass → rotational ataletler yok
- Kütle katılım oranları hesaplanmıyor (sonraki iterasyon)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if TYPE_CHECKING:
    from ..assembly.dof_numbering import DofMap
    from ..pipeline import ModeResult

logger = logging.getLogger(__name__)


def solve_modal(
    K: sp.csc_matrix,
    M: sp.csc_matrix,
    dof_map: "DofMap",
    n_modes: int,
) -> "list[ModeResult]":
    from ..pipeline import ModeResult  # döngüsel import engeli

    N = dof_map.n_free
    if N == 0:
        logger.warning("Modal: hiç serbest DOF yok, atlanıyor.")
        return []

    K11 = K[0:N, 0:N]
    M11 = M[0:N, 0:N]

    if np.all(M11.diagonal() == 0):
        logger.error("Modal: M11 tümüyle sıfır — kütle ataması yok.")
        return []

    k = min(n_modes, N - 2)
    if k < 1:
        logger.warning("Modal: N (%d) çok küçük, atlanıyor.", N)
        return []

    try:
        eigvals, eigvecs = spla.eigsh(K11, k=k, M=M11, sigma=0, which="LM")
    except Exception as exc:
        logger.warning("eigsh sigma=0 başarısız (%s), SM ile denenecek.", exc)
        try:
            eigvals, eigvecs = spla.eigsh(K11, k=k, M=M11, which="SM")
        except Exception as exc2:
            logger.error("Modal eigsh başarısız: %s", exc2)
            return []

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    modes: list[ModeResult] = []
    M_full = dof_map.n_total

    # Kütle katılım oranlarını hesapla: r vektörleri + toplam kütleler
    r_x = np.zeros(M_full)
    r_y = np.zeros(M_full)
    r_z = np.zeros(M_full)
    for code in dof_map.codes.values():
        r_x[code[0]] = 1.0
        r_y[code[1]] = 1.0
        r_z[code[2]] = 1.0
    # Reduced space'e indir (free DOF kısmı)
    r_x_f = r_x[0:N]
    r_y_f = r_y[0:N]
    r_z_f = r_z[0:N]
    M_total_x = float(r_x_f @ M11 @ r_x_f)
    M_total_y = float(r_y_f @ M11 @ r_y_f)
    M_total_z = float(r_z_f @ M11 @ r_z_f)

    for idx, lam in enumerate(eigvals):
        if lam <= 0 or not np.isfinite(lam):
            continue
        omega = math.sqrt(float(lam))
        freq = omega / (2 * math.pi)
        period = 1.0 / freq if freq > 0 else float("inf")

        # Ham mod şekli (ölçeklenmemiş) — katılım için gereklidir
        phi_raw = np.zeros(M_full)
        phi_raw[0:N] = eigvecs[:, idx]
        phi_f = eigvecs[:, idx]

        # Modal genel kütle m_n* = φ^T M φ (eigsh mass-normalized verir ama
        # kalıcı olmak için yeniden hesapla)
        m_gen = float(phi_f @ M11 @ phi_f)
        participation: dict[str, float] = {}
        if m_gen > 0:
            for label, r_f, M_tot in [
                ("ux", r_x_f, M_total_x),
                ("uy", r_y_f, M_total_y),
                ("uz", r_z_f, M_total_z),
            ]:
                if M_tot <= 0:
                    participation[label] = 0.0
                    continue
                gamma = float(phi_f @ M11 @ r_f) / m_gen
                m_eff = gamma * gamma * m_gen
                participation[label] = m_eff / M_tot   # oran (0..1)

        # UI için normalize edilmiş şekil (max|φ| = 1)
        phi_disp = phi_raw.copy()
        peak = np.max(np.abs(phi_disp))
        if peak > 0:
            phi_disp = phi_disp / peak

        modes.append(
            ModeResult(
                mode_no=idx + 1,
                period=period,
                frequency=freq,
                angular_frequency=omega,
                shape=_shape_to_dict(phi_disp, dof_map),
                mass_participation=participation,
            )
        )
    return modes


def _shape_to_dict(phi: np.ndarray, dof_map: "DofMap") -> dict[int, dict[str, float]]:
    keys = ("ux", "uy", "uz", "rx", "ry", "rz")
    return {
        nid: {k: float(phi[code[i]]) for i, k in enumerate(keys)}
        for nid, code in dof_map.codes.items()
    }
