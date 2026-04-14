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
    for idx, lam in enumerate(eigvals):
        if lam <= 0 or not np.isfinite(lam):
            continue
        omega = math.sqrt(float(lam))
        freq = omega / (2 * math.pi)
        period = 1.0 / freq if freq > 0 else float("inf")

        phi_full = np.zeros(M_full)
        phi_full[0:N] = eigvecs[:, idx]
        peak = np.max(np.abs(phi_full))
        if peak > 0:
            phi_full = phi_full / peak

        modes.append(
            ModeResult(
                mode_no=idx + 1,
                period=period,
                frequency=freq,
                angular_frequency=omega,
                shape=_shape_to_dict(phi_full, dof_map),
            )
        )
    return modes


def _shape_to_dict(phi: np.ndarray, dof_map: "DofMap") -> dict[int, dict[str, float]]:
    keys = ("ux", "uy", "uz", "rx", "ry", "rz")
    return {
        nid: {k: float(phi[code[i]]) for i, k in enumerate(keys)}
        for nid, code in dof_map.codes.items()
    }
