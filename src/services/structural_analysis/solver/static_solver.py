"""Statik lineer çözüm.

SEA_Book referansı ile aynı partitioned çözüm:

    K11 U1 = P1 + RHS1 - K12 U2
    P2 = K21 U1 + K22 U2 - RHS2

Orijinal ``bicg`` yerine ``scipy.sparse.linalg.spsolve`` (direct) kullanılır
— küçük/orta modeller için hem daha kararlı hem daha hızlı. Tol ile hassasiyet
kaybı yok.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..assembly.dof_numbering import DofMap
from ..exceptions import SolverError


@dataclass
class StaticSolution:
    """Tek bir yük durumunun çözüm sonucu."""

    case_id: str
    U: np.ndarray          # M uzunluklu (serbestler + tutulular, partitioned)
    P: np.ndarray          # M uzunluklu (dış kuvvetler + mesnet reaksiyonları)


def solve_static(
    case_id: str,
    K: sp.csc_matrix,
    PS: np.ndarray,
    RHS: np.ndarray,
    US: np.ndarray,
    dof_map: DofMap,
) -> StaticSolution:
    """Tek yük durumu çözümü."""
    N, M = dof_map.n_free, dof_map.n_total
    if N == 0:
        # Tüm DOF'lar tutulu → U = US, P = P2 reaksiyonları
        P2 = K @ US - RHS
        return StaticSolution(case_id=case_id, U=US.copy(), P=P2)

    K11 = K[0:N, 0:N]
    K12 = K[0:N, N:M]
    K21 = K[N:M, 0:N]
    K22 = K[N:M, N:M]

    U2 = US[N:M]
    P1 = PS[0:N]
    RHS1 = RHS[0:N]
    RHS2 = RHS[N:M]

    rhs_eff = P1 + RHS1 - (K12 @ U2 if U2.any() else 0.0)
    try:
        U1 = spla.spsolve(K11, rhs_eff)
    except Exception as exc:  # pragma: no cover - nadir sparse hatası
        raise SolverError(f"spsolve başarısız ({case_id}): {exc}") from exc
    U1 = np.asarray(U1).ravel()

    P2 = K21 @ U1 + (K22 @ U2 if U2.any() else 0.0) - RHS2

    U_full = np.concatenate((U1, U2))
    P_full = np.concatenate((P1, P2))
    return StaticSolution(case_id=case_id, U=U_full, P=P_full)
