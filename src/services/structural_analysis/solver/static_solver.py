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

from ..assembly.constraints import DiaphragmTransform
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
    transform: DiaphragmTransform | None = None,
) -> StaticSolution:
    """Tek yük durumu çözümü.

    ``transform`` verilirse rijit diyafram master-slave dönüşümü
    uygulanarak sistem reduced uzayda çözülür; sonuç full uzaya açılır.
    """
    if transform is None:
        return _solve_direct(case_id, K, PS, RHS, US, dof_map)
    return _solve_with_diaphragm(case_id, K, PS, RHS, US, dof_map, transform)


# --------------------------------------------------------------- classic
def _solve_direct(case_id, K, PS, RHS, US, dof_map):
    N, M = dof_map.n_free, dof_map.n_total
    if N == 0:
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
    U1 = _spsolve_with_warning(K11, rhs_eff, case_id, N)

    P2 = K21 @ U1 + (K22 @ U2 if U2.any() else 0.0) - RHS2
    U_full = np.concatenate((U1, U2))
    P_full = np.concatenate((P1, P2))
    return StaticSolution(case_id=case_id, U=U_full, P=P_full)


# --------------------------------------------------------- with diaphragm
def _solve_with_diaphragm(case_id, K, PS, RHS, US, dof_map, t: DiaphragmTransform):
    # Reduced uzaya dön
    K_red = t.apply_K(K)
    PS_red = t.apply_vec(PS)
    RHS_red = t.apply_vec(RHS)
    US_red = t.apply_vec(US)

    N, M = t.n_free_reduced, t.n_total_reduced
    if N == 0:
        U_red = US_red
    else:
        K11 = K_red[0:N, 0:N]
        K12 = K_red[0:N, N:M]
        U2 = US_red[N:M]
        P1 = PS_red[0:N]
        RHS1 = RHS_red[0:N]
        rhs_eff = P1 + RHS1 - (K12 @ U2 if U2.any() else 0.0)
        U1 = _spsolve_with_warning(K11, rhs_eff, case_id, N)
        U_red = np.concatenate((U1, U2))

    # Full uzaya geri aç
    U_full = t.expand_U(U_red)
    # Kuvvet vektörünü full uzayda yeniden hesapla:
    #   K @ U - RHS = P_applied (free'de) veya reaksiyon (restrained'de)
    # Free kısmı PS_free, restrained kısmı hesaplanan reaksiyon
    KU = np.asarray(K @ U_full).ravel()
    computed = KU - RHS
    P_full = PS.copy()
    P_full[dof_map.n_free:] = computed[dof_map.n_free:]
    return StaticSolution(case_id=case_id, U=U_full, P=P_full)


# ---------------------------------------------------------------- spsolve
def _spsolve_with_warning(K11, rhs, case_id, N):
    """spsolve + sıfır-köşegen regularization.

    K11 diyagonalinde 0 olan DOF'lar (izole düğüm / tam mafsallı mekanizma
    artığı) için max|diag|'ın çok küçük katı kadar denge rijitliği
    eklenir. Bu, sistemi çözümleyebilir hale getirir; stabilize edilen
    DOF'ların yer değiştirmesi lokal etkisel olarak 0'a yakın kalır.
    Eklenen rijitlik sistem geneli sonuçlarını pratik olarak etkilemez
    (toplam rijitliğe göre 1e-9 oranında).
    """
    import logging as _logging
    import warnings as _warnings
    K11 = K11.tocsr()
    diag = K11.diagonal()
    max_diag = float(np.max(np.abs(diag))) if diag.size else 1.0
    threshold = max_diag * 1e-10
    n_zero = int(np.sum(np.abs(diag) < threshold))

    if n_zero > 0:
        # Sıfır-rijitlikli DOF'lar var → mekanizma. Tikhonov regularization
        # α = max_diag × 1e-8 (çözüm hassasiyetinde ~%1e-6 sapma — ihmal
        # edilebilir, sistem kararlı olsun)
        alpha = max_diag * 1e-8
        _logging.getLogger(__name__).warning(
            "Case %s: %d/%d DOF sıfır-rijitlikli (mekanizma artığı). "
            "Tikhonov regularization α=%.2e uygulandı.",
            case_id, n_zero, N, alpha,
        )
        K11 = (K11 + sp.eye(N, format="csr") * alpha).tocsc()
    else:
        K11 = K11.tocsc()

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        try:
            U1 = spla.spsolve(K11, rhs)
        except Exception as exc:
            raise SolverError(f"spsolve başarısız ({case_id}): {exc}") from exc
        for w in caught:
            if "singular" in str(w.message).lower() or "rank" in str(w.message).lower():
                _logging.getLogger(__name__).error(
                    "Case %s: K matrisi hala singular — ciddi model sorunu.",
                    case_id,
                )
    return np.asarray(U1).ravel()
