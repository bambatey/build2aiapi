"""Rijit diyafram kısıtları — master-slave dönüşümü.

Her diyafram (axis=Z için) bir düzlemde yer alan düğüm kümesidir; tümü
aynı rijit kat gibi hareket eder. Matematik:

    slave.ux = master.ux - (y_s - y_m) × master.rz
    slave.uy = master.uy + (x_s - x_m) × master.rz
    slave.rz = master.rz

Slave'in uz, rx, ry DOF'ları bağımsız kalır.

Uygulama: ``U_full = T @ U_red`` dönüşüm matrisi. Slave DOF'ları reduced
uzaydan çıkarılır. Yalnızca axis=Z destekleniyor (bina katı senaryosu).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..model.dto import ModelDTO
from .dof_numbering import DofMap

logger = logging.getLogger(__name__)


@dataclass
class DiaphragmTransform:
    T: sp.csr_matrix
    reduced_dofs: list[int]
    n_free_reduced: int
    n_total_reduced: int
    n_slaves_eliminated: int

    def apply_K(self, K: sp.csc_matrix) -> sp.csc_matrix:
        return (self.T.T @ K @ self.T).tocsc()

    def apply_vec(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(self.T.T @ v).ravel()

    def expand_U(self, U_red: np.ndarray) -> np.ndarray:
        return np.asarray(self.T @ U_red).ravel()


def build_diaphragm_transform(
    model: ModelDTO, dof_map: DofMap
) -> DiaphragmTransform | None:
    """Diyafram yoksa ``None`` (kimlik). Varsa T + sayımlar döner."""
    if not model.diaphragms:
        return None

    M = dof_map.n_total
    slave_equations: dict[int, list[tuple[int, float]]] = {}

    for diaph in model.diaphragms:
        if diaph.axis != "Z":
            logger.warning(
                "Diaphragm %s axis=%s — sadece Z destekli, atlanıyor.",
                diaph.name, diaph.axis,
            )
            continue
        joints = [j for j in diaph.joints if j in model.nodes]
        if len(joints) < 2:
            continue
        master = _pick_master(joints, model)
        if master is None:
            logger.warning(
                "Diaphragm %s: master seçilemedi.", diaph.name,
            )
            continue
        n_m = model.nodes[master]
        code_m = dof_map.codes[master]
        m_ux, m_uy, m_rz = code_m[0], code_m[1], code_m[5]

        for s in joints:
            if s == master:
                continue
            n_s = model.nodes[s]
            code_s = dof_map.codes[s]
            dx = n_s.x - n_m.x
            dy = n_s.y - n_m.y
            if not n_s.restraints[0]:
                slave_equations[code_s[0]] = [(m_ux, 1.0), (m_rz, -dy)]
            if not n_s.restraints[1]:
                slave_equations[code_s[1]] = [(m_uy, 1.0), (m_rz, dx)]
            if not n_s.restraints[5]:
                slave_equations[code_s[5]] = [(m_rz, 1.0)]

    if not slave_equations:
        return None

    slave_set = set(slave_equations)
    reduced_dofs = [i for i in range(M) if i not in slave_set]
    full_to_red = {i: k for k, i in enumerate(reduced_dofs)}

    n_slaves_free = sum(1 for i in slave_set if i < dof_map.n_free)
    n_free_red = dof_map.n_free - n_slaves_free
    n_total_red = len(reduced_dofs)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(M):
        if i in slave_equations:
            for master_full, coef in slave_equations[i]:
                if master_full in full_to_red:
                    rows.append(i)
                    cols.append(full_to_red[master_full])
                    data.append(coef)
        else:
            rows.append(i)
            cols.append(full_to_red[i])
            data.append(1.0)

    T = sp.csr_matrix((data, (rows, cols)), shape=(M, n_total_red))
    return DiaphragmTransform(
        T=T,
        reduced_dofs=reduced_dofs,
        n_free_reduced=n_free_red,
        n_total_reduced=n_total_red,
        n_slaves_eliminated=len(slave_set),
    )


def _pick_master(joints: list[int], model: ModelDTO) -> int | None:
    for j in joints:
        n = model.nodes[j]
        if not any(n.restraints[:2]) and not n.restraints[5]:
            return j
    return joints[0] if joints else None
