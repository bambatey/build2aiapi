"""Mesnet reaksiyonlarını P (sistem kuvvet) vektöründen geri topla.

P vektörü partitioned: [0..N-1] dış uygulanan kuvvetler, [N..M-1] mesnet
reaksiyonları. Bu modül sadece reaksiyon kısmını düğüm başına sözlüğe çevirir.
"""

from __future__ import annotations

import numpy as np

from ..assembly.dof_numbering import DofMap
from ..model.dto import ModelDTO


def node_reactions(
    P: np.ndarray, dof_map: DofMap, model: ModelDTO
) -> dict[int, dict[str, float]]:
    """Tutulu DOF'u olan her düğüm için reaksiyon vektörü döner.

    Sadece gerçek mesneti olan (en az bir restraint=True) düğümler listelenir.
    """
    keys = ("fx", "fy", "fz", "mx", "my", "mz")
    out: dict[int, dict[str, float]] = {}
    for nid, node in model.nodes.items():
        if not any(node.restraints):
            continue
        code = dof_map.codes[nid]
        out[nid] = {k: float(P[code[i]]) for i, k in enumerate(keys)}
    return out
