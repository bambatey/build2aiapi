"""U sistem vektöründen düğüm yer değiştirmelerini geri topla."""

from __future__ import annotations

import numpy as np

from ..assembly.dof_numbering import DofMap


def node_displacements(
    U: np.ndarray, dof_map: DofMap
) -> dict[int, dict[str, float]]:
    """Her düğüm için ``{ux, uy, uz, rx, ry, rz}`` sözlüğü döner."""
    keys = ("ux", "uy", "uz", "rx", "ry", "rz")
    out: dict[int, dict[str, float]] = {}
    for nid, code in dof_map.codes.items():
        out[nid] = {k: float(U[code[i]]) for i, k in enumerate(keys)}
    return out
