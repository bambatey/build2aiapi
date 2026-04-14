"""Serbestlik numaralama — partitioned strateji.

Serbest (tutulmamış) DOF'lar [0..N-1] aralığına, tutulu DOF'lar [N..M-1]
aralığına yerleştirilir. Bu, çözümde ``K11 U1 = P1 + RHS1 - K12 U2``
bölümlenmesini doğrudan destekler.

SEA_Book/solver/__init__.py ile aynı sıralama kuralı: düğümler dict
iterasyon sırası, her düğümde DOF sırası [ux, uy, uz, rx, ry, rz].
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.dto import ModelDTO


@dataclass(frozen=True)
class DofMap:
    """Düğüm DOF'ları → global indeks eşlemesi.

    Attributes:
        codes: ``{node_id: [6 int kod]}`` — her DOF için global indeks.
        n_free: Serbest DOF sayısı (N).
        n_total: Toplam DOF sayısı (M).
    """

    codes: dict[int, list[int]]
    n_free: int
    n_total: int

    def element_code(self, node_i: int, node_j: int) -> list[int]:
        """12 elemanlık frame kod vektörü (düğüm i + düğüm j)."""
        return self.codes[node_i] + self.codes[node_j]


def number_dofs(model: ModelDTO) -> DofMap:
    """ModelDTO içindeki düğümlere partitioned DOF numaralaması uygula."""
    codes: dict[int, list[int]] = {nid: [-1] * 6 for nid in model.nodes}
    m = 0
    # 1. pas: serbestler
    for nid, node in model.nodes.items():
        for i, restrained in enumerate(node.restraints):
            if not restrained:
                codes[nid][i] = m
                m += 1
    n_free = m
    # 2. pas: tutulular
    for nid, node in model.nodes.items():
        for i, restrained in enumerate(node.restraints):
            if restrained:
                codes[nid][i] = m
                m += 1
    return DofMap(codes=codes, n_free=n_free, n_total=m)
