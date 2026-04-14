"""Kütle matrisi birleştirme — lumped (yığılı) kütle.

Her frame elemanın kütlesi (``ρ × A × L``) iki ucuna yarı yarıya dağıtılır;
yalnızca öteleme DOF'larına uygulanır (ux, uy, uz). Dönme DOF'larına
atalet moment katkısı şu an atlanmakta — modal ilk modlar için yeterli
yaklaşım. Öz ağırlık çarpanı (``self_weight_factor``) modal analizde rol
oynamaz; sadece geometrik ve malzeme verisi kullanılır.

Uyarı: tüm kabuk elemanları şu an kütleye dahil değil.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from ..model.dto import FrameSectionDTO, ModelDTO
from .dof_numbering import DofMap

logger = logging.getLogger(__name__)


def assemble_mass(model: ModelDTO, dof_map: DofMap) -> sp.csc_matrix:
    """Diagonal (lumped) kütle matrisi — sadece öteleme DOF'larında."""
    M = dof_map.n_total
    diag = np.zeros(M)

    for el in model.frame_elements.values():
        section = model.sections.get(el.section_id)
        material = model.materials.get(el.material_id)
        if not isinstance(section, FrameSectionDTO) or material is None:
            continue
        n1, n2 = model.nodes[el.nodes[0]], model.nodes[el.nodes[1]]
        L = np.sqrt(
            (n2.x - n1.x) ** 2 + (n2.y - n1.y) ** 2 + (n2.z - n1.z) ** 2
        )
        # Kütle: ρ × A × L (SAP kN-m'de rho = UnitMass, t/m³)
        m_total = material.rho * section.A * L
        if m_total <= 0:
            continue
        half = 0.5 * m_total
        # Her uca 3 öteleme DOF'una half ekle
        for node_id in (n1.id, n2.id):
            code = dof_map.codes[node_id]
            diag[code[0]] += half   # ux
            diag[code[1]] += half   # uy
            diag[code[2]] += half   # uz

    if diag.sum() == 0:
        logger.warning(
            "Kütle matrisi tamamen sıfır — malzeme rho'ları ya da kesit "
            "atamaları eksik olabilir."
        )
    return sp.diags(diag, format="csc")
