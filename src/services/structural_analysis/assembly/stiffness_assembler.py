"""Global rijitlik matrisi birleştirme.

Her frame elemanın 12×12 global K_e matrisini COO sparse formatında
toplar, sonra CSC'ye çevirir. Shell elemanları henüz desteklenmemektedir
(uyarı verilir, atlanır).
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from ..elements import FrameElement3D
from ..model.dto import FrameSectionDTO, ModelDTO
from .dof_numbering import DofMap

logger = logging.getLogger(__name__)


def assemble_stiffness(model: ModelDTO, dof_map: DofMap) -> sp.csc_matrix:
    """Frame elemanlarının katkısıyla global K matrisini oluştur."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for el_dto in model.frame_elements.values():
        section = model.sections.get(el_dto.section_id)
        if not isinstance(section, FrameSectionDTO):
            logger.warning(
                "Frame %s için FrameSection bulunamadı (section_id=%r), atlandı.",
                el_dto.id, el_dto.section_id,
            )
            continue
        material = model.materials.get(el_dto.material_id)
        if material is None:
            logger.warning(
                "Frame %s için malzeme bulunamadı (material_id=%r), atlandı.",
                el_dto.id, el_dto.material_id,
            )
            continue
        ni, nj = el_dto.nodes
        element = FrameElement3D(
            element=el_dto,
            node_i=model.nodes[ni],
            node_j=model.nodes[nj],
            section=section,
            material=material,
        )
        K_e = element.global_stiffness()
        code = dof_map.element_code(ni, nj)
        # COO scatter
        rows += np.repeat(code, 12).tolist()
        cols += code * 12
        data += K_e.flatten().tolist()

    if model.shell_elements:
        logger.warning(
            "%d shell elemanı şu anki motor sürümünde desteklenmiyor, K'ya katkıları atlandı.",
            len(model.shell_elements),
        )

    M = dof_map.n_total
    return sp.coo_matrix((data, (rows, cols)), shape=(M, M), dtype=float).tocsc()
