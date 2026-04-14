"""Global rijitlik matrisi birleştirme.

Her frame elemanın 12×12 global K_e matrisini COO sparse formatında
toplar, sonra CSC'ye çevirir. Shell elemanları henüz desteklenmemektedir
(uyarı verilir, atlanır).
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from ..elements import FrameElement3D, build_shell
from ..model.dto import FrameSectionDTO, ModelDTO, ShellSectionDTO
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

    # Shell elemanları — membran + plate bending birleşik
    shell_count = 0
    skipped_shells = 0
    for el_dto in model.shell_elements.values():
        if len(el_dto.nodes) != 4:
            # Q9 / T3 / T6 şu an desteklenmiyor
            skipped_shells += 1
            continue
        section = model.sections.get(el_dto.section_id)
        material = model.materials.get(el_dto.material_id)
        if not isinstance(section, ShellSectionDTO) or material is None:
            skipped_shells += 1
            continue
        if any(n not in model.nodes for n in el_dto.nodes):
            skipped_shells += 1
            continue
        shell_nodes = [model.nodes[n] for n in el_dto.nodes]
        try:
            shell = build_shell(shell_nodes, section, material)
            K_sh = shell.global_stiffness()   # 24×24
        except Exception as exc:
            logger.warning("Shell %s build başarısız: %s", el_dto.id, exc)
            skipped_shells += 1
            continue
        # Scatter: shell her düğümde 6 DOF
        code: list[int] = []
        for nid in el_dto.nodes:
            code.extend(dof_map.codes[nid])
        rows += np.repeat(code, 24).tolist()
        cols += code * 24
        data += K_sh.flatten().tolist()
        shell_count += 1

    if shell_count:
        logger.info("Shell katkısı: %d eleman K'ya eklendi.", shell_count)
    if skipped_shells:
        logger.warning(
            "%d shell elemanı atlandı (Q4 olmayan veya eksik veri).",
            skipped_shells,
        )

    M = dof_map.n_total
    return sp.coo_matrix((data, (rows, cols)), shape=(M, M), dtype=float).tocsc()
