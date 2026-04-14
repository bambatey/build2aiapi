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


GRAVITY = 9.80665


def _default_rho(E: float) -> float:
    """Malzeme yoğunluğu 0 ise E'den türetilen yaklaşık varsayılan."""
    if 180e6 <= E <= 220e6:
        return 7.85
    if 20e6 <= E <= 50e6:
        return 2.5
    return 0.0


def assemble_mass(
    model: ModelDTO,
    dof_map: DofMap,
    load_vectors: dict[str, tuple] | None = None,
) -> sp.csc_matrix:
    """Diagonal (lumped) kütle matrisi.

    Strateji (SAP MASS SOURCE ile tutarlı):
    1. ``model.mass_source`` varsa, bayraklarına göre davran:
       - ``from_elements=True`` → materyal ρ × V'den eleman kütlesi
       - ``from_loads=True`` → belirtilen load pattern'lerin Z yönündeki
         nodal yüklerinin toplamı / g → düğüm kütlesi
    2. ``model.mass_source`` yoksa (eski davranış): tüm elemanlardan
       materyal ρ × V.

    Lumped: her düğüm kütlesi 3 öteleme DOF'una (ux, uy, uz) eşit paylaştırılır.
    """
    M = dof_map.n_total
    diag = np.zeros(M)

    ms = model.mass_source
    use_elements = (ms is None) or ms.from_elements
    use_loads = ms is not None and ms.from_loads

    fallback_used: set[str] = set()

    # ------------------------------- (1) Element bazlı kütle (ρ × V)
    if use_elements:
        for el in model.frame_elements.values():
            section = model.sections.get(el.section_id)
            material = model.materials.get(el.material_id)
            if not isinstance(section, FrameSectionDTO) or material is None:
                continue
            rho = material.rho
            if rho <= 0:
                rho = _default_rho(material.E)
                if rho > 0:
                    fallback_used.add(material.id)
            if rho <= 0:
                continue
            n1, n2 = model.nodes[el.nodes[0]], model.nodes[el.nodes[1]]
            L = np.sqrt(
                (n2.x - n1.x) ** 2 + (n2.y - n1.y) ** 2 + (n2.z - n1.z) ** 2
            )
            m_total = rho * section.A * L
            if m_total <= 0:
                continue
            half = 0.5 * m_total
            for node_id in (n1.id, n2.id):
                code = dof_map.codes[node_id]
                diag[code[0]] += half
                diag[code[1]] += half
                diag[code[2]] += half

        from ..model.dto import ShellSectionDTO
        for el in model.shell_elements.values():
            section = model.sections.get(el.section_id)
            material = model.materials.get(el.material_id)
            if not isinstance(section, ShellSectionDTO) or material is None:
                continue
            if len(el.nodes) != 4:
                continue
            rho = material.rho
            if rho <= 0:
                rho = _default_rho(material.E)
                if rho > 0:
                    fallback_used.add(material.id)
            if rho <= 0:
                continue
            pts = np.asarray([
                [model.nodes[n].x, model.nodes[n].y, model.nodes[n].z]
                for n in el.nodes
            ])
            total = np.zeros(3)
            for i in range(1, len(pts) - 1):
                total = total + np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])
            area = 0.5 * float(np.linalg.norm(total))
            m_total = rho * area * section.thickness
            if m_total <= 0:
                continue
            quarter = 0.25 * m_total
            for nid in el.nodes:
                code = dof_map.codes[nid]
                diag[code[0]] += quarter
                diag[code[1]] += quarter
                diag[code[2]] += quarter

    # --------------------- (2) Yük bazlı kütle (SAP MASS SOURCE: Loads=Yes)
    if use_loads and ms is not None and load_vectors:
        for ms_pat in ms.load_patterns:
            lv = load_vectors.get(ms_pat.load_pat)
            if lv is None:
                logger.warning(
                    "MASS SOURCE: LoadPat=%s çözüm yükleri arasında bulunamadı.",
                    ms_pat.load_pat,
                )
                continue
            PS, RHS, _US = lv
            F_total = PS + RHS
            # Her düğümün Z yönündeki toplam yükü → kütle = |Fz| / g × mult
            # (SAP konvansiyonu: yalnızca düşey yük bileşeni kütleye çevrilir;
            #  yatay noktasal yükler kütle üretmez)
            for nid, code in dof_map.codes.items():
                fz = float(F_total[code[2]])
                if fz == 0:
                    continue
                m_add = abs(fz) / GRAVITY * ms_pat.multiplier
                # Lumped: aynı kütle 3 öteleme DOF'una
                diag[code[0]] += m_add
                diag[code[1]] += m_add
                diag[code[2]] += m_add
        logger.info(
            "MASS SOURCE yük bazlı kütle uygulandı: pattern'ler=%s",
            [(p.load_pat, p.multiplier) for p in ms.load_patterns],
        )

    if fallback_used:
        logger.warning(
            "Kütle: %s materyallerinde ρ=0 olduğu için modül içi varsayılan "
            "uygulandı (çelik 7.85, beton 2.5 t/m³). SAP'ta MATERIAL "
            "PROPERTIES 02 - BASIC MECHANICAL PROPERTIES'teki UnitMass'leri "
            "doldurun ki analiz kesin olsun.",
            sorted(fallback_used),
        )

    if diag.sum() == 0:
        logger.warning(
            "Kütle matrisi tamamen sıfır — malzeme rho'ları ya da kesit "
            "atamaları eksik olabilir."
        )
    return sp.diags(diag, format="csc")
