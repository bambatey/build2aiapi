"""Yük vektörü birleştirme — her yük durumu için PS (nodal) + RHS (elemansal).

``PS[code]``: düğüme etki eden dış tekil kuvvet. Düğümün ``euler_zyx``
değeri varsa global kuvvet, düğüm yerel eksenine dönüştürülür.

``RHS[code]``: elemansal yayılı yük + öz ağırlık katkıları. Her yük önce
elemanın lokal eksenine rotated edilir, sonra ``global_load_vector``
üzerinden 12-DOF eş-nodal vektöre dönüştürülüp code DOF'larına dağıtılır.

Desteklenen yayılı yük yönleri:
    - ``local_1/2/3``: direkt atanır
    - ``x/y/z``: global kuvvet, eleman lokaline döndürülür
    - ``gravity``: global -Z yönünde, eleman lokaline döndürülür

Öz ağırlık: ``load_case.self_weight_factor × ρ × g × A`` (kN/m) global -Z
yönünde yayılı yük olarak eklenir.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..elements import FrameElement3D
from ..model.dto import DistributedLoadDTO, FrameSectionDTO, ModelDTO
from .dof_numbering import DofMap

logger = logging.getLogger(__name__)

# Standart yerçekimi ivmesi (m/s²). SAP ve TBDY varsayılanıyla uyumlu.
GRAVITY = 9.80665


def assemble_load_vectors(
    model: ModelDTO, dof_map: DofMap
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Her yük durumu için ``(PS, RHS, US)`` döner."""
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    M = dof_map.n_total

    for case_id, case in model.load_cases.items():
        PS = np.zeros(M)
        RHS = np.zeros(M)
        US = np.zeros(M)

        # Nodal tekil yükler
        for pl in case.point_loads:
            if pl.node_id not in dof_map.codes:
                continue
            node = model.nodes[pl.node_id]
            code = dof_map.codes[pl.node_id]
            values_local = _rotate_to_node_local(pl.values, node.euler_zyx)
            for i, v in enumerate(values_local):
                PS[code[i]] += v

        # Elemanlara uygulanan yükler: yayılı + öz ağırlık
        # Elemanı yeniden inşa edip birden çok yük katkısını toplu işleyelim.
        per_element_loads: dict[int, list[DistributedLoadDTO]] = {}
        for dl in case.distributed_loads:
            per_element_loads.setdefault(dl.element_id, []).append(dl)

        frame_ids_to_process = set(per_element_loads.keys())
        if case.self_weight_factor:
            # Tüm frame elemanlarına öz ağırlık eklenir
            frame_ids_to_process.update(model.frame_elements.keys())

        for el_id in frame_ids_to_process:
            el_dto = model.frame_elements.get(el_id)
            if el_dto is None:
                continue
            section = model.sections.get(el_dto.section_id)
            material = model.materials.get(el_dto.material_id)
            if not isinstance(section, FrameSectionDTO) or material is None:
                continue
            element = FrameElement3D(
                element=el_dto,
                node_i=model.nodes[el_dto.nodes[0]],
                node_j=model.nodes[el_dto.nodes[1]],
                section=section,
                material=material,
            )

            q_local_total = np.zeros(6)
            for dl in per_element_loads.get(el_id, []):
                q = _distributed_to_local_q(dl, element)
                if q is None:
                    logger.warning(
                        "Load case %s, element %d: yayılı yük henüz desteklenmiyor "
                        "(dir=%s, kind=%s, trapezoidal=%s).",
                        case_id, el_id, dl.direction, dl.kind,
                        dl.magnitude_a != dl.magnitude_b,
                    )
                    continue
                q_local_total += q

            if case.self_weight_factor:
                sw = _self_weight_local_q(element, case.self_weight_factor)
                q_local_total += sw

            if not np.any(q_local_total):
                continue

            b = element.global_load_vector(q_local_total.tolist())
            code = dof_map.element_code(el_dto.nodes[0], el_dto.nodes[1])
            for i, v in enumerate(b):
                RHS[code[i]] += v

        results[case_id] = (PS, RHS, US)

    if not results:
        results["_empty"] = (np.zeros(M), np.zeros(M), np.zeros(M))

    return results


# ------------------------------------------------------------------- helpers
def _rotate_to_node_local(values: list[float], euler_zyx: tuple[float, float, float]) -> list[float]:
    """Global eksendeki [Fx,Fy,Fz,Mx,My,Mz]'i düğümün yerel eksenine çevir."""
    if all(a == 0.0 for a in euler_zyx):
        return list(values)
    Fx, Fy, Fz, Mx, My, Mz = values
    T = _euler_matrix(euler_zyx)
    T_inv = np.linalg.inv(T)
    f_local = T_inv @ [Fx, Fy, Fz]
    m_local = T_inv @ [Mx, My, Mz]
    return [*f_local.tolist(), *m_local.tolist()]


def _euler_matrix(euler_zyx: tuple[float, float, float]) -> np.ndarray:
    ezd, eyd, exd = euler_zyx
    rx, ry, rz = math.radians(exd), math.radians(eyd), math.radians(ezd)
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)
    return np.asarray(
        [
            [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy],
        ]
    )


def _distributed_to_local_q(
    dl: DistributedLoadDTO, element: FrameElement3D
) -> np.ndarray | None:
    """SAP yayılı yükü, eleman lokalindeki q=[qx,qy,qz,mx,my,mz]'ye çevir."""
    if dl.magnitude_a != dl.magnitude_b:
        return None  # trapez henüz yok
    mag = dl.magnitude_a
    q = np.zeros(6)

    if dl.direction.startswith("local_"):
        idx = int(dl.direction.split("_", 1)[1]) - 1  # local_1 → 0
        offset = 0 if dl.kind == "force" else 3
        q[offset + idx] = mag
        return q

    # Global yönde tanımlı — eleman lokaline döndür
    global_vec = np.zeros(3)
    if dl.direction == "x":
        global_vec[0] = mag
    elif dl.direction == "y":
        global_vec[1] = mag
    elif dl.direction == "z":
        global_vec[2] = mag
    elif dl.direction == "gravity":
        # "Gravity" = global -Z yönünde magnitude kadar (SAP konvansiyonu:
        # magnitude pozitifken aşağı doğru çeker)
        global_vec[2] = -mag
    else:
        return None

    TE = element.element_axes_transform()
    local_vec = TE @ global_vec
    if dl.kind == "force":
        q[0:3] = local_vec
    else:
        q[3:6] = local_vec
    return q


def _self_weight_local_q(element: FrameElement3D, factor: float) -> np.ndarray:
    """Öz ağırlık katkısını eleman lokalindeki q vektörüne dönüştür.

    w = ρ × g × A × factor  (kN/m, global -Z yönünde)
    """
    rho = element.material.rho
    A = element.section.A
    w = rho * GRAVITY * A * factor
    if w == 0.0:
        return np.zeros(6)
    TE = element.element_axes_transform()
    # -Z yönünde
    global_vec = np.asarray([0.0, 0.0, -w])
    local_vec = TE @ global_vec
    q = np.zeros(6)
    q[0:3] = local_vec
    return q
