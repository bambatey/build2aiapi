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

from ..elements import FrameElement3D, FrameKernel, build_frame_kernels
from ..model.dto import DistributedLoadDTO, FrameSectionDTO, ModelDTO
from .dof_numbering import DofMap

logger = logging.getLogger(__name__)

# Standart yerçekimi ivmesi (m/s²). SAP ve TBDY varsayılanıyla uyumlu.
GRAVITY = 9.80665


def assemble_load_vectors(
    model: ModelDTO,
    dof_map: DofMap,
    frame_kernels: dict[int, FrameKernel] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Her yük durumu için ``(PS, RHS, US)`` döner.

    ``frame_kernels`` verilirse TE/T_inv/sw_w gibi değerler pre-computed
    cache'ten okunur. Yoksa caller'a saydam şekilde yerel bir cache
    oluşturulur (performans kaybı olmaz, eski API korunur).
    """
    if frame_kernels is None:
        frame_kernels = build_frame_kernels(model)

    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    M = dof_map.n_total

    # Area → çevre frame'lere dağıtılmış ek yük listesi (yük durumu bazında)
    area_to_frame_loads = _distribute_area_uniform_to_frames(model)

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

        # Elemanlara uygulanan yükler: yayılı + area-from-shell + öz ağırlık
        per_element_loads: dict[int, list[DistributedLoadDTO]] = {}
        for dl in case.distributed_loads:
            per_element_loads.setdefault(dl.element_id, []).append(dl)
        # Area uniform yükleri bu case için frame'lere eklenir
        for dl in area_to_frame_loads.get(case_id, []):
            per_element_loads.setdefault(dl.element_id, []).append(dl)

        frame_ids_to_process = set(per_element_loads.keys())
        if case.self_weight_factor:
            # Tüm frame elemanlarına öz ağırlık eklenir
            frame_ids_to_process.update(model.frame_elements.keys())

        for el_id in frame_ids_to_process:
            kernel = frame_kernels.get(el_id)
            if kernel is None:
                continue
            el_dto = kernel.element

            q_local_total = np.zeros(6)
            for dl in per_element_loads.get(el_id, []):
                q = _distributed_to_local_q_from_kernel(dl, kernel)
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
                q_local_total += _self_weight_local_q_from_kernel(
                    kernel, case.self_weight_factor,
                )

            if not np.any(q_local_total):
                continue

            # global_load_vector = T_inv @ local_load_vector(q)
            q_eq_local = _local_load_vector(q_local_total, kernel.length)
            b = kernel.T_inv @ q_eq_local
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


def _distributed_to_local_q_pure(
    dl: DistributedLoadDTO, TE: np.ndarray,
) -> np.ndarray | None:
    """SAP yayılı yükü, eleman lokalindeki q=[qx,qy,qz,mx,my,mz]'ye çevir.

    Pure fonksiyon — element yerine 3×3 ``TE`` (element axes transform)
    alır. Rust port adayı (# RUST_KERNEL_CANDIDATE Tier 3 helper).
    """
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

    local_vec = TE @ global_vec
    if dl.kind == "force":
        q[0:3] = local_vec
    else:
        q[3:6] = local_vec
    return q


def _distributed_to_local_q(
    dl: DistributedLoadDTO, element: FrameElement3D
) -> np.ndarray | None:
    """Geriye uyumlu sarmalayıcı — FrameElement3D API'sini korur."""
    return _distributed_to_local_q_pure(dl, element.element_axes_transform())


def _distributed_to_local_q_from_kernel(
    dl: DistributedLoadDTO, kernel: "FrameKernel",
) -> np.ndarray | None:
    """Kernel versiyonu — TE cache'ten okunur, yeniden hesap yok."""
    return _distributed_to_local_q_pure(dl, kernel.TE)


# RUST_KERNEL_CANDIDATE (Tier 3 helper)
def _local_load_vector(q: np.ndarray, L: float) -> np.ndarray:
    """Yayılı yük q=[qx,qy,qz,mx,my,mz] için 12-uzunluklu eş-nodal vektör.

    ``FrameElement3D.local_load_vector`` ile bire bir aynı matematik,
    ama pure fonksiyon (yalnızca q ve L'e bağlı).
    """
    qx, qy, qz, mx, my, mz = q
    L2 = L * L
    return np.asarray(
        [
            0.5 * qx * L,
            0.5 * qy * L - mz,
            0.5 * qz * L + my,
            0.5 * mx * L,
            -qz * L2 / 12,
            qy * L2 / 12,
            0.5 * qx * L,
            0.5 * qy * L + mz,
            0.5 * qz * L - my,
            0.5 * mx * L,
            qz * L2 / 12,
            -qy * L2 / 12,
        ]
    )


# -------------------------------- AREA → FRAME yük dağıtımı (MVP)
def _distribute_area_uniform_to_frames(
    model: ModelDTO,
) -> dict[str, list[DistributedLoadDTO]]:
    """SAP AREA LOADS - UNIFORM TO FRAME → ekvivalent frame distributed loads.

    Basit model (MVP): alan üstündeki toplam yük, o alanın çevresindeki
    kenar frame'lere eşit ``q = UnifLoad × area / perimeter`` (kN/m) olarak
    dağıtılır. Toplam yük korunur; gerçekçi tributary dağılım
    (trapezoidal/triangular) sonraki iterasyonda iyileştirilebilir.

    Alan-kenar frame'i tespiti: alan düğümleri ardışık ikili (i, j) için,
    eğer böyle bir frame varsa (JointI,JointJ ya da tersi), onu kenar say.
    """
    out: dict[str, list[DistributedLoadDTO]] = {}
    if not model.area_uniform_loads:
        return out

    # Frame hızlı arama: (node_a, node_b) → frame_id (yön-bağımsız)
    edge_to_frame: dict[frozenset[int], int] = {}
    for fid, el in model.frame_elements.items():
        if len(el.nodes) >= 2:
            edge_to_frame[frozenset([el.nodes[0], el.nodes[1]])] = fid

    for aul in model.area_uniform_loads:
        shell = model.shell_elements.get(aul.area_id)
        if shell is None or len(shell.nodes) < 3:
            continue
        # Alan hesabı (poligon, 3D noktalardan)
        coords = np.asarray([
            [model.nodes[nid].x, model.nodes[nid].y, model.nodes[nid].z]
            for nid in shell.nodes
        ])
        area = _polygon_area_3d(coords)
        if area <= 0:
            continue
        # Kenar frame'lerini bul
        perimeter_frames: list[tuple[int, float]] = []
        total_perim = 0.0
        for i in range(len(shell.nodes)):
            a = shell.nodes[i]
            b = shell.nodes[(i + 1) % len(shell.nodes)]
            frame_id = edge_to_frame.get(frozenset([a, b]))
            if frame_id is None:
                continue
            pa = np.asarray([model.nodes[a].x, model.nodes[a].y, model.nodes[a].z])
            pb = np.asarray([model.nodes[b].x, model.nodes[b].y, model.nodes[b].z])
            L = float(np.linalg.norm(pb - pa))
            perimeter_frames.append((frame_id, L))
            total_perim += L
        if total_perim <= 0 or not perimeter_frames:
            continue
        # q = toplam yük / perimetre — tüm kenar frame'lere aynı kN/m
        total_load = aul.magnitude * area
        q = total_load / total_perim
        # aul.direction → DistributedLoadDTO direction
        for frame_id, _L in perimeter_frames:
            out.setdefault(aul.load_pat, []).append(
                DistributedLoadDTO(
                    element_id=frame_id,
                    coord_sys="global",
                    direction=aul.direction,   # type: ignore[arg-type]
                    kind="force",
                    magnitude_a=q,
                    magnitude_b=q,
                )
            )
    return out


def _polygon_area_3d(pts: np.ndarray) -> float:
    """3D poligonun alanı (kenarların cross product'larının toplamı/2).

    Düzlemsel olmayan poligonlarda yaklaşık alan verir — bina döşemeleri
    için yeterlidir.
    """
    if pts.shape[0] < 3:
        return 0.0
    total = np.zeros(3)
    origin = pts[0]
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - origin
        v2 = pts[i + 1] - origin
        total = total + np.cross(v1, v2)
    return 0.5 * float(np.linalg.norm(total))


def _self_weight_local_q_pure(
    w: float, TE: np.ndarray,
) -> np.ndarray:
    """``w = ρ g A factor`` verildiğinde, global -Z yönündeki öz ağırlığı
    eleman lokalindeki q vektörüne dönüştür. Pure.
    """
    if w == 0.0:
        return np.zeros(6)
    # -Z yönünde
    global_vec = np.asarray([0.0, 0.0, -w])
    local_vec = TE @ global_vec
    q = np.zeros(6)
    q[0:3] = local_vec
    return q


def _self_weight_local_q(element: FrameElement3D, factor: float) -> np.ndarray:
    """Geriye uyumlu sarmalayıcı — FrameElement3D API'sini korur."""
    w = element.material.rho * GRAVITY * element.section.A * factor
    return _self_weight_local_q_pure(w, element.element_axes_transform())


def _self_weight_local_q_from_kernel(
    kernel: "FrameKernel", factor: float,
) -> np.ndarray:
    """Kernel versiyonu — ρgA ve TE cache'ten, sadece factor'e bağlı."""
    return _self_weight_local_q_pure(kernel.sw_w * factor, kernel.TE)
