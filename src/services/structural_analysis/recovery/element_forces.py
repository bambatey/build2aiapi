"""3D frame eleman kesit tesirleri — her eleman boyunca istasyon bazlı çıktı.

Her frame için:
    u_local   = T @ u_global                   (12-vector)
    f_local   = K_local_with_releases @ u_local - q_eq_local
    P(x), V2(x), V3(x), T(x), M2(x), M3(x)      x ∈ [0..L]

SAP/mühendislik konvansiyonu (doğrulanmış):
    P > 0   = çekme (axial tension)
    M3 > 0  = alt-lif çekmesi (sagging for gravity beam)

İstasyon formülleri (uniform q_local = [qx, qy, qz, mx, my, mz]):
    P(x)  = -f_local[0] - qx·x
    V2(x) = -f_local[1] - qy·x
    V3(x) = -f_local[2] - qz·x
    T(x)  = -f_local[3] - mx·x
    M2(x) = -f_local[4] + x·f_local[2] + qz·x²/2 - my·x
    M3(x) = -f_local[5] + x·f_local[1] + qy·x²/2 - mz·x

Açıklık extremumu (shear=0 noktası):
    dM3/dx = f_local[1] + qy·x - mz = 0 → x* = (mz - f_local[1]) / qy
    (benzer M2 için)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assembly.dof_numbering import DofMap
from ..assembly.load_assembler import (
    _distribute_area_uniform_to_frames,
    _distributed_to_local_q_from_kernel,
    _local_load_vector,
    _self_weight_local_q_from_kernel,
)
from ..elements import FrameKernel, build_frame_kernels
from ..model.dto import CombinationDTO, ModelDTO


@dataclass
class StationForce:
    x: float          # I-ucundan uzaklık (m)
    x_rel: float      # x / L (0..1)
    P: float          # axial (+ çekme)
    V2: float         # local-2 shear
    V3: float         # local-3 shear
    T: float          # torsion (about local-1)
    M2: float         # bending about local-2
    M3: float         # bending about local-3 (+ sagging)


@dataclass
class ElementForces:
    element_id: int
    length: float
    node_i: int
    node_j: int
    stations: list[StationForce]
    # Uç kuvvetleri (hızlı UI erişimi)
    P_I: float
    V2_I: float
    V3_I: float
    T_I: float
    M2_I: float
    M3_I: float
    P_J: float
    V2_J: float
    V3_J: float
    T_J: float
    M2_J: float
    M3_J: float
    # Açıklık içi extremum (0<x<L), yoksa 0
    M3_span_ext: float         # imzalı (sagging için +)
    M3_span_ext_x: float
    M2_span_ext: float
    M2_span_ext_x: float
    # Özet maksimumlar
    V2_max_abs: float
    V3_max_abs: float
    # Eleman üzerindeki etkin yayılı yük (UI tooltip/detay için)
    q_local: list[float]       # [qx, qy, qz, mx, my, mz]


# ----------------------------------------------------- per-case q_local
def build_case_q_local(
    case_id: str,
    model: ModelDTO,
    frame_kernels: dict[int, FrameKernel] | None = None,
) -> dict[int, np.ndarray]:
    """Bir yük durumunun her frame için toplam lokal yayılı yükü.

    Distributed + area-uniform-to-frame + self-weight birleşik.
    ``frame_kernels`` cache'i verilirse TE/ρgA yeniden hesaplanmaz.
    """
    case = model.load_cases.get(case_id)
    if case is None:
        return {}
    if frame_kernels is None:
        frame_kernels = build_frame_kernels(model)

    per_el: dict[int, list] = {}
    for dl in case.distributed_loads:
        per_el.setdefault(dl.element_id, []).append(dl)
    for dl in _distribute_area_uniform_to_frames(model).get(case_id, []):
        per_el.setdefault(dl.element_id, []).append(dl)

    out: dict[int, np.ndarray] = {}
    sw_factor = case.self_weight_factor
    for el_id, kernel in frame_kernels.items():
        q = np.zeros(6)
        for dl in per_el.get(el_id, []):
            q_dl = _distributed_to_local_q_from_kernel(dl, kernel)
            if q_dl is not None:
                q += q_dl
        if sw_factor:
            q += _self_weight_local_q_from_kernel(kernel, sw_factor)
        if np.any(q):
            out[el_id] = q
    return out


def combine_case_q_local(
    combo: CombinationDTO,
    base_q: dict[str, dict[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Kombinasyon için q_local = Σ factor × q_base."""
    out: dict[int, np.ndarray] = {}
    for case_id, factor in combo.factors.items():
        q_case = base_q.get(case_id, {})
        for el_id, q in q_case.items():
            if el_id in out:
                out[el_id] = out[el_id] + factor * q
            else:
                out[el_id] = factor * q
    return out


# --------------------------------------------------- station formülleri
# RUST_KERNEL_CANDIDATE — Tier 1 (her frame × her case × her station)
def _station(
    x: float, x_rel: float, f_local: np.ndarray, q: np.ndarray,
) -> StationForce:
    qx, qy, qz, mx, my, mz = q
    P = -f_local[0] - qx * x
    V2 = -f_local[1] - qy * x
    V3 = -f_local[2] - qz * x
    T_ = -f_local[3] - mx * x
    M2 = -f_local[4] + x * f_local[2] + qz * x * x / 2.0 - my * x
    M3 = -f_local[5] + x * f_local[1] + qy * x * x / 2.0 - mz * x
    return StationForce(
        x=float(x), x_rel=float(x_rel),
        P=float(P), V2=float(V2), V3=float(V3),
        T=float(T_), M2=float(M2), M3=float(M3),
    )


# RUST_KERNEL_CANDIDATE — Tier 1 (per frame × per case)
def _interior_extremum(
    f_local_i: float, f_local_force: float,
    q_force: float, q_moment: float,
    L: float, kind: str,
) -> tuple[float, float]:
    """dM/dx = 0 noktasını (0<x<L) bul.

    kind="M3": M3(x) = -f_local[5] + x·f_local[1] + qy·x²/2 - mz·x
               dM3/dx = f_local[1] + qy·x - mz
    kind="M2": M2(x) = -f_local[4] + x·f_local[2] + qz·x²/2 - my·x
               dM2/dx = f_local[2] + qz·x - my

    Dönüş: (M değeri imzalı, x konumu). Interior'da extremum yoksa (0, 0).
    """
    if abs(q_force) < 1e-12:
        return 0.0, 0.0
    x_c = (q_moment - f_local_force) / q_force
    if not (1e-9 < x_c < L - 1e-9):
        return 0.0, 0.0
    if kind == "M3":
        M = -f_local_i + x_c * f_local_force + q_force * x_c * x_c / 2.0 - q_moment * x_c
    else:  # M2
        M = -f_local_i + x_c * f_local_force + q_force * x_c * x_c / 2.0 - q_moment * x_c
    return float(M), float(x_c)


# ------------------------------------------------- asıl compute fonksiyonu
def compute_element_forces(
    U: np.ndarray,
    dof_map: DofMap,
    model: ModelDTO,
    q_local_per_element: dict[int, np.ndarray] | None = None,
    n_stations: int = 3,
    frame_kernels: dict[int, FrameKernel] | None = None,
) -> list[ElementForces]:
    """Verilen sistem U vektörü ve q_local sözlüğü için kesit tesirleri.

    ``q_local_per_element``: her frame için 6-uzunluklu yayılı yük vektörü
    (build_case_q_local / combine_case_q_local ile hazırlanır). Yoksa q=0.
    ``n_stations``: I..J arası örnekleme (min 2, varsayılan 3 — I/mid/J;
    docs/architecture/01-performance.md Faz 1 önerisi).
    ``frame_kernels``: pre-computed cache; yoksa oluşturulur.
    """
    q_map = q_local_per_element or {}
    n_stations = max(2, n_stations)
    if frame_kernels is None:
        frame_kernels = build_frame_kernels(model)
    out: list[ElementForces] = []

    for el_id, kernel in frame_kernels.items():
        el_dto = kernel.element
        L = kernel.length

        q = q_map.get(el_id, np.zeros(6))

        code = dof_map.element_code(el_dto.nodes[0], el_dto.nodes[1])
        u_global = np.asarray([U[c] for c in code], dtype=float)
        u_local = kernel.T @ u_global

        q_eq = _local_load_vector(q, L)
        f_local = kernel.K_local_released @ u_local - q_eq

        # NaN/inf guard — singular K çözümleri pipeline başka yerde
        # logluyor; buradaki değerleri 0'a sanitize ediyoruz.
        if not np.all(np.isfinite(f_local)):
            f_local = np.where(np.isfinite(f_local), f_local, 0.0)

        stations: list[StationForce] = []
        for i in range(n_stations):
            x_rel = i / (n_stations - 1)
            stations.append(_station(x_rel * L, x_rel, f_local, q))

        # Açıklık içi extremum
        M3_ext, M3_ext_x = _interior_extremum(
            f_local[5], f_local[1], q[1], q[5], L, "M3",
        )
        M2_ext, M2_ext_x = _interior_extremum(
            f_local[4], f_local[2], q[2], q[4], L, "M2",
        )

        V2_vals = [s.V2 for s in stations]
        V3_vals = [s.V3 for s in stations]

        out.append(ElementForces(
            element_id=el_id,
            length=float(L),
            node_i=el_dto.nodes[0],
            node_j=el_dto.nodes[1],
            stations=stations,
            P_I=stations[0].P, V2_I=stations[0].V2, V3_I=stations[0].V3,
            T_I=stations[0].T, M2_I=stations[0].M2, M3_I=stations[0].M3,
            P_J=stations[-1].P, V2_J=stations[-1].V2, V3_J=stations[-1].V3,
            T_J=stations[-1].T, M2_J=stations[-1].M2, M3_J=stations[-1].M3,
            M3_span_ext=M3_ext, M3_span_ext_x=M3_ext_x,
            M2_span_ext=M2_ext, M2_span_ext_x=M2_ext_x,
            V2_max_abs=float(max(abs(v) for v in V2_vals)),
            V3_max_abs=float(max(abs(v) for v in V3_vals)),
            q_local=[float(x) for x in q.tolist()],
        ))

    return out
