"""Frame elemanı cache'i — geometri/materyal invariant tüm 12×12 matrisler
frame başına **tek** sefer hesaplanır.

Pipeline'da 90 kombinasyon × 200 frame × (K_local + T + K_local_released +
element_axes_transform) hesapları, her case için tekrar ediyordu
(docs/architecture/01-performance.md Bulgu 2). Bu modül bu işi
"preprocess once, read many" pattern'ine çevirir.

FrameKernel her frame için şunları tutar:
    - DTO referansları (element, node_i, node_j, section, material)
    - L (boy)
    - TE (3×3 element axes transform — global→local)
    - T, T_inv (12×12 — düğüm lokal Euler dahil, releases'ten bağımsız)
    - K_local (12×12 ham rijitlik)
    - K_local_released (12×12 releases static-condense edilmiş)
    - sw_w (öz ağırlık skaları: ρ × g × A, factor hariç)
    - released_idx (release'li lokal DOF indexleri — istatistik için)

build_frame_kernels(model) → dict[frame_id, FrameKernel].

Rust port yol haritası: FrameKernel.serialize() → (f64 flat arrays) Rust'a
geçer, aynı algoritma PyO3 üzerinden çalışır (docs/architecture/03-
rust-wasm-hybrid.md Tier 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.dto import (
    FrameElementDTO,
    FrameSectionDTO,
    MaterialDTO,
    ModelDTO,
    NodeDTO,
)
from .frame_3d import FrameElement3D


@dataclass(frozen=True)
class FrameKernel:
    """Bir frame elemanı için case-invariant pre-computed state."""

    element: FrameElementDTO
    node_i: NodeDTO
    node_j: NodeDTO
    section: FrameSectionDTO
    material: MaterialDTO

    length: float
    TE: np.ndarray               # (3, 3) element axes transform
    T: np.ndarray                # (12, 12) global → local (node Euler dahil)
    T_inv: np.ndarray            # (12, 12) T'nin tersi
    K_local: np.ndarray          # (12, 12) ham lokal rijitlik
    K_local_released: np.ndarray # (12, 12) releases static-condense edilmiş
    sw_w: float                  # ρ × g × A (öz ağırlık faktörü = 1 için w)
    released_idx: tuple[int, ...]

    @property
    def has_releases(self) -> bool:
        return bool(self.released_idx)


# Öz ağırlık ivmesi — load_assembler ile aynı sabit
_GRAVITY = 9.80665


def build_frame_kernels(model: ModelDTO) -> dict[int, FrameKernel]:
    """Tüm frame elemanları için FrameKernel cache'i oluştur.

    Geçersiz eleman (eksik malzeme/kesit/düğüm, L<=0) sessizce atlanır;
    caller ``kernels.get(el_id)`` kontrolü yapar. assemble_stiffness vb.
    mevcut "uyarı ver, atla" davranışını korur — yani bu fonksiyon
    warning üretmez (yukarıdakiler üretir).
    """
    out: dict[int, FrameKernel] = {}
    for el_id, el_dto in model.frame_elements.items():
        section = model.sections.get(el_dto.section_id)
        material = model.materials.get(el_dto.material_id)
        if not isinstance(section, FrameSectionDTO) or material is None:
            continue
        try:
            node_i = model.nodes[el_dto.nodes[0]]
            node_j = model.nodes[el_dto.nodes[1]]
        except (KeyError, IndexError):
            continue

        element = FrameElement3D(
            element=el_dto, node_i=node_i, node_j=node_j,
            section=section, material=material,
        )
        L = element.length
        if L <= 0:
            continue

        TE = element.element_axes_transform()
        T = element.local_to_global_transform()
        T_inv = np.linalg.inv(T)
        K_local = element.local_stiffness()
        K_local_released = element.local_stiffness_with_releases()
        sw_w = material.rho * _GRAVITY * section.A

        released: tuple[int, ...] = ()
        if el_dto.hinges:
            from .frame_3d import _release_indices  # lokal import — döngü olmasın
            released = tuple(sorted(set(_release_indices(el_dto.hinges))))

        out[el_id] = FrameKernel(
            element=el_dto,
            node_i=node_i,
            node_j=node_j,
            section=section,
            material=material,
            length=float(L),
            TE=TE,
            T=T,
            T_inv=T_inv,
            K_local=K_local,
            K_local_released=K_local_released,
            sw_w=float(sw_w),
            released_idx=released,
        )
    return out
