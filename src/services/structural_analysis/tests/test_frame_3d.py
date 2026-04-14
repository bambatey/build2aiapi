"""FrameElement3D characterization testleri.

Portlanmış sınıfın (``elements/frame_3d.py``) lokal/global rijitlik ve yük
vektörleri, SEA_Book orijinal ``ElementFrame3D`` ile bire bir aynı olmalı.
Karşılaştırma: ``benchmarks/frame3d_golden.json`` — orijinal koddan üretilmiş
altın çıktı.

Altın çıktı değişirse (yeni test vakası eklenirse) scripti yeniden çalıştır:

    python src/services/structural_analysis/tests/benchmarks/generate_frame3d_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from services.structural_analysis.elements import FrameElement3D
from services.structural_analysis.model.dto import (
    FrameElementDTO,
    FrameSectionDTO,
    MaterialDTO,
    NodeDTO,
)
from services.structural_analysis.model.enums import ElementType

GOLDEN = Path(__file__).parent / "benchmarks" / "frame3d_golden.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


def _make_element(spec: dict, golden: dict) -> FrameElement3D:
    """Altın çıktıdaki bir eleman kaydını yeni API ile yeniden kur."""
    mat_data = golden["material"]
    sec_data = golden["section"]
    nodes_data = golden["nodes"]

    material = MaterialDTO(id=mat_data["id"], E=mat_data["E"], nu=mat_data["p"])
    section = FrameSectionDTO(
        id=sec_data["id"],
        A=sec_data["A"],
        Iy=sec_data["Iy"],
        Iz=sec_data["Iz"],
        J=sec_data["Ix"],        # SEA_Book'ta Ix = burulma → DTO'da J
        Iyz=sec_data["Iyz"],
    )
    ni_id, nj_id = spec["conn"]
    # NodeDTO id: int olmalı; string id'leri hash ile int'e çevirelim
    # (orijinal: "A", "B", "C")
    id_map = {nid: idx for idx, nid in enumerate(nodes_data.keys(), start=1)}

    def _make_node(nid: str) -> NodeDTO:
        n = nodes_data[nid]
        return NodeDTO(
            id=id_map[nid],
            x=n["X"],
            y=n["Y"],
            z=n["Z"],
            euler_zyx=tuple(n["EulerZYX"]),
        )

    el_dto = FrameElementDTO(
        id=spec["id"],
        type=ElementType.FRAME_3D,
        nodes=[id_map[ni_id], id_map[nj_id]],
        section_id=section.id,
        material_id=material.id,
        local_axis_angle=spec["omega_deg"],
    )
    return FrameElement3D(
        element=el_dto,
        node_i=_make_node(ni_id),
        node_j=_make_node(nj_id),
        section=section,
        material=material,
    )


@pytest.mark.parametrize("element_idx", [0, 1])
def test_length_and_direction_cosines(golden, element_idx):
    spec = golden["elements"][element_idx]
    el = _make_element(spec, golden)
    dx, dy, dz, L = el.direction_cosines_and_length()
    np.testing.assert_allclose([dx, dy, dz], spec["direction_cosines"], rtol=1e-12)
    assert L == pytest.approx(spec["L"], rel=1e-12)


@pytest.mark.parametrize("element_idx", [0, 1])
def test_local_to_global_transform_matches_golden(golden, element_idx):
    spec = golden["elements"][element_idx]
    el = _make_element(spec, golden)
    np.testing.assert_allclose(
        el.local_to_global_transform(), np.asarray(spec["TLG"]), rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("element_idx", [0, 1])
def test_local_stiffness_matches_golden(golden, element_idx):
    spec = golden["elements"][element_idx]
    el = _make_element(spec, golden)
    np.testing.assert_allclose(
        el.local_stiffness(), np.asarray(spec["K_Local"]), rtol=1e-12, atol=1e-9
    )


@pytest.mark.parametrize("element_idx", [0, 1])
def test_global_stiffness_matches_golden(golden, element_idx):
    spec = golden["elements"][element_idx]
    el = _make_element(spec, golden)
    np.testing.assert_allclose(
        el.global_stiffness(), np.asarray(spec["K"]), rtol=1e-10, atol=1e-8
    )


@pytest.mark.parametrize("element_idx", [0, 1])
def test_load_vectors_match_golden(golden, element_idx):
    spec = golden["elements"][element_idx]
    el = _make_element(spec, golden)
    q = spec["q"]
    np.testing.assert_allclose(
        el.local_load_vector(q), np.asarray(spec["q_Local"]), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        el.global_load_vector(q), np.asarray(spec["B"]), rtol=1e-10, atol=1e-10
    )


def test_stiffness_is_symmetric(golden):
    """Herhangi bir 3B frame'in lokal rijitlik matrisi simetrik olmalı
    (asimetrik kesit Iyz olsa bile)."""
    spec = golden["elements"][0]
    el = _make_element(spec, golden)
    K = el.local_stiffness()
    np.testing.assert_allclose(K, K.T, atol=1e-8)


def test_shear_modulus_formula():
    """G = E/(2(1+ν)) — malzemenin G alanı DTO'da yok, sınıf hesaplıyor."""
    mat = MaterialDTO(id="test", E=210e6, nu=0.3)
    sec = FrameSectionDTO(id="s", A=0.01, Iy=1e-5, Iz=1e-5, J=1e-6)
    n1 = NodeDTO(id=1, x=0, y=0, z=0)
    n2 = NodeDTO(id=2, x=1, y=0, z=0)
    el = FrameElement3D(
        element=FrameElementDTO(
            id=1,
            type=ElementType.FRAME_3D,
            nodes=[1, 2],
            section_id="s",
            material_id="test",
        ),
        node_i=n1,
        node_j=n2,
        section=sec,
        material=mat,
    )
    assert el.shear_modulus == pytest.approx(210e6 / (2 * 1.3))
