"""Pipeline end-to-end testleri.

İki kapsam:

1. **Kontrollü test vakası** (SEA_Book 3-düğümlü 2-eleman): pipeline
   çıktısı altın US/PS (``benchmarks/pipeline_golden.json``) ile
   eşleşmeli. Bu, DOF numaralama + K birleştirme + solve + recovery
   zincirinin SEA_Book referansıyla uyumlu olduğunu kanıtlar.

2. **Gerçek SAP fixture**: parse → pipeline ucu uca çalışmalı, temel
   mantıksal sağlama testlerinden geçmeli (DOF sayısı, denge, mesnet
   reaksiyonları).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from services.structural_analysis.model.dto import (
    DistributedLoadDTO,
    FrameElementDTO,
    FrameSectionDTO,
    LoadCaseDTO,
    MaterialDTO,
    ModelDTO,
    NodeDTO,
    PointLoadDTO,
)
from services.structural_analysis.model.enums import ElementType, LoadType
from services.structural_analysis.parser import parse_s2k
from services.structural_analysis.pipeline import (
    run_from_s2k,
    run_static_analysis,
)

GOLDEN = Path(__file__).parent / "benchmarks" / "pipeline_golden.json"
FIXTURE = Path(__file__).parent / "fixtures" / "sap_dd2_iter3.s2k"


# -------------------------------------------------------- SEA_Book test vakası
def _build_sea_book_case() -> ModelDTO:
    """sec5_frame_3d.py'deki 3-düğüm 2-eleman örneği ModelDTO olarak."""
    model = ModelDTO()
    model.materials["steel"] = MaterialDTO(id="steel", E=210e6, nu=0.3)
    model.sections["L"] = FrameSectionDTO(
        id="L",
        A=0.002364,
        Iy=0.00000458124835,
        Iz=0.00001597484835,
        J=0.00000002805048,
        Iyz=0.00000501624365,
    )
    # Node id'leri int (SEA_Book string "A","B","C" yerine): A=1, B=2, C=3
    model.nodes[1] = NodeDTO(id=1, x=-3, y=-3, z=0, restraints=[True] * 6)
    model.nodes[2] = NodeDTO(id=2, x=3, y=3, z=0, restraints=[True] * 6)
    model.nodes[3] = NodeDTO(id=3, x=3, y=-3, z=3, euler_zyx=(30.0, 20.0, 10.0))

    model.frame_elements[1] = FrameElementDTO(
        id=1, type=ElementType.FRAME_3D, nodes=[1, 3],
        section_id="L", material_id="steel", local_axis_angle=20.0,
    )
    model.frame_elements[2] = FrameElementDTO(
        id=2, type=ElementType.FRAME_3D, nodes=[2, 3],
        section_id="L", material_id="steel",
    )

    # Load: C düğümünde -10 kN Z yönünde
    model.load_cases["POINT"] = LoadCaseDTO(
        id="POINT", type=LoadType.OTHER,
        point_loads=[PointLoadDTO(node_id=3, values=[0, 0, -10, 0, 0, 0])],
    )
    return model


def test_sea_book_case_matches_golden():
    golden = json.loads(GOLDEN.read_text())
    model = _build_sea_book_case()
    result = run_static_analysis(model)

    sol = result.cases["POINT"].raw
    assert sol.U.size == golden["M_total"] == 18
    # 3 düğüm × 6 DOF = 18, C serbest (6) + A,B tutulu (12) → N_free=6
    assert result.summary["n_dofs_free"] == golden["N_free"] == 6

    np.testing.assert_allclose(sol.U, golden["US"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sol.P, golden["PS"], rtol=1e-5, atol=1e-5)


def test_sea_book_recovery_dicts():
    model = _build_sea_book_case()
    result = run_static_analysis(model)
    case = result.cases["POINT"]
    # C düğümü (serbest) yer değiştirmesi ~ golden[0:6]
    golden = json.loads(GOLDEN.read_text())
    c_disp = case.displacements[3]
    np.testing.assert_allclose(
        [c_disp["ux"], c_disp["uy"], c_disp["uz"], c_disp["rx"], c_disp["ry"], c_disp["rz"]],
        golden["US"][0:6], rtol=1e-6, atol=1e-8,
    )
    # A ve B düğümleri reaksiyonlu olmalı
    assert 1 in case.reactions
    assert 2 in case.reactions
    assert 3 not in case.reactions   # C serbest, reaksiyon yok


def test_local_distributed_load_on_element_end_up_as_equivalent_nodal_forces():
    """Lokal bir yayılı yük elemanın uçlarına eş-nodal olarak dağıtılmalı.

    Yöntem: basit bir kiriş ele al, her iki ucu tutulu; ortada olmasa da
    lokal_2 yönünde q yayılı yük uygulandığında uç reaksiyonlarının yarı
    yarıya (qL/2) olması gerekir.
    """
    model = ModelDTO()
    model.materials["steel"] = MaterialDTO(id="steel", E=210e6, nu=0.3)
    model.sections["S"] = FrameSectionDTO(
        id="S", A=0.01, Iy=8.33e-6, Iz=8.33e-6, J=1.4e-5,
    )
    L = 4.0
    q = 10.0  # kN/m
    model.nodes[1] = NodeDTO(id=1, x=0, y=0, z=0, restraints=[True] * 6)
    model.nodes[2] = NodeDTO(id=2, x=L, y=0, z=0, restraints=[True] * 6)
    model.frame_elements[1] = FrameElementDTO(
        id=1, type=ElementType.FRAME_3D, nodes=[1, 2],
        section_id="S", material_id="steel",
    )
    model.load_cases["DIST"] = LoadCaseDTO(
        id="DIST", type=LoadType.OTHER,
        distributed_loads=[
            DistributedLoadDTO(
                element_id=1, coord_sys="local", direction="local_2",
                magnitude_a=q, magnitude_b=q,
            )
        ],
    )
    result = run_static_analysis(model)
    r1 = result.cases["DIST"].reactions[1]
    r2 = result.cases["DIST"].reactions[2]
    # Yatay kiriş (X ekseninde) için lokal 2 → global Z (TALFA'dan).
    # Tümü tutulu olduğundan reaksiyon toplamı q·L'yi dengelemeli.
    total_fz = r1["fz"] + r2["fz"]
    assert abs(total_fz) == pytest.approx(q * L, rel=1e-6)
    # Simetrik yükleme → iki uçta eşit
    assert r1["fz"] == pytest.approx(r2["fz"], rel=1e-6)


# ---------------------------------------------------------- gerçek SAP fixture
def test_real_sap_fixture_pipeline_runs():
    """MVP smoke test — gerçek 3 katlı bina modelini uçtan uca çöz."""
    result = run_from_s2k(FIXTURE.read_text())
    s = result.summary
    assert s["n_nodes"] == 80
    assert s["n_frame_elements"] == 153
    # 80 düğüm × 6 = 480 DOF, 20 düğüm tam tutulu → 120 tutulu, 360 serbest
    assert s["n_dofs_total"] == 480
    assert s["n_dofs_free"] == 360
    assert s["n_load_cases"] == 4
    # En azından bir yük durumunda anlamlı yer değiştirme olmalı (tümü 0 değil)
    assert s["max_displacement"] > 0.0


def test_real_sap_fixture_gravity_reactions_balance():
    """G yük durumunda toplam mesnet reaksiyonu = uygulanan toplam düşey yük.

    Uygulanan yük = öz ağırlık (SelfWtMult=1) + Dir=Gravity yayılı yükler
    + (varsa) noktasal yükler.
    """
    model = parse_s2k(FIXTURE.read_text())
    result = run_static_analysis(model)
    g_case = result.cases["G"]

    # Uygulanan toplam düşey yük (kN, +Z = yukarı; gravity yükü -Z yönünde)
    total_applied_z = 0.0
    # Yayılı yükler: dir=gravity → -Z, dir=z → +Z, diğerleri Z'ye katkı yok
    for dl in model.load_cases["G"].distributed_loads:
        el = model.frame_elements.get(dl.element_id)
        if el is None:
            continue
        ni, nj = el.nodes
        n1, n2 = model.nodes[ni], model.nodes[nj]
        L = ((n2.x - n1.x) ** 2 + (n2.y - n1.y) ** 2 + (n2.z - n1.z) ** 2) ** 0.5
        if dl.direction == "gravity":
            total_applied_z -= dl.magnitude_a * L
        elif dl.direction == "z":
            total_applied_z += dl.magnitude_a * L

    # Öz ağırlık: ρ × g × A × L (SelfWtMult=1)
    from services.structural_analysis.assembly.load_assembler import GRAVITY
    for el in model.frame_elements.values():
        sec = model.sections.get(el.section_id)
        mat = model.materials.get(el.material_id)
        if sec is None or mat is None or not hasattr(sec, "A"):
            continue
        n1, n2 = model.nodes[el.nodes[0]], model.nodes[el.nodes[1]]
        L = ((n2.x - n1.x) ** 2 + (n2.y - n1.y) ** 2 + (n2.z - n1.z) ** 2) ** 0.5
        total_applied_z -= mat.rho * GRAVITY * sec.A * L

    total_reaction_z = sum(r["fz"] for r in g_case.reactions.values())
    # Reaksiyon = -uygulanan yük (denge); toleransı %0.5
    assert total_reaction_z == pytest.approx(-total_applied_z, rel=5e-3)
    assert total_reaction_z > 0  # mesnetler bina ağırlığını yukarı taşır


def test_real_sap_fixture_earthquake_produces_displacement():
    """EQX yük durumunda yatay yer değiştirme pozitif olmalı."""
    model = parse_s2k(FIXTURE.read_text())
    result = run_static_analysis(model)
    disp = result.cases["EQX"].displacements
    max_ux = max(d["ux"] for d in disp.values())
    assert max_ux > 0.0
