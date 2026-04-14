"""Gerçek SAP2000 .s2k fixture'ı üzerinde entegrasyon testleri.

Kaynak: `fixtures/sap_dd2_iter3.s2k` — 3 katlı 3 açıklıklı RC bina (kN-m-C).
Parser bu dosyayı güvenle ayrıştırabilmeli, sayımlar ve kritik değerler
SAP2000'in raporladığı ile örtüşmeli.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.structural_analysis.model.dto import (
    FrameElementDTO,
    FrameSectionDTO,
    ShellElementDTO,
    ShellSectionDTO,
)
from services.structural_analysis.model.enums import LoadType
from services.structural_analysis.parser import parse_s2k

FIXTURE = Path(__file__).parent / "fixtures" / "sap_dd2_iter3.s2k"


@pytest.fixture(scope="module")
def model():
    return parse_s2k(FIXTURE.read_text())


def test_units_parsed(model):
    assert model.units.force == "kN"
    assert model.units.length == "m"
    assert model.units.temperature == "C"


def test_materials_count_and_values(model):
    # c35, rebar, S355, Tendon
    assert set(model.materials) == {"c35", "rebar", "S355", "Tendon"}
    c35 = model.materials["c35"]
    # SAP: E1=33600000 kN/m² (33.6 GPa), U12=0.2
    assert c35.E == pytest.approx(33_600_000.0, rel=1e-6)
    assert c35.nu == pytest.approx(0.2, rel=1e-3)
    s355 = model.materials["S355"]
    assert s355.E == pytest.approx(210_000_000.0, rel=1e-4)
    assert s355.nu == pytest.approx(0.3, rel=1e-3)


def test_frame_sections_count_and_geometry(model):
    # 6 frame + 1 shell = 7 kesit
    frame_ids = {sid for sid, s in model.sections.items() if isinstance(s, FrameSectionDTO)}
    assert frame_ids == {"70*80", "80*90", "80*70", "90*80", "dis kiris", "ic kiris"}
    s70_80 = model.sections["70*80"]
    assert s70_80.A == pytest.approx(0.12, rel=1e-3)
    # SAP dosyasından: I33=0.02286..., I22=0.02986...
    assert s70_80.Iz == pytest.approx(0.02286666, rel=1e-4)
    assert s70_80.Iy == pytest.approx(0.02986666, rel=1e-4)


def test_shell_sections(model):
    shells = {sid for sid, s in model.sections.items() if isinstance(s, ShellSectionDTO)}
    assert shells == {"ASEC1"}
    assert model.sections["ASEC1"].thickness == pytest.approx(0.14, rel=1e-3)


def test_node_count_and_restraints(model):
    # 80 joint, 20'si tabanda (restraint tam tutulu)
    assert len(model.nodes) == 80
    restrained = [n for n in model.nodes.values() if all(n.restraints)]
    assert len(restrained) == 20
    # Joint 3: taban, tam ankastre
    assert model.nodes[3].restraints == [True] * 6
    # Joint 4: kat seviyesinde serbest
    assert model.nodes[4].restraints == [False] * 6


def test_frame_elements_and_assignments(model):
    assert len(model.frame_elements) == 153
    # Frame 2 → JointI=3, JointJ=4, Section=80*90
    f2 = model.frame_elements[2]
    assert f2.nodes == [3, 4]
    assert f2.section_id == "80*90"
    assert f2.material_id == "c35"


def test_shell_elements_and_assignments(model):
    assert len(model.shell_elements) == 36
    # Hepsi ASEC1 kesitinde, c35 malzemesinde olmalı (SAP Section=None olsa bile
    # tek tanımlı kabuk kesitine düşülür)
    for sh in model.shell_elements.values():
        assert sh.section_id == "ASEC1"
        assert sh.material_id == "c35"
        assert len(sh.nodes) == 4


def test_frame_and_shell_ids_can_overlap(model):
    # SAP: Frame=14 + Area=14 aynı dosyada yaşar; ayrı sözlüklerde korunmalı
    assert 14 in model.frame_elements
    assert 14 in model.shell_elements


def test_load_patterns(model):
    assert set(model.load_cases) == {"G", "Q", "EQX", "EQY"}
    assert model.load_cases["G"].type == LoadType.DEAD
    assert model.load_cases["G"].self_weight_factor == pytest.approx(1.0)
    assert model.load_cases["Q"].type == LoadType.LIVE
    assert model.load_cases["EQX"].type == LoadType.EARTHQUAKE_X
    assert model.load_cases["EQY"].type == LoadType.EARTHQUAKE_Y


def test_joint_loads(model):
    # 6 satır: 3 EQX + 3 EQY
    eqx = model.load_cases["EQX"].point_loads
    eqy = model.load_cases["EQY"].point_loads
    assert len(eqx) == 3
    assert len(eqy) == 3
    # Joint=7, EQX: F1=380.2
    j7_eqx = next(p for p in eqx if p.node_id == 7)
    assert j7_eqx.values[0] == pytest.approx(380.2, rel=1e-4)
    assert j7_eqx.values[1] == 0.0
    # Joint=8, EQY: F2=657.9
    j8_eqy = next(p for p in eqy if p.node_id == 8)
    assert j8_eqy.values[1] == pytest.approx(657.9, rel=1e-4)


def test_distributed_loads(model):
    # SAP dosyasında: 103 satır G (ölü yük), 30 satır Q (hareketli yük)
    g_loads = model.load_cases["G"].distributed_loads
    q_loads = model.load_cases["Q"].distributed_loads
    assert len(g_loads) == 103
    assert len(q_loads) == 30
    first = g_loads[0]
    assert first.direction == "gravity"
    assert first.coord_sys == "global"
    assert first.kind == "force"
    assert first.magnitude_a == pytest.approx(13.44, rel=1e-3)
    assert first.magnitude_b == pytest.approx(13.44, rel=1e-3)
    assert first.rel_dist_a == 0.0
    assert first.rel_dist_b == pytest.approx(1.0)


def test_combinations(model):
    combos = {c.id: c for c in model.combinations}
    assert set(combos) == {"COMB1", "COMB_EX+", "COMB_EX-", "COMB_EY+", "COMB_EY-"}
    comb1 = combos["COMB1"]
    assert comb1.factors == {"G": pytest.approx(1.4), "Q": pytest.approx(1.6)}
    ex_plus = combos["COMB_EX+"]
    assert ex_plus.factors["G"] == pytest.approx(1.0)
    assert ex_plus.factors["Q"] == pytest.approx(0.3)
    assert ex_plus.factors["EQX"] == pytest.approx(1.0)
    ex_minus = combos["COMB_EX-"]
    assert ex_minus.factors["EQX"] == pytest.approx(-1.0)
