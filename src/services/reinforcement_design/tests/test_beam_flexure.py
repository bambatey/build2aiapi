"""Kiriş eğilme tasarımı testleri — TS 500 §7.4.

Beklenen değerler iki kaynaktan:
1. Kanonik kitap örneği (Celal Bayar Üniversitesi ders notu, C30/B500C
   örneği) — geleneksel TR betonarme proje pratiği
2. Manuel el hesabı — denklem zinciri belgelenmiş (beam_flexure.py docstring)
"""

from __future__ import annotations

import math

import pytest

from services.reinforcement_design import (
    BeamFlexureInput,
    ConcreteGrade,
    MaterialProperties,
    RectangularSection,
    SteelGrade,
    design_beam_flexure,
)
from services.reinforcement_design.beam_flexure import _beta_1
from services.reinforcement_design.bar_sizes import bar_area_mm2, select_bars


# --------------------------------------------------------------------- malzeme
def test_fcd_concrete_C30():
    m = MaterialProperties(ConcreteGrade.C30_37, SteelGrade.B500C)
    assert m.f_ck == pytest.approx(30.0)
    assert m.f_cd == pytest.approx(20.0, rel=1e-3)        # 30 / 1.5


def test_fyd_steel_B500C():
    m = MaterialProperties(ConcreteGrade.C30_37, SteelGrade.B500C)
    assert m.f_yk == pytest.approx(500.0)
    assert m.f_yd == pytest.approx(434.78, rel=1e-3)      # 500 / 1.15


def test_fctd_C30():
    m = MaterialProperties(ConcreteGrade.C30_37, SteelGrade.B500C)
    # f_ctk = 0.35 × √30 = 1.917
    # f_ctd = 1.917 / 1.5 ≈ 1.278
    assert m.f_ctd == pytest.approx(1.278, rel=1e-2)


def test_beta_1_breakpoints():
    assert _beta_1(20.0) == pytest.approx(0.85)
    assert _beta_1(28.0) == pytest.approx(0.85)
    assert _beta_1(40.0) == pytest.approx(0.85 - 0.0075 * 12)
    assert _beta_1(60.0) == pytest.approx(0.65)


# ------------------------------------------------------------------- bar seçimi
def test_select_bars_minimum_excess():
    # 528 mm² gereksinim → 3Ø16 (603 mm²) seçilir (4Ø14 = 615 mm²'den daha az fazla)
    sel = select_bars(528.0, min_diameter_mm=12, max_diameter_mm=25)
    assert sel is not None
    assert sel.count == 3
    assert sel.diameter_mm == 16
    assert sel.area_mm2 == pytest.approx(3 * bar_area_mm2(16))


def test_select_bars_zero_returns_min():
    sel = select_bars(0.0, min_count=2, min_diameter_mm=12, max_diameter_mm=12)
    assert sel is not None
    assert sel.count == 2
    assert sel.diameter_mm == 12


# --------------------------------------------------------- eğilme tasarımı
def _make_input(M_kNm: float, **kwargs):
    sec = RectangularSection(b_mm=250.0, h_mm=500.0, cover_mm=25.0)
    mat = MaterialProperties(ConcreteGrade.C30_37, SteelGrade.B500C)
    return BeamFlexureInput(section=sec, materials=mat, M_design_kNm=M_kNm, **kwargs)


def test_kanonik_100kNm_C30_B500C():
    """Kanonik el hesabı:
    d = 500 − 25 − 8 − 8 = 459 mm
    K_d = 100e6 / (250 × 459² × 20) ≈ 0.0949
    ω = 1 − √(1 − 2 × 0.0949) ≈ 0.0999
    ρ ≈ 0.0046, A_s ≈ 528 mm² → 3Ø16
    """
    result = design_beam_flexure(_make_input(100.0))
    assert result.K_d == pytest.approx(0.0949, rel=2e-2)
    assert result.omega == pytest.approx(0.0999, rel=2e-2)
    assert 500.0 <= result.A_s_required_mm2 <= 560.0
    assert result.bars is not None
    assert result.bars.selection.label == "3Ø16"
    assert result.requires_double_reinforcement is False
    assert result.warnings == []


def test_zero_moment_uses_rho_min():
    result = design_beam_flexure(_make_input(0.0))
    # ρ_min = 0.8 × 1.278 / 434.78 ≈ 0.00235
    assert result.rho_min == pytest.approx(0.00235, rel=2e-2)
    assert result.A_s_required_mm2 == pytest.approx(result.A_s_min_mm2, rel=1e-2)
    assert result.warnings  # ρ < ρ_min uyarısı var
    assert result.bars is not None
    # Min donatı: 2 × Ø12 = 226 mm² (yeterli), veya 2Ø14 ≈ 308 → 269 gereksinim üstünde
    assert result.bars.A_s_provided_mm2 >= result.A_s_min_mm2


def test_very_large_moment_flags_double_reinforcement():
    """M sünek sınırı aşar ama K_d < 0.5 — çift donatı flag.

    Kesite göre K_d,max ≈ 0.29, M_max ≈ 305 kNm.
    M = 350 kNm: K_d ≈ 0.33 > K_d,max → requires_double_reinforcement.
    """
    result = design_beam_flexure(_make_input(350.0))
    assert result.requires_double_reinforcement is True
    assert any("çift donatılı" in w.lower() for w in result.warnings)


def test_extreme_moment_raises():
    """M kesite fiziksel olarak sığmaz → DesignError (K_d ≥ 0.5)."""
    from services.reinforcement_design.types import DesignError
    with pytest.raises(DesignError, match="K_d"):
        design_beam_flexure(_make_input(1500.0))


def test_invalid_section_raises():
    sec = RectangularSection(b_mm=250.0, h_mm=40.0, cover_mm=25.0,
                              stirrup_diameter_mm=8.0,
                              longitudinal_diameter_mm=16.0)
    mat = MaterialProperties(ConcreteGrade.C30_37, SteelGrade.B500C)
    # d = 40 - 25 - 8 - 8 = -1 → DesignError
    from services.reinforcement_design.types import DesignError
    with pytest.raises(DesignError, match="Yararlı yükseklik"):
        design_beam_flexure(BeamFlexureInput(section=sec, materials=mat, M_design_kNm=10.0))


def test_negative_moment_uses_abs():
    """Pozitif ve negatif M aynı sonucu vermeli (mutlak değer)."""
    pos = design_beam_flexure(_make_input(80.0))
    neg = design_beam_flexure(_make_input(-80.0))
    assert pos.A_s_required_mm2 == pytest.approx(neg.A_s_required_mm2)


def test_different_concrete_class_changes_result():
    """C20 daha zayıf → aynı M için daha fazla donatı."""
    sec = RectangularSection(b_mm=250.0, h_mm=500.0, cover_mm=25.0)
    M = 100.0
    res_c20 = design_beam_flexure(BeamFlexureInput(
        section=sec,
        materials=MaterialProperties(ConcreteGrade.C20_25, SteelGrade.B500C),
        M_design_kNm=M,
    ))
    res_c40 = design_beam_flexure(BeamFlexureInput(
        section=sec,
        materials=MaterialProperties(ConcreteGrade.C40_50, SteelGrade.B500C),
        M_design_kNm=M,
    ))
    assert res_c20.A_s_required_mm2 > res_c40.A_s_required_mm2
