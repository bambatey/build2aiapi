"""TBDY 2018 spektrum ve response spectrum analiz testleri."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.structural_analysis.parser import parse_s2k
from services.structural_analysis.pipeline import (
    AnalysisOptions,
    SpectrumOptions,
    run_static_analysis,
)
from services.structural_analysis.spectra import (
    TBDY2018Spectrum,
    soil_coefficients,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sap_dd2_iter3.s2k"


# -------------------------------------------------------- spektrum eğrisi
def test_soil_coefficients_zc_typical():
    # TBDY 2018 Tablo 2.1: ZC, Ss=0.5 → FS=1.3
    FS, F1 = soil_coefficients("ZC", 0.5, 0.2)
    assert FS == pytest.approx(1.3)
    assert F1 == pytest.approx(1.5)


def test_soil_coefficients_zd_interpolates():
    # ZD, Ss=0.375 (tablo 0.25'te 1.6, 0.50'de 1.4): ortalama ≈ 1.5
    FS, _ = soil_coefficients("ZD", 0.375, 0.1)
    assert FS == pytest.approx(1.5, rel=1e-6)


def test_spectrum_curve_plateau():
    """TA < T < TB aralığında Sae = SDS."""
    s = TBDY2018Spectrum(Ss=1.0, S1=0.3, soil="ZC", R=4.0, I=1.0)
    # SDS = 1.0 × 1.2 = 1.2; SD1 = 0.3 × 1.5 = 0.45
    # TA = 0.2 × 0.45 / 1.2 = 0.075 s; TB = 0.375 s
    assert s.SDS == pytest.approx(1.2)
    assert s.SD1 == pytest.approx(0.45)
    assert s.TA == pytest.approx(0.075)
    assert s.TB == pytest.approx(0.375)
    # Plato
    assert s.Sa_elastic(0.2) == pytest.approx(1.2)
    # Azalan bölge (TB < T ≤ TL)
    assert s.Sa_elastic(1.0) == pytest.approx(0.45)
    assert s.Sa_elastic(2.0) == pytest.approx(0.225)


def test_spectrum_design_reduction():
    """Sa_design = Sa_elastic × I / Ra. T ≥ TB için Ra = R/I."""
    s = TBDY2018Spectrum(Ss=1.0, S1=0.3, soil="ZC", R=4.0, I=1.0)
    # T = 0.5 s (plato sonrası): Ra = 4
    assert s.Sa_design(0.5) == pytest.approx(s.Sa_elastic(0.5) / 4.0, rel=1e-9)


# ----------------------------------------------------- pipeline entegrasyon
@pytest.fixture(scope="module")
def model():
    return parse_s2k(FIXTURE.read_text())


def test_response_spectrum_produces_rs_cases(model):
    result = run_static_analysis(
        model,
        AnalysisOptions(
            selected_load_cases=[],
            selected_combinations=[],
            run_modal=True,
            modal_n_modes=9,
            run_response_spectrum=True,
            spectrum=SpectrumOptions(Ss=1.0, S1=0.3, soil="ZC", R=4.0),
        ),
    )
    # EQX_RS + EQY_RS case'leri oluşmalı
    assert "EQX_RS" in result.cases
    assert "EQY_RS" in result.cases
    assert result.cases["EQX_RS"].kind == "response_spectrum"
    assert result.summary["n_response_spectrum"] == 2
    # RS her zaman pozitif (SRSS) → max displacement pozitif
    assert result.summary["max_displacement"] > 0


def test_rs_direction_filter(model):
    # Sadece X çalışsın
    result = run_static_analysis(
        model,
        AnalysisOptions(
            selected_load_cases=[],
            selected_combinations=[],
            run_modal=True,
            run_response_spectrum=True,
            spectrum=SpectrumOptions(run_x=True, run_y=False),
        ),
    )
    assert "EQX_RS" in result.cases
    assert "EQY_RS" not in result.cases
