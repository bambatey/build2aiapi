"""Pipeline ileri özellik testleri: yük durumu filtreleme, kombinasyonlar, modal."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.structural_analysis.parser import parse_s2k
from services.structural_analysis.pipeline import (
    AnalysisOptions,
    run_static_analysis,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sap_dd2_iter3.s2k"


@pytest.fixture(scope="module")
def model():
    return parse_s2k(FIXTURE.read_text())


# ---------------------------------------------------------------- filtering
def test_default_runs_all_cases_and_combinations(model):
    result = run_static_analysis(model)
    case_kinds = {c.case_id: c.kind for c in result.cases.values()}
    # 4 base case + 5 kombinasyon
    n_case = sum(1 for k in case_kinds.values() if k == "case")
    n_combo = sum(1 for k in case_kinds.values() if k == "combination")
    assert n_case == 4
    assert n_combo == 5


def test_selected_load_cases_filters(model):
    result = run_static_analysis(
        model, AnalysisOptions(selected_load_cases=["G"], selected_combinations=[])
    )
    # Sadece G case + G'yi referans eden kombinasyonlar yok (selected_combinations=[])
    # Ama kombinasyon için referans edilen case'ler yine çözülür — bu test'te yok
    assert "G" in result.cases
    # Q EQX EQY selected değil → çözülmedi
    assert "Q" not in result.cases
    # Kombinasyon yok
    assert result.summary["n_combinations"] == 0


def test_combination_auto_includes_base_cases(model):
    """Kombinasyon seçilirse referans ettiği base case otomatik çözülür."""
    # COMB1 = 1.4 G + 1.6 Q → G ve Q otomatik çözülmeli
    result = run_static_analysis(
        model,
        AnalysisOptions(
            selected_load_cases=[],          # hiç base case seçilmedi
            selected_combinations=["COMB1"],
        ),
    )
    assert "G" in result.cases
    assert "Q" in result.cases
    assert "COMB1" in result.cases
    # Summary'de base case = 0 (çünkü seçim boş), combination = 1
    assert result.summary["n_combinations"] == 1


def test_combination_linear_superposition(model):
    result = run_static_analysis(model)
    # COMB1 = 1.4 G + 1.6 Q
    g = result.cases["G"].raw.U
    q = result.cases["Q"].raw.U
    c1 = result.cases["COMB1"].raw.U
    import numpy as np
    expected = 1.4 * g + 1.6 * q
    np.testing.assert_allclose(c1, expected, rtol=1e-10, atol=1e-10)


# ------------------------------------------------------------------- modal
def test_modal_analysis_finds_periods(model):
    result = run_static_analysis(
        model,
        AnalysisOptions(
            selected_load_cases=[],
            selected_combinations=[],
            run_modal=True,
            modal_n_modes=6,
        ),
    )
    assert len(result.modes) > 0
    # Modlar temel frekanstan başlayarak sıralı → T1 > T2 > T3 (period azalan)
    periods = [m.period for m in result.modes]
    assert periods == sorted(periods, reverse=True)
    # 3 katlı RC bina: T1 tipik olarak 0.03-1.0 s arası.
    # Shell K dahil edildiği için döşemeler de rijitlik sağlar;
    # periyot frame-only'den daha kısa çıkar.
    T1 = result.modes[0].period
    assert 0.02 < T1 < 2.0, f"Beklenmedik T1={T1:.3f}s"
    # Summary alanları doldu mu?
    assert result.summary["n_modes"] == len(result.modes)
    assert result.summary["fundamental_period"] == pytest.approx(T1)


def test_modal_without_mass_skips_gracefully():
    """Malzeme rho=0 ise kütle matrisi sıfır → boş mode listesi."""
    from services.structural_analysis.model.dto import (
        FrameElementDTO, FrameSectionDTO, MaterialDTO, ModelDTO, NodeDTO,
    )
    from services.structural_analysis.model.enums import ElementType
    m = ModelDTO()
    # rho=0 → kütle sıfır
    m.materials["x"] = MaterialDTO(id="x", E=210e6, nu=0.3, rho=0.0)
    m.sections["s"] = FrameSectionDTO(id="s", A=0.01, Iy=1e-5, Iz=1e-5, J=1e-6)
    m.nodes[1] = NodeDTO(id=1, x=0, y=0, z=0, restraints=[True] * 6)
    m.nodes[2] = NodeDTO(id=2, x=1, y=0, z=0)
    m.frame_elements[1] = FrameElementDTO(
        id=1, type=ElementType.FRAME_3D, nodes=[1, 2],
        section_id="s", material_id="x",
    )
    result = run_static_analysis(
        m, AnalysisOptions(selected_load_cases=[], run_modal=True)
    )
    assert result.modes == []     # güvenle atlandı, crash yok
