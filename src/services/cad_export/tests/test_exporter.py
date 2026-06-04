"""End-to-end smoke testleri — fixture .s2k → DXF → readback.

Faz 1 hedefi: üretilen DXF ezdxf ile audit'ten geçsin (0 error),
multi-sheet (her kat ayrı), tüm beklenen layer'lar kullanılmış.
"""

from __future__ import annotations

from pathlib import Path

import ezdxf
import pytest

from services.cad_export import ProjectInfo, export_model_to_dxf
from services.cad_export.geometry import (
    build_story_geom,
    classify_frame,
    derive_grid,
    detect_stories,
)
from services.structural_analysis.parser import parse_s2k

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "structural_analysis" / "tests" / "fixtures" / "sap_dd2_iter3.s2k"
)


@pytest.fixture(scope="module")
def model():
    text = FIXTURE.read_text(encoding="utf-8", errors="ignore")
    return parse_s2k(text)


@pytest.fixture(scope="module")
def dxf_result(model):
    info = ProjectInfo(
        project_name="DD2-Iter3 Test Yapısı",
        city="İstanbul",
        district="Kadıköy",
        ada="123",
        parsel="45",
        engineer_name="Test Mühendis",
    )
    return export_model_to_dxf(model, info)


def test_parser_extracts_section_dims(model):
    """Parser FRAME SECTION PROPERTIES 01 → t2, t3, shape, ConcCol/Beam."""
    sec_70x80 = model.sections.get("70*80")
    assert sec_70x80 is not None
    assert sec_70x80.shape == "Rectangular"
    assert sec_70x80.t2 == pytest.approx(0.8)
    assert sec_70x80.t3 == pytest.approx(0.7)
    assert sec_70x80.is_column is True


def test_story_detection(model):
    stories = detect_stories(model)
    assert len(stories) >= 2          # Multi-kat
    assert stories[0].label == "Zemin Kat"


def test_frame_classification(model):
    """ConcCol flag yanıltıyor olsa bile geometri doğru sınıflandırıyor."""
    classes = [classify_frame(model, fid) for fid in model.frame_elements]
    assert "column" in classes
    assert "beam" in classes


def test_grid_derivation(model):
    """Kolon XY'lerinden aks gridi türetilebiliyor (A-E × 1-4 beklenir)."""
    stories = detect_stories(model)
    geom = build_story_geom(model, stories[0])
    assert len(geom.grid.x_positions) >= 3
    assert len(geom.grid.y_positions) >= 3
    assert geom.grid.x_labels[0] == "A"
    assert geom.grid.y_labels[0] == "1"


def test_dxf_audit_clean(dxf_result, tmp_path):
    """ezdxf audit: 0 error, 0 fix — AutoCAD'in açabileceği temiz DXF."""
    out = tmp_path / "kalip.dxf"
    out.write_bytes(dxf_result.dxf_bytes)
    doc = ezdxf.readfile(out)
    auditor = doc.audit()
    assert len(auditor.errors) == 0, [str(e) for e in auditor.errors[:5]]
    assert len(auditor.fixes) == 0, [str(f) for f in auditor.fixes[:5]]


def test_dxf_has_expected_layers(dxf_result, tmp_path):
    out = tmp_path / "kalip.dxf"
    out.write_bytes(dxf_result.dxf_bytes)
    doc = ezdxf.readfile(out)
    used_layers = {e.dxf.layer for e in doc.modelspace()
                   if e.dxf.hasattr("layer")}
    expected = {"GRID", "COLUMN", "BEAM", "SLAB", "DIMENSION",
                "TITLE-BLOCK", "TABLE", "TEXT"}
    missing = expected - used_layers
    assert not missing, f"Beklenen layer'lar boş: {missing}"


def test_dxf_multi_sheet(dxf_result, model):
    """Detected story sayısı kadar sayfa var."""
    stories = detect_stories(model)
    assert dxf_result.sheet_count == len(stories)
    assert dxf_result.sheet_count >= 2


def test_dxf_has_grid_bubbles(dxf_result, tmp_path):
    """Aks bubble bloğu en az kolon * kat sayısı kadar insert edilmiş."""
    out = tmp_path / "kalip.dxf"
    out.write_bytes(dxf_result.dxf_bytes)
    doc = ezdxf.readfile(out)
    inserts = [
        e for e in doc.modelspace()
        if e.dxftype() == "INSERT" and e.dxf.name == "GRID_BUBBLE"
    ]
    # Her kat için (x_aks + y_aks) × 2 (üst+alt / sol+sağ) ≈ 18 bubble bekleriz.
    assert len(inserts) >= 18


def test_empty_model_does_not_crash():
    """Boş ModelDTO ile bile (degenerate case) çıkış valid bir DXF üretmeli."""
    from services.structural_analysis.model.dto import ModelDTO
    result = export_model_to_dxf(ModelDTO())
    assert result.dxf_bytes.startswith(b"  0\nSECTION")  # DXF magic
    assert result.sheet_count >= 1
