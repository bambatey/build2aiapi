"""Design router HTTP endpoint testi.

Auth bypass + saf calculator → end-to-end serialize/validate kontrolü.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dependencies import get_uid
from routers.design import router


FAKE_UID = "test-user-1"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_uid] = lambda: FAKE_UID
    return TestClient(app)


# -------------------------------------------------------------- happy path
def test_beam_flexure_happy_path(client):
    """100 kNm C30/B500C → 3Ø16 + warnings boş."""
    r = client.post("/api/design/beam/flexure", json={
        "section": {"b_mm": 250, "h_mm": 500, "cover_mm": 25},
        "concrete": "C30/37",
        "steel": "B500C",
        "M_design_kNm": 100,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    data = body["data"]
    assert data["bars"]["label"] == "3Ø16"
    assert 500 <= data["A_s_required_mm2"] <= 560
    assert data["requires_double_reinforcement"] is False
    assert data["warnings"] == []
    assert data["concrete"] == "C30/37"
    assert data["steel"] == "B500C"


def test_beam_flexure_uses_defaults(client):
    """Concrete/steel verilmezse C30/37 + B500C default."""
    r = client.post("/api/design/beam/flexure", json={
        "section": {"b_mm": 250, "h_mm": 500},
        "M_design_kNm": 50,
    })
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["concrete"] == "C30/37"
    assert data["steel"] == "B500C"


def test_beam_flexure_rho_min_warning(client):
    """M = 0 → ρ_min uygulanır, uyarı listede görünür."""
    r = client.post("/api/design/beam/flexure", json={
        "section": {"b_mm": 250, "h_mm": 500},
        "M_design_kNm": 0,
    })
    assert r.status_code == 200
    data = r.json()["data"]
    assert any("minimum donatı" in w.lower() or "ρ_min" in w for w in data["warnings"])


def test_beam_flexure_invalid_section_400(client):
    """K_d ≥ 0.5 → 400 + Türkçe hata mesajı."""
    r = client.post("/api/design/beam/flexure", json={
        "section": {"b_mm": 200, "h_mm": 300, "cover_mm": 25},
        "M_design_kNm": 1000,  # ufak kesite çok büyük
    })
    assert r.status_code == 400
    assert "K_d" in r.json()["detail"]


def test_beam_flexure_invalid_geometry_422(client):
    """b_mm = 0 → pydantic validation 422."""
    r = client.post("/api/design/beam/flexure", json={
        "section": {"b_mm": 0, "h_mm": 500},
        "M_design_kNm": 100,
    })
    assert r.status_code == 422


def test_unknown_concrete_grade_422(client):
    r = client.post("/api/design/beam/flexure", json={
        "section": {"b_mm": 250, "h_mm": 500},
        "concrete": "C99/99",
        "M_design_kNm": 100,
    })
    assert r.status_code == 422
