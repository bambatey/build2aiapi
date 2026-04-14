"""Analysis router HTTP entegrasyon testi.

Firebase'e dokunmadan, bağımlılıklar monkeypatch ile bellek içine
yönlendirilerek tüm endpoint'ler doğrulanır.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dependencies import get_uid
from repositories import analysis_repository, file_repository
from routers.analysis import router
from services import storage_service

FIXTURE = Path(__file__).parent / "fixtures" / "sap_dd2_iter3.s2k"
FAKE_UID = "test-user-1"
FAKE_PID = "proj-1"
FAKE_FID = "file-1"


@pytest.fixture
def client(monkeypatch) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    # Auth bypass
    app.dependency_overrides[get_uid] = lambda: FAKE_UID

    # --- in-memory fakes ---
    s2k_text = FIXTURE.read_text()

    async def fake_file_get(uid, project_id, file_id):
        assert uid == FAKE_UID
        if file_id != FAKE_FID:
            return None
        return {
            "id": FAKE_FID, "name": "sap_dd2_iter3.s2k",
            "storage_path": f"users/{uid}/projects/{project_id}/files/{file_id}_fake",
        }

    async def fake_download(storage_path: str) -> str:
        return s2k_text

    store: dict[str, dict] = {}

    async def fake_create(uid, project_id, file_id, options, summary, cases,
                          warnings, duration_ms, modes=None,
                          status="completed", error=None):
        import uuid
        aid = f"an_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()
        record = {
            "analysis_id": aid, "project_id": project_id, "file_id": file_id,
            "status": status, "options": options, "summary": summary,
            "cases": cases, "modes": modes or [],
            "warnings": warnings, "error": error,
            "created_at": now,
            "completed_at": now if status == "completed" else None,
            "duration_ms": duration_ms,
        }
        store[aid] = record
        return record

    async def fake_get(uid, project_id, file_id, analysis_id):
        return store.get(analysis_id)

    async def fake_list(uid, project_id, file_id):
        return sorted(store.values(), key=lambda r: r["created_at"], reverse=True)

    async def fake_delete(uid, project_id, file_id, analysis_id):
        store.pop(analysis_id, None)

    monkeypatch.setattr(file_repository, "get", fake_file_get)
    monkeypatch.setattr(storage_service, "download_file", fake_download)
    monkeypatch.setattr(analysis_repository, "create", fake_create)
    monkeypatch.setattr(analysis_repository, "get", fake_get)
    monkeypatch.setattr(analysis_repository, "list_by_file", fake_list)
    monkeypatch.setattr(analysis_repository, "delete", fake_delete)

    return TestClient(app)


def _analyze(client: TestClient) -> dict:
    r = client.post(
        f"/api/projects/{FAKE_PID}/files/{FAKE_FID}/analyze",
        json={"options": {}},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["success"] is True
    return body["data"]


def test_trigger_analysis_returns_summary(client):
    data = _analyze(client)
    assert data["status"] == "completed"
    assert data["duration_ms"] >= 0
    s = data["summary"]
    assert s["n_nodes"] == 80
    assert s["n_frame_elements"] == 153
    assert s["n_dofs_free"] == 360
    assert s["n_load_cases"] == 4


def test_trigger_analysis_404_when_file_missing(client):
    r = client.post(
        f"/api/projects/{FAKE_PID}/files/nonexistent/analyze",
        json={"options": {}},
    )
    assert r.status_code == 404


def test_list_analyses_returns_created(client):
    _analyze(client)
    r = client.get(f"/api/projects/{FAKE_PID}/files/{FAKE_FID}/analyses")
    assert r.status_code == 200
    items = r.json()["data"]
    assert len(items) == 1
    assert items[0]["status"] == "completed"


def test_get_status_and_summary(client):
    data = _analyze(client)
    aid = data["analysis_id"]

    r = client.get(f"/api/projects/{FAKE_PID}/files/{FAKE_FID}/analyses/{aid}")
    assert r.status_code == 200
    assert r.json()["data"]["analysis_id"] == aid

    r = client.get(
        f"/api/projects/{FAKE_PID}/files/{FAKE_FID}/analyses/{aid}/summary"
    )
    assert r.status_code == 200
    assert r.json()["data"]["n_nodes"] == 80


def test_get_displacements_filtered_by_load_case(client):
    data = _analyze(client)
    aid = data["analysis_id"]
    r = client.get(
        f"/api/projects/{FAKE_PID}/files/{FAKE_FID}/analyses/{aid}/displacements",
        params={"load_case": "EQX"},
    )
    assert r.status_code == 200
    disps = r.json()["data"]
    # 80 düğüm × 1 yük durumu = 80 kayıt
    assert len(disps) == 80
    assert all(d["load_case"] == "EQX" for d in disps)
    # En az bir düğümde yatay yer değiştirme olmalı
    assert any(abs(d["ux"]) > 0 for d in disps)


def test_get_reactions_all_cases(client):
    data = _analyze(client)
    aid = data["analysis_id"]
    r = client.get(
        f"/api/projects/{FAKE_PID}/files/{FAKE_FID}/analyses/{aid}/reactions"
    )
    assert r.status_code == 200
    reacts = r.json()["data"]
    # Fixture: 4 base case + 5 kombinasyon = 9 case; 20 mesnet × 9 = 180
    assert len(reacts) == 180
    # G base case (self-weight + gravity) → düşey reaksiyon toplamı pozitif
    g_fz = [r["fz"] for r in reacts if r["load_case"] == "G"]
    assert len(g_fz) == 20
    assert sum(g_fz) > 0


def test_delete_analysis(client):
    data = _analyze(client)
    aid = data["analysis_id"]

    r = client.delete(f"/api/projects/{FAKE_PID}/files/{FAKE_FID}/analyses/{aid}")
    assert r.status_code == 204

    r = client.get(f"/api/projects/{FAKE_PID}/files/{FAKE_FID}/analyses/{aid}")
    assert r.status_code == 404


def test_404_on_missing_analysis(client):
    r = client.get(
        f"/api/projects/{FAKE_PID}/files/{FAKE_FID}/analyses/an_missing"
    )
    assert r.status_code == 404
