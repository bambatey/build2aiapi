"""Firestore: users/{uid}/projects/{pid}/files/{fid}/analyses/{aid}

Analiz meta + sonuç tek dokümanda tutulur (MVP). >1MB olacak büyük
modeller için Storage'a gzip JSON ayırma desteği sonraki iterasyonda.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from services.firebase_service import firebase_service


class AnalysisRepository:

    def _collection(self, uid: str, project_id: str, file_id: str):
        return (
            firebase_service.db
            .collection("users").document(uid)
            .collection("projects").document(project_id)
            .collection("files").document(file_id)
            .collection("analyses")
        )

    async def create(
        self,
        uid: str,
        project_id: str,
        file_id: str,
        options: dict[str, Any],
        summary: dict[str, Any],
        cases: dict[str, Any],
        warnings: list[str],
        duration_ms: int,
        status: str = "completed",
        error: str | None = None,
    ) -> dict[str, Any]:
        analysis_id = f"an_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()
        data = {
            "analysis_id": analysis_id,
            "project_id": project_id,
            "file_id": file_id,
            "status": status,
            "options": options,
            "summary": summary,
            "cases": cases,       # {case_id: {displacements, reactions, summary}}
            "warnings": warnings,
            "error": error,
            "created_at": now,
            "completed_at": now if status == "completed" else None,
            "duration_ms": duration_ms,
        }
        self._collection(uid, project_id, file_id).document(analysis_id).set(data)
        return data

    async def get(
        self, uid: str, project_id: str, file_id: str, analysis_id: str
    ) -> dict[str, Any] | None:
        doc = (
            self._collection(uid, project_id, file_id)
            .document(analysis_id).get()
        )
        if not doc.exists:
            return None
        return doc.to_dict()

    async def list_by_file(
        self, uid: str, project_id: str, file_id: str
    ) -> list[dict[str, Any]]:
        docs = (
            self._collection(uid, project_id, file_id)
            .order_by("created_at", direction="DESCENDING")
            .stream()
        )
        return [d.to_dict() for d in docs]

    async def delete(
        self, uid: str, project_id: str, file_id: str, analysis_id: str
    ) -> None:
        self._collection(uid, project_id, file_id).document(analysis_id).delete()


analysis_repository = AnalysisRepository()
