"""Firestore: users/{uid}/projects/{pid}/files/{fid}/analyses/{aid}

Küçük modellerde tüm analiz verisi tek dokümanda yaşar. Büyük
modellerde (Firestore 1MB doküman limitini aşan) ``cases`` ve
``modes`` bölümleri otomatik olarak Firebase Storage'a gzip JSON
olarak yazılır; Firestore dokümanında yalnızca özet + storage path'ler
kalır. Okuma sırasında şeffaf olarak Storage'dan geri inflate edilir.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from services.firebase_service import firebase_service
from services.storage_service import storage_service

logger = logging.getLogger(__name__)

# Firestore doküman hard limit: 1 MiB = 1_048_576 bytes.
# Güvenli payoff eşiği: 800 KB (metadata + marshalling payı).
_OFFLOAD_THRESHOLD_BYTES = 800_000


class AnalysisRepository:

    def _collection(self, uid: str, project_id: str, file_id: str):
        return (
            firebase_service.db
            .collection("users").document(uid)
            .collection("projects").document(project_id)
            .collection("files").document(file_id)
            .collection("analyses")
        )

    def _storage_path(self, uid: str, project_id: str, file_id: str,
                      analysis_id: str, kind: str) -> str:
        return (
            f"users/{uid}/projects/{project_id}/files/{file_id}"
            f"/analyses/{analysis_id}/{kind}.json.gz"
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
        modes: list[dict[str, Any]] | None = None,
        element_forces: dict[str, Any] | None = None,
        status: str = "completed",
        error: str | None = None,
    ) -> dict[str, Any]:
        analysis_id = f"an_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()
        storage_paths: dict[str, str] = {}

        # element_forces her zaman ayrı Storage blob'una yazılır — Firestore'a
        # asla inline girmez (kombinasyon başına ~MB'lara çıkabiliyor).
        if element_forces:
            path = self._storage_path(uid, project_id, file_id, analysis_id, "forces")
            await storage_service.upload_json_gzip(path, element_forces)
            storage_paths["forces"] = path

        # Büyüklüğü hesapla — cases ve modes genelde büyük
        def _size(x) -> int:
            return len(json.dumps(x, default=str).encode("utf-8"))

        cases_size = _size(cases) if cases else 0
        modes_size = _size(modes) if modes else 0
        total_estimate = cases_size + modes_size + _size(summary) + _size(warnings)

        if total_estimate > _OFFLOAD_THRESHOLD_BYTES:
            logger.info(
                "Analiz %s büyük (%.1f KB) — cases+modes Storage'a offload ediliyor.",
                analysis_id, total_estimate / 1024,
            )
            if cases:
                path = self._storage_path(uid, project_id, file_id, analysis_id, "cases")
                await storage_service.upload_json_gzip(path, cases)
                storage_paths["cases"] = path
                cases = {}   # Firestore'a boş yaz
            if modes:
                path = self._storage_path(uid, project_id, file_id, analysis_id, "modes")
                await storage_service.upload_json_gzip(path, modes)
                storage_paths["modes"] = path
                modes = []

        data = {
            "analysis_id": analysis_id,
            "project_id": project_id,
            "file_id": file_id,
            "status": status,
            "options": options,
            "summary": summary,
            "cases": cases,
            "modes": modes or [],
            "warnings": warnings,
            "error": error,
            "created_at": now,
            "completed_at": now if status == "completed" else None,
            "duration_ms": duration_ms,
            "storage_paths": storage_paths,
        }
        self._collection(uid, project_id, file_id).document(analysis_id).set(data)
        return data

    async def get_forces(
        self, uid: str, project_id: str, file_id: str, analysis_id: str,
    ) -> dict[str, Any]:
        """Kesit tesirleri blob'unu (varsa) Storage'dan getir.

        ``{case_id: [element_forces_row, ...]}`` sözlüğü döner. Yoksa {} döner.
        """
        doc = (
            self._collection(uid, project_id, file_id)
            .document(analysis_id).get()
        )
        if not doc.exists:
            return {}
        record = doc.to_dict() or {}
        paths = record.get("storage_paths") or {}
        forces_path = paths.get("forces")
        if not forces_path:
            return {}
        try:
            return await storage_service.download_json_gzip(forces_path)
        except Exception as exc:   # pragma: no cover
            logger.error("forces gzip indirilemedi: %s", exc)
            return {}

    async def get(
        self, uid: str, project_id: str, file_id: str, analysis_id: str,
        inflate: bool = True,
    ) -> dict[str, Any] | None:
        """Dokümanı oku. ``inflate=True`` ise Storage'daki cases/modes'u
        geri inflate eder; listeleme gibi yalnızca özet gerektiren yerler
        için ``inflate=False`` geçilebilir."""
        doc = (
            self._collection(uid, project_id, file_id)
            .document(analysis_id).get()
        )
        if not doc.exists:
            return None
        record = doc.to_dict()
        if inflate:
            paths = record.get("storage_paths") or {}
            if paths.get("cases") and not record.get("cases"):
                try:
                    record["cases"] = await storage_service.download_json_gzip(
                        paths["cases"]
                    )
                except Exception as exc:   # pragma: no cover
                    logger.error("cases gzip indirilemedi: %s", exc)
            if paths.get("modes") and not record.get("modes"):
                try:
                    record["modes"] = await storage_service.download_json_gzip(
                        paths["modes"]
                    )
                except Exception as exc:   # pragma: no cover
                    logger.error("modes gzip indirilemedi: %s", exc)
        return record

    async def list_by_file(
        self, uid: str, project_id: str, file_id: str
    ) -> list[dict[str, Any]]:
        """Sadece özet — cases/modes inflate edilmez (hızlı liste)."""
        docs = (
            self._collection(uid, project_id, file_id)
            .order_by("created_at", direction="DESCENDING")
            .stream()
        )
        return [d.to_dict() for d in docs]

    async def delete(
        self, uid: str, project_id: str, file_id: str, analysis_id: str
    ) -> None:
        # Önce Firestore dokümanını oku → storage path'leri bul → Storage'dan sil
        doc = (
            self._collection(uid, project_id, file_id)
            .document(analysis_id).get()
        )
        if doc.exists:
            record = doc.to_dict() or {}
            for path in (record.get("storage_paths") or {}).values():
                try:
                    await storage_service.delete_file(path)
                except Exception as exc:   # pragma: no cover
                    logger.warning("Storage sil başarısız: %s → %s", path, exc)
        self._collection(uid, project_id, file_id).document(analysis_id).delete()


analysis_repository = AnalysisRepository()
