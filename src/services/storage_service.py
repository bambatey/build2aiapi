"""
Firebase Storage — .s2k ve diğer dosyaların saklanması.
Firestore'daki 1MB doküman limiti nedeniyle dosya içerikleri burada tutulur.
"""
import gzip
import json
import logging
from typing import Any

from services.firebase_service import firebase_service

logger = logging.getLogger(__name__)


class StorageService:

    def _get_path(self, uid: str, project_id: str, file_id: str, file_name: str) -> str:
        """Storage path: users/{uid}/projects/{pid}/files/{fid}_{name}"""
        return f"users/{uid}/projects/{project_id}/files/{file_id}_{file_name}"

    async def upload_file(
        self,
        uid: str,
        project_id: str,
        file_id: str,
        file_name: str,
        content: str,
    ) -> str:
        """
        Dosya içeriğini Firebase Storage'a yükle.
        Returns: storage path
        """
        path = self._get_path(uid, project_id, file_id, file_name)
        bucket = firebase_service.bucket
        blob = bucket.blob(path)
        blob.upload_from_string(content, content_type="text/plain")
        logger.info(f"Dosya yüklendi: {path}")
        return path

    async def download_file(self, storage_path: str) -> str:
        """Dosya içeriğini Firebase Storage'dan indir."""
        bucket = firebase_service.bucket
        blob = bucket.blob(storage_path)
        content = blob.download_as_text()
        return content

    async def delete_file(self, storage_path: str) -> None:
        """Dosyayı Firebase Storage'dan sil."""
        bucket = firebase_service.bucket
        blob = bucket.blob(storage_path)
        blob.delete()
        logger.info(f"Dosya silindi: {storage_path}")

    # ----------------------------------------- JSON (gzip) yardımcıları
    async def upload_json_gzip(self, path: str, data: Any) -> str:
        """``data``'yı JSON olarak encode et, gzip sıkıştır, Storage'a koy.

        Büyük analiz sonuçları (>1MB Firestore limiti aşanlar) için.
        Döndüğü path `storage_paths` field'ında Firestore'a yazılır.
        """
        bucket = firebase_service.bucket
        blob = bucket.blob(path)
        payload = gzip.compress(json.dumps(data).encode("utf-8"))
        blob.upload_from_string(payload, content_type="application/gzip")
        logger.info(f"JSON gzip yüklendi: {path} ({len(payload)} bytes)")
        return path

    async def download_json_gzip(self, path: str) -> Any:
        bucket = firebase_service.bucket
        blob = bucket.blob(path)
        payload = blob.download_as_bytes()
        return json.loads(gzip.decompress(payload).decode("utf-8"))


storage_service = StorageService()
