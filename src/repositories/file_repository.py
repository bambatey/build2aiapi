"""
Firestore: users/{uid}/projects/{pid}/files/{file_id}
Dosya içeriği Firebase Storage'da, metadata burada.
"""
import uuid
from datetime import datetime

from services.firebase_service import firebase_service


class FileRepository:

    def _collection(self, uid: str, project_id: str):
        return (
            firebase_service.db
            .collection("users").document(uid)
            .collection("projects").document(project_id)
            .collection("files")
        )

    async def list_by_project(self, uid: str, project_id: str) -> list[dict]:
        docs = self._collection(uid, project_id).stream()
        files = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            files.append(data)
        return files

    async def get(self, uid: str, project_id: str, file_id: str) -> dict | None:
        doc = self._collection(uid, project_id).document(file_id).get()
        if doc.exists:
            data = doc.to_dict()
            data["id"] = doc.id
            return data
        return None

    async def create(
        self,
        uid: str,
        project_id: str,
        name: str,
        format: str,
        storage_path: str,
        size_bytes: int = 0,
        line_count: int = 0,
    ) -> dict:
        file_id = str(uuid.uuid4())
        now = datetime.utcnow()
        data = {
            "name": name,
            "type": "file",
            "path": f"/{project_id}/{name}",
            "format": format,
            "storage_path": storage_path,
            "size_bytes": size_bytes,
            "line_count": line_count,
            "created_at": now,
            "updated_at": now,
        }
        self._collection(uid, project_id).document(file_id).set(data)
        data["id"] = file_id
        return data

    async def update_storage(
        self,
        uid: str,
        project_id: str,
        file_id: str,
        storage_path: str,
        size_bytes: int,
        line_count: int,
    ) -> dict | None:
        self._collection(uid, project_id).document(file_id).update({
            "storage_path": storage_path,
            "size_bytes": size_bytes,
            "line_count": line_count,
            "updated_at": datetime.utcnow(),
        })
        return await self.get(uid, project_id, file_id)

    async def delete(self, uid: str, project_id: str, file_id: str) -> None:
        self._collection(uid, project_id).document(file_id).delete()


file_repository = FileRepository()
