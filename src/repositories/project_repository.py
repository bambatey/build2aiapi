"""
Firestore: users/{uid}/projects/{project_id}
"""
import uuid
from datetime import datetime

from services.firebase_service import firebase_service


class ProjectRepository:

    def _collection(self, uid: str):
        return firebase_service.db.collection("users").document(uid).collection("projects")

    async def list_by_user(self, uid: str) -> list[dict]:
        docs = self._collection(uid).order_by("updated_at", direction="DESCENDING").stream()
        projects = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            projects.append(data)
        return projects

    async def get(self, uid: str, project_id: str) -> dict | None:
        doc = self._collection(uid).document(project_id).get()
        if doc.exists:
            data = doc.to_dict()
            data["id"] = doc.id
            return data
        return None

    async def create(self, uid: str, name: str, format: str = ".s2k", tags: list[str] | None = None) -> dict:
        project_id = str(uuid.uuid4())
        now = datetime.utcnow()
        data = {
            "name": name,
            "format": format,
            "tags": tags or [],
            "progress": 0,
            "file_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        self._collection(uid).document(project_id).set(data)
        data["id"] = project_id
        return data

    async def update(self, uid: str, project_id: str, updates: dict) -> dict | None:
        updates["updated_at"] = datetime.utcnow()
        self._collection(uid).document(project_id).update(updates)
        return await self.get(uid, project_id)

    async def delete(self, uid: str, project_id: str) -> None:
        self._collection(uid).document(project_id).delete()

    async def increment_file_count(self, uid: str, project_id: str, delta: int = 1) -> None:
        from google.cloud.firestore_v1 import Increment
        self._collection(uid).document(project_id).update({
            "file_count": Increment(delta),
            "updated_at": datetime.utcnow(),
        })


project_repository = ProjectRepository()
