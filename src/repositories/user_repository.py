"""
Firestore: users/{firebase_uid}
"""
from datetime import datetime
from typing import Any

from google.cloud.firestore_v1 import FieldFilter

from services.firebase_service import firebase_service


class UserRepository:

    @property
    def _collection(self):
        return firebase_service.db.collection("users")

    async def get_by_uid(self, uid: str) -> dict | None:
        doc = self._collection.document(uid).get()
        if doc.exists:
            data = doc.to_dict()
            data["uid"] = doc.id
            return data
        return None

    async def create(self, uid: str, email: str | None, display_name: str | None, photo_url: str | None = None) -> dict:
        now = datetime.utcnow()
        data = {
            "email": email,
            "display_name": display_name,
            "photo_url": photo_url,
            "settings": {},
            "created_at": now,
            "updated_at": now,
        }
        self._collection.document(uid).set(data)
        data["uid"] = uid
        return data

    async def get_or_create(self, uid: str, email: str | None, display_name: str | None, photo_url: str | None = None) -> dict:
        user = await self.get_by_uid(uid)
        if user:
            return user
        return await self.create(uid, email, display_name, photo_url)

    async def update_settings(self, uid: str, settings: dict[str, Any]) -> dict:
        self._collection.document(uid).update({
            "settings": settings,
            "updated_at": datetime.utcnow(),
        })
        return await self.get_by_uid(uid)


user_repository = UserRepository()
