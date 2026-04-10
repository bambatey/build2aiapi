"""
Firestore:
  users/{uid}/projects/{pid}/chatSessions/{sid}
  users/{uid}/projects/{pid}/chatSessions/{sid}/messages/{mid}
"""
import uuid
from datetime import datetime

from services.firebase_service import firebase_service


class ChatRepository:

    def _sessions_col(self, uid: str, project_id: str):
        return (
            firebase_service.db
            .collection("users").document(uid)
            .collection("projects").document(project_id)
            .collection("chatSessions")
        )

    def _messages_col(self, uid: str, project_id: str, session_id: str):
        return self._sessions_col(uid, project_id).document(session_id).collection("messages")

    # ---- Sessions ----

    async def list_sessions(self, uid: str, project_id: str) -> list[dict]:
        docs = (
            self._sessions_col(uid, project_id)
            .order_by("last_active", direction="DESCENDING")
            .stream()
        )
        sessions = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            data["project_id"] = project_id
            sessions.append(data)
        return sessions

    async def get_session(self, uid: str, project_id: str, session_id: str) -> dict | None:
        doc = self._sessions_col(uid, project_id).document(session_id).get()
        if doc.exists:
            data = doc.to_dict()
            data["id"] = doc.id
            data["project_id"] = project_id
            return data
        return None

    async def create_session(self, uid: str, project_id: str, name: str = "Yeni Sohbet") -> dict:
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        data = {
            "name": name,
            "created_at": now,
            "last_active": now,
        }
        self._sessions_col(uid, project_id).document(session_id).set(data)
        data["id"] = session_id
        data["project_id"] = project_id
        return data

    async def update_session_activity(self, uid: str, project_id: str, session_id: str) -> None:
        self._sessions_col(uid, project_id).document(session_id).set({
            "last_active": datetime.utcnow(),
        }, merge=True)

    async def rename_session(self, uid: str, project_id: str, session_id: str, name: str) -> None:
        self._sessions_col(uid, project_id).document(session_id).set({
            "name": name,
            "last_active": datetime.utcnow(),
        }, merge=True)

    async def delete_session(self, uid: str, project_id: str, session_id: str) -> None:
        self._sessions_col(uid, project_id).document(session_id).delete()

    # ---- Messages ----

    async def list_messages(self, uid: str, project_id: str, session_id: str, limit: int = 100) -> list[dict]:
        docs = (
            self._messages_col(uid, project_id, session_id)
            .order_by("created_at")
            .limit(limit)
            .stream()
        )
        messages = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            data["session_id"] = session_id
            messages.append(data)
        return messages

    async def add_message(
        self,
        uid: str,
        project_id: str,
        session_id: str,
        role: str,
        content: str,
        diff: list[dict] | None = None,
        model: str | None = None,
    ) -> dict:
        message_id = str(uuid.uuid4())
        now = datetime.utcnow()
        data = {
            "role": role,
            "content": content,
            "diff": diff,
            "model": model,
            "created_at": now,
        }
        self._messages_col(uid, project_id, session_id).document(message_id).set(data)
        # Oturum son aktivite güncelle
        await self.update_session_activity(uid, project_id, session_id)
        data["id"] = message_id
        data["session_id"] = session_id
        return data


chat_repository = ChatRepository()
