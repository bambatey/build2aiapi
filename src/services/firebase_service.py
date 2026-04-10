"""
Firebase Admin SDK — Auth token verification & Firestore client.
"""
import logging
from pathlib import Path

import firebase_admin
from firebase_admin import auth, credentials, firestore, storage

from config import app_config

logger = logging.getLogger(__name__)


class FirebaseService:
    def __init__(self) -> None:
        self._app: firebase_admin.App | None = None
        self._db = None

    def initialize(self) -> None:
        """Firebase Admin SDK'yı başlat."""
        if self._app:
            return

        cred_path = Path(__file__).parent.parent.parent / app_config.firebase_credentials_path
        if not cred_path.exists():
            raise FileNotFoundError(f"Firebase credentials bulunamadı: {cred_path}")

        cred = credentials.Certificate(str(cred_path))
        self._app = firebase_admin.initialize_app(cred, {
            "storageBucket": app_config.firebase_storage_bucket,
        })
        self._db = firestore.client()
        logger.info("Firebase Admin SDK başlatıldı")

    @property
    def db(self):
        """Firestore client."""
        if not self._db:
            raise RuntimeError("Firebase henüz başlatılmadı. initialize() çağırın.")
        return self._db

    @property
    def bucket(self):
        """Firebase Storage bucket."""
        return storage.bucket()

    async def verify_token(self, id_token: str) -> dict:
        """
        Firebase ID token'ı doğrula.
        Returns: decoded token dict (uid, email, name, picture, vb.)
        """
        try:
            decoded = auth.verify_id_token(id_token)
            return decoded
        except auth.InvalidIdTokenError:
            raise ValueError("Geçersiz Firebase token")
        except auth.ExpiredIdTokenError:
            raise ValueError("Firebase token süresi dolmuş")
        except Exception as e:
            logger.error(f"Token doğrulama hatası: {e}")
            raise ValueError(f"Token doğrulama başarısız: {e}")

    def get_user(self, uid: str):
        """Firebase Auth'dan kullanıcı bilgisi getir."""
        return auth.get_user(uid)


firebase_service = FirebaseService()
