"""
FastAPI dependency'leri — auth token verification middleware.
Her korumalı endpoint'te Depends(get_current_user) ile kullanılır.

DEV-ONLY bypass: `app_config.dev_auth_bypass=True` ise hiçbir token
doğrulanmaz, sabit bir fake user (`dev_fake_uid`) döner. Bypass production'a
kazara sızmasın diye boot log'una uyarı düşülür (bkz. `app.py`).
"""
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import app_config
from services.firebase_service import firebase_service

logger = logging.getLogger(__name__)

# auto_error=False: bypass modunda Authorization header eksikse 403 atmasın.
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """
    Authorization: Bearer <firebase_id_token> header'ından
    token'ı alır, Firebase ile doğrular, decoded user bilgisini döner.

    DEV bypass: token yoksa veya verify başarısız olsa bile fake user döner.
    """
    if app_config.dev_auth_bypass:
        return _fake_dev_user()

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header gerekli",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        decoded = await firebase_service.verify_token(credentials.credentials)
        return decoded
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_uid(current_user: dict = Depends(get_current_user)) -> str:
    """Kısa yol: sadece uid döner."""
    return current_user["uid"]


def _fake_dev_user() -> dict:
    """DEV bypass'ta dönen sabit user objesi."""
    return {
        "uid": app_config.dev_fake_uid,
        "email": "dev@localhost",
        "name": "Dev User",
        "email_verified": True,
        "_dev_bypass": True,
    }
