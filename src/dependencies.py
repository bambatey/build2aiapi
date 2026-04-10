"""
FastAPI dependency'leri — auth token verification middleware.
Her korumalı endpoint'te Depends(get_current_user) ile kullanılır.
"""
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.firebase_service import firebase_service

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Authorization: Bearer <firebase_id_token> header'ından
    token'ı alır, Firebase ile doğrular, decoded user bilgisini döner.
    """
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
