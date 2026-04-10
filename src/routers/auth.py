"""
Auth Router
POST /api/auth/login      → Firebase token verify, kullanıcı oluştur/getir
GET  /api/auth/me          → Mevcut kullanıcı bilgisi
PUT  /api/auth/settings    → Kullanıcı ayarlarını güncelle
"""
from fastapi import APIRouter, Depends, HTTPException

from dependencies import get_current_user, get_uid
from models.dto import (
    BusinessLogicDto,
    LoginRequest,
    UpdateSettingsRequest,
    UserDto,
)
from repositories import user_repository

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=BusinessLogicDto)
async def login(request: LoginRequest):
    """
    Firebase ID token ile giriş yap.
    Kullanıcı yoksa otomatik oluşturulur.
    """
    from services.firebase_service import firebase_service

    try:
        decoded = await firebase_service.verify_token(request.id_token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    uid = decoded["uid"]
    email = decoded.get("email")
    name = decoded.get("name")
    picture = decoded.get("picture")

    user = await user_repository.get_or_create(uid, email, name, picture)

    return BusinessLogicDto(
        success=True,
        data=UserDto(
            uid=uid,
            email=user.get("email"),
            display_name=user.get("display_name"),
            photo_url=user.get("photo_url"),
            settings=user.get("settings", {}),
            created_at=user.get("created_at"),
            updated_at=user.get("updated_at"),
        ),
    )


@router.get("/me", response_model=BusinessLogicDto)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Mevcut oturum sahibi kullanıcı bilgisi."""
    uid = current_user["uid"]
    user = await user_repository.get_by_uid(uid)

    if not user:
        user = await user_repository.create(
            uid,
            current_user.get("email"),
            current_user.get("name"),
            current_user.get("picture"),
        )

    return BusinessLogicDto(
        success=True,
        data=UserDto(
            uid=uid,
            email=user.get("email"),
            display_name=user.get("display_name"),
            photo_url=user.get("photo_url"),
            settings=user.get("settings", {}),
            created_at=user.get("created_at"),
            updated_at=user.get("updated_at"),
        ),
    )


@router.put("/settings", response_model=BusinessLogicDto)
async def update_settings(
    request: UpdateSettingsRequest,
    uid: str = Depends(get_uid),
):
    """Kullanıcı ayarlarını güncelle (dil, tema, API key, yönetmelikler vb.)."""
    user = await user_repository.update_settings(uid, request.settings)
    return BusinessLogicDto(
        success=True,
        data=UserDto(
            uid=uid,
            email=user.get("email"),
            display_name=user.get("display_name"),
            photo_url=user.get("photo_url"),
            settings=user.get("settings", {}),
            created_at=user.get("created_at"),
            updated_at=user.get("updated_at"),
        ),
    )
