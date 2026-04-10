"""
Chat Router
GET  /api/projects/{pid}/chat/sessions                    → Sohbet oturumları
POST /api/projects/{pid}/chat/sessions                    → Yeni oturum
GET  /api/chat/sessions/{sid}/messages?project_id=X       → Mesajlar
POST /api/llm-proxy/chat/stream                           → AI streaming (SSE)
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from dependencies import get_current_user, get_uid
from models.dto import (
    BusinessLogicDto,
    ChatMessageDto,
    ChatSessionCreateRequest,
    ChatSessionDto,
    ChatStreamRequest,
)
from repositories import chat_repository
from services import chat_service

router = APIRouter(tags=["chat"])


# ---! Session endpoints
@router.get("/api/projects/{project_id}/chat/sessions", response_model=BusinessLogicDto)
async def list_sessions(project_id: str, uid: str = Depends(get_uid)):
    """Projedeki sohbet oturumlarını listele."""
    sessions = await chat_repository.list_sessions(uid, project_id)
    session_dtos = [
        ChatSessionDto(
            id=s["id"],
            project_id=s.get("project_id", project_id),
            name=s.get("name", ""),
            created_at=s.get("created_at"),
            last_active=s.get("last_active"),
        )
        for s in sessions
    ]
    return BusinessLogicDto(success=True, data=session_dtos)


@router.post("/api/projects/{project_id}/chat/sessions", response_model=BusinessLogicDto)
async def create_session(
    project_id: str,
    request: ChatSessionCreateRequest,
    uid: str = Depends(get_uid),
):
    """Yeni sohbet oturumu oluştur."""
    session = await chat_repository.create_session(uid, project_id, request.name)
    return BusinessLogicDto(
        success=True,
        data=ChatSessionDto(
            id=session["id"],
            project_id=project_id,
            name=session["name"],
            created_at=session.get("created_at"),
            last_active=session.get("last_active"),
        ),
    )


@router.get("/api/chat/sessions/{session_id}/messages", response_model=BusinessLogicDto)
async def list_messages(
    session_id: str,
    project_id: str = Query(...),
    uid: str = Depends(get_uid),
):
    """Oturumdaki mesajları getir."""
    messages = await chat_repository.list_messages(uid, project_id, session_id)
    message_dtos = [
        ChatMessageDto(
            id=m["id"],
            session_id=m.get("session_id", session_id),
            role=m["role"],
            content=m["content"],
            diff=m.get("diff"),
            model=m.get("model"),
            created_at=m.get("created_at"),
        )
        for m in messages
    ]
    return BusinessLogicDto(success=True, data=message_dtos)


# ---! AI Streaming endpoint
@router.post("/api/llm-proxy/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    AI chat streaming endpoint (SSE).
    Frontend'in postStream() metoduyla uyumlu SSE formatında yanıt döner.
    """
    uid = current_user["uid"]

    # Kullanıcı mesajını DB'ye kaydet
    user_messages = [m for m in request.messages if m.get("role") == "user"]
    if user_messages:
        last_user_msg = user_messages[-1]
        await chat_repository.add_message(
            uid=uid,
            project_id=request.project_id,
            session_id=request.session_id,
            role="user",
            content=last_user_msg["content"],
        )

    # Streaming response oluştur
    async def generate():
        full_content = ""
        async for chunk in chat_service.stream_chat(
            messages=request.messages,
            model=request.model,
            file_context=request.file_context,
        ):
            yield chunk
            # İçeriği biriktir (DB'ye kaydetmek için)
            if '"type": "delta"' in chunk or '"type":"delta"' in chunk:
                import json
                try:
                    data_str = chunk.replace("data: ", "").strip()
                    parsed = json.loads(data_str)
                    full_content += parsed.get("content", "")
                except Exception:
                    pass

        # AI yanıtını DB'ye kaydet
        if full_content:
            await chat_repository.add_message(
                uid=uid,
                project_id=request.project_id,
                session_id=request.session_id,
                role="assistant",
                content=full_content,
                model=request.model,
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
