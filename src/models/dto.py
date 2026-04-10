"""
Frontend ile uyumlu DTO modelleri.
Frontend'deki BusinessLogicDto, ApiStreamingResponse ve store model'leriyle birebir eşleşir.
"""
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic API response wrapper  (frontend: BusinessLogicDto<T>)
# ---------------------------------------------------------------------------
class BusinessLogicDto(BaseModel, Generic[T]):
    success: bool = True
    data: T | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Streaming SSE response  (frontend: ApiStreamingResponse<T>)
# ---------------------------------------------------------------------------
class ApiStreamingResponse(BaseModel):
    type: Literal["delta", "finish", "done", "file_update"] = "delta"
    content: str = ""
    reasoning: str = ""
    isComplete: bool = False
    error: str | None = None
    file_changes: list[dict[str, Any]] | None = None  # [{line_number, old_value, new_value, reason}]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    id_token: str


class UserDto(BaseModel):
    uid: str
    email: str | None = None
    display_name: str | None = None
    photo_url: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class UpdateSettingsRequest(BaseModel):
    settings: dict[str, Any]


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------
class ProjectCreateRequest(BaseModel):
    name: str
    format: str = ".s2k"
    tags: list[str] = Field(default_factory=list)


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    format: str | None = None
    tags: list[str] | None = None
    progress: int | None = None


class ProjectDto(BaseModel):
    id: str
    name: str
    format: str
    file_count: int = 0
    last_modified: datetime | None = None
    progress: int = 0
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ProjectDetailDto(ProjectDto):
    files: list["FileNodeDto"] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------
class FileNodeDto(BaseModel):
    id: str
    name: str
    type: Literal["file", "folder"] = "file"
    path: str = ""
    format: str | None = None
    size: int | None = None
    line_count: int | None = None
    last_modified: datetime | None = None
    storage_path: str | None = None  # Firebase Storage path


class FileCreateRequest(BaseModel):
    name: str
    format: str = ".s2k"
    content: str = ""


class FileUpdateRequest(BaseModel):
    content: str


class FileContentDto(BaseModel):
    id: str
    name: str
    content: str
    size: int = 0
    line_count: int = 0


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
class ChatSessionDto(BaseModel):
    id: str
    project_id: str
    name: str
    created_at: datetime | None = None
    last_active: datetime | None = None


class ChatSessionCreateRequest(BaseModel):
    name: str = "Yeni Sohbet"


class ChatMessageDto(BaseModel):
    id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    diff: list[dict[str, Any]] | None = None
    model: str | None = None
    created_at: datetime | None = None


class ChatStreamRequest(BaseModel):
    session_id: str
    project_id: str
    messages: list[dict[str, str]]  # [{role, content}]
    model: str = "anthropic/claude-sonnet-4-20250514"
    file_context: str | None = None  # Aktif .s2k dosya içeriği


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
class DocumentDto(BaseModel):
    id: str
    title: str
    content: str | None = None
    document_type: str = "1"
    created_at: datetime | None = None
