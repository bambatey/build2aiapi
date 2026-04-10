from .auth import router as auth_router
from .projects import router as projects_router
from .files import router as files_router
from .chat import router as chat_router
from .documents import router as documents_router

__all__ = [
    "auth_router",
    "projects_router",
    "files_router",
    "chat_router",
    "documents_router",
]
