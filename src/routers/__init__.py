from .analysis import router as analysis_router
from .auth import router as auth_router
from .cad_export import router as cad_export_router
from .chat import router as chat_router
from .design import router as design_router
from .documents import router as documents_router
from .files import router as files_router
from .projects import router as projects_router

__all__ = [
    "analysis_router",
    "auth_router",
    "cad_export_router",
    "chat_router",
    "design_router",
    "documents_router",
    "files_router",
    "projects_router",
]
