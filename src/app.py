import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import app_config
from services.firebase_service import firebase_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: Firebase init / Shutdown: cleanup."""
    logger.info("Uygulama başlatılıyor...")
    firebase_service.initialize()
    logger.info("Firebase Admin SDK hazır")
    yield
    logger.info("Uygulama kapatılıyor...")


app = FastAPI(
    title="Build2AI API",
    description="StructAI yapısal mühendislik AI asistanı backend API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---! CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---! Routers
from routers import (
    auth_router,
    chat_router,
    documents_router,
    files_router,
    projects_router,
)

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(files_router)
app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=app_config.port,
        log_level="info",
        reload=True,
    )
