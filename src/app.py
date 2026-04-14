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
    """Startup: Firebase + DSPy init / Shutdown: cleanup."""
    logger.info("Uygulama başlatılıyor...")
    firebase_service.initialize()
    logger.info("Firebase Admin SDK hazır")

    # ---! DSPy LM yapılandır
    if app_config.llm_api_key:
        import dspy
        provider = app_config.llm_provider
        model_name = app_config.default_llm_model

        if provider == "replicate":
            lm = dspy.LM(
                model=f"replicate/{model_name}",
                api_key=app_config.llm_api_key,
            )
        elif provider == "gemini":
            lm = dspy.LM(
                model=f"gemini/{model_name}",
                api_key=app_config.llm_api_key,
            )
        elif provider == "openrouter":
            lm = dspy.LM(
                model=f"openrouter/{model_name}",
                api_key=app_config.llm_api_key,
                api_base="https://openrouter.ai/api/v1",
            )
        else:
            lm = dspy.LM(model=model_name, api_key=app_config.llm_api_key)

        dspy.configure(lm=lm)
        logger.info(f"DSPy LM yapılandırıldı: {provider}/{model_name}")

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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---! Routers
from routers import (
    analysis_router,
    auth_router,
    chat_router,
    documents_router,
    files_router,
    projects_router,
)

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(files_router)
app.include_router(analysis_router)
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
