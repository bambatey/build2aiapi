import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Uygulama konfigürasyon ayarları"""

    # ---! Firebase
    firebase_credentials_path: str = Field(
        default="build2ai-firebase-adminsdk-fbsvc-3d1630b69e.json",
        description="Firebase service account JSON dosya yolu",
    )
    firebase_storage_bucket: str = Field(
        default="build2ai.firebasestorage.app",
        description="Firebase Storage bucket adı",
    )

    # ---! LLM
    llm_provider: str = Field(
        default="gemini",
        description="LLM provider: 'gemini' veya 'openrouter'",
    )
    llm_api_key: str = Field(
        default="",
        description="LLM API anahtarı (Gemini veya OpenRouter)",
    )
    default_llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Varsayılan LLM modeli",
    )

    # ---! Uygulama
    port: int = Field(default=8000)
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="İzin verilen CORS origin'leri",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_config() -> AppConfig:
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    return AppConfig()


app_config = get_config()
