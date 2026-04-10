from contextlib import asynccontextmanager
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from opensearchpy import AsyncOpenSearch, RequestsHttpConnection
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import aiohttp
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AppConfig(BaseSettings):
    """Uygulama konfigürasyon ayarları"""

    # ---! OpenRouter API Key
    openrouter_api_key: str = Field(
        description="OpenRouter API anahtarı. OpenRouter üzerinden dil modeli çağrıları için gereklidir."
    )

    # ---! DSPy LM Model
    dspy_lm_model: str = Field(
        default="gpt-4o",
        description="DSPy ile kullanılacak dil modeli. Örneğin: 'gpt-4o', 'gpt-3.5-turbo', vb.",
    )

    # ---! Uygulama Portu
    port: int = Field(
        default=8000,
        description="FastAPI uygulamasının çalışacağı port numarası"
    )

    @field_validator("openrouter_api_key", mode="before")
    @classmethod
    def validate_openrouter_api_key(cls, v):
        if not v:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        return v