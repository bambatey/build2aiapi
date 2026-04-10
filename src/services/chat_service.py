"""
AI Chat Service — OpenRouter üzerinden LLM streaming.
Frontend'in beklediği SSE formatında (data: {type, content, reasoning, isComplete}) yanıt üretir.
"""
import json
import logging
from collections.abc import AsyncGenerator

import httpx

from config import app_config
from models.dto import ApiStreamingResponse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sen StructAI, yapısal mühendislik konusunda uzmanlaşmış bir AI asistanısın.
Görevlerin:
- SAP2000, ETABS, RISA-3D, STAAD Pro model dosyalarını analiz etmek
- TBDY 2018, ASCE 7-22, Eurocode 8, ACI 318 yönetmeliklerine göre kontrol yapmak
- Kesit optimizasyonu, deprem analizi, yük kontrolleri yapmak
- Yapısal mühendislik sorularını yanıtlamak

Kullanıcı sana bir .s2k veya benzeri dosya içeriği verebilir. Bu durumda dosyayı analiz et.
Yanıtlarını Türkçe ver (kullanıcı İngilizce sorarsa İngilizce yanıtla).
Markdown formatı kullan."""


class ChatService:

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        file_context: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        OpenRouter API'ye streaming request at, SSE formatında yield et.
        Frontend postStream() metodunun beklediği formatta.
        """
        llm_model = model or app_config.default_llm_model
        api_key = app_config.openrouter_api_key

        if not api_key:
            error_response = ApiStreamingResponse(
                type="finish",
                content="",
                error="OpenRouter API key ayarlanmamış. .env dosyasına OPENROUTER_API_KEY ekleyin.",
                isComplete=True,
            )
            yield f"data: {error_response.model_dump_json()}\n\n"
            return

        # ---! Sistem prompt'u ve dosya bağlamını mesajlara ekle
        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if file_context:
            llm_messages.append({
                "role": "system",
                "content": f"Kullanıcının aktif dosya içeriği:\n```\n{file_context[:50000]}\n```",
            })

        llm_messages.extend(messages)

        # ---! OpenRouter streaming request
        async with httpx.AsyncClient(timeout=300) as client:
            try:
                async with client.stream(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": llm_model,
                        "messages": llm_messages,
                        "stream": True,
                    },
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        error_msg = f"LLM API hatası ({response.status_code}): {error_body.decode()}"
                        logger.error(error_msg)
                        error_response = ApiStreamingResponse(
                            type="finish",
                            content="",
                            error=error_msg,
                            isComplete=True,
                        )
                        yield f"data: {error_response.model_dump_json()}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # "data: " prefix'ini kaldır
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                sse_response = ApiStreamingResponse(
                                    type="delta",
                                    content=content,
                                    reasoning="",
                                    isComplete=False,
                                )
                                yield f"data: {sse_response.model_dump_json()}\n\n"

                        except json.JSONDecodeError:
                            continue

            except httpx.ReadTimeout:
                error_response = ApiStreamingResponse(
                    type="finish",
                    content="",
                    error="LLM yanıt zaman aşımına uğradı (5 dakika)",
                    isComplete=True,
                )
                yield f"data: {error_response.model_dump_json()}\n\n"
                return

            except Exception as e:
                logger.error(f"Streaming hatası: {e}")
                error_response = ApiStreamingResponse(
                    type="finish",
                    content="",
                    error=str(e),
                    isComplete=True,
                )
                yield f"data: {error_response.model_dump_json()}\n\n"
                return

        # ---! Stream tamamlandı
        finish_response = ApiStreamingResponse(
            type="finish",
            content="",
            reasoning="",
            isComplete=True,
        )
        yield f"data: {finish_response.model_dump_json()}\n\n"


chat_service = ChatService()
