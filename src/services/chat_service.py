"""
AI Chat Service — Çoklu LLM provider desteği.
- gemini / openrouter: OpenAI-compatible streaming
- replicate: DSPy ile yapılandırılmış çıktı (structured output)
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
        api_key = app_config.llm_api_key
        if not api_key:
            yield f"data: {ApiStreamingResponse(type='finish', error='LLM API key ayarlanmamış.', isComplete=True).model_dump_json()}\n\n"
            return

        provider = app_config.llm_provider

        if provider == "replicate":
            async for chunk in self._stream_dspy(messages, file_context):
                yield chunk
        else:
            llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if file_context:
                llm_messages.append({
                    "role": "system",
                    "content": f"Kullanıcının aktif dosya içeriği:\n```\n{file_context[:50000]}\n```",
                })
            llm_messages.extend(messages)
            llm_model = model or app_config.default_llm_model
            async for chunk in self._stream_openai_compatible(provider, api_key, llm_model, llm_messages):
                yield chunk

    # -------------------------------------------------------------------
    # DSPy — Yapılandırılmış çıktı (Replicate ve diğer DSPy-destekli)
    # -------------------------------------------------------------------
    async def _stream_dspy(
        self, messages: list[dict], file_context: str | None
    ) -> AsyncGenerator[str, None]:
        import asyncio
        try:
            import dspy
            from signature.structural_chat_signature import (
                StructuralChatSignature,
                StructuralChatResponse,
            )

            # Conversation history oluştur (son mesaj hariç)
            history_msgs = [m for m in messages if m["role"] != "system"]
            user_question = history_msgs[-1]["content"] if history_msgs else ""

            conversation_history = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in history_msgs[:-1]
            ) or "İlk mesaj"

            file_content = file_context[:50000] if file_context else "Dosya yüklenmemiş"

            # DSPy çağrısı (sync — asyncio.to_thread ile async yap)
            predict = dspy.Predict(StructuralChatSignature)

            def run_predict():
                return predict(
                    conversation_history=conversation_history,
                    user_question=user_question,
                    file_content=file_content,
                )

            # "Düşünüyorum..." göster
            yield f"data: {ApiStreamingResponse(type='delta', content='', isComplete=False).model_dump_json()}\n\n"

            result = await asyncio.to_thread(run_predict)

            # Structured response'u parse et
            response: StructuralChatResponse = result.response

            # Ana yanıtı markdown formatında oluştur
            full_content = response.answer

            # Bulgular
            if response.findings:
                full_content += "\n\n### Bulgular\n"
                for f in response.findings:
                    full_content += f"- {f}\n"

            # Öneriler
            if response.recommendations:
                full_content += "\n\n### Öneriler\n"
                for r in response.recommendations:
                    full_content += f"- {r}\n"

            # Yönetmelik referansları
            if response.code_references:
                full_content += "\n\n### Yönetmelik Referansları\n"
                for ref in response.code_references:
                    full_content += f"- **{ref.code_name}** {ref.section}: {ref.description}\n"

            # Uyarı
            if response.warning:
                full_content += f"\n\n> **Uyarı:** {response.warning}\n"

            # Güven skoru
            full_content += f"\n\n---\n*Güven skoru: {response.confidence:.0%} | Analiz türü: {response.analysis_type}*"

            # İçeriği chunk'lar halinde gönder (streaming hissi)
            chunk_size = 50
            for i in range(0, len(full_content), chunk_size):
                chunk = full_content[i:i + chunk_size]
                yield f"data: {ApiStreamingResponse(type='delta', content=chunk, isComplete=False).model_dump_json()}\n\n"
                await asyncio.sleep(0.02)  # Küçük gecikme ile streaming hissi

            # Dosya değişiklik önerileri varsa, diff olarak ekle
            if response.suggested_changes:
                diff_content = "\n\n### Önerilen Değişiklikler\n```diff\n"
                for change in response.suggested_changes:
                    diff_content += f"# Satır {change.line_number}: {change.reason}\n"
                    diff_content += f"- {change.old_value}\n"
                    diff_content += f"+ {change.new_value}\n"
                diff_content += "```\n"
                yield f"data: {ApiStreamingResponse(type='delta', content=diff_content, isComplete=False).model_dump_json()}\n\n"

        except Exception as e:
            logger.error(f"DSPy hatası: {e}", exc_info=True)
            yield f"data: {ApiStreamingResponse(type='delta', content=f'DSPy analiz hatası: {str(e)}', isComplete=False).model_dump_json()}\n\n"

        yield f"data: {ApiStreamingResponse(type='finish', isComplete=True).model_dump_json()}\n\n"

    # -------------------------------------------------------------------
    # OpenAI-compatible streaming (Gemini, OpenRouter)
    # -------------------------------------------------------------------
    async def _stream_openai_compatible(
        self, provider: str, api_key: str, model: str, messages: list[dict]
    ) -> AsyncGenerator[str, None]:

        base_urls = {
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        base_url = base_urls.get(provider, base_urls["openrouter"])

        async with httpx.AsyncClient(timeout=300) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": model, "messages": messages, "stream": True},
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        logger.error(f"LLM API hatası ({response.status_code}): {error_body.decode()}")
                        yield f"data: {ApiStreamingResponse(type='finish', error=f'LLM API hatası ({response.status_code}): {error_body.decode()}', isComplete=True).model_dump_json()}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield f"data: {ApiStreamingResponse(type='delta', content=content, isComplete=False).model_dump_json()}\n\n"
                        except json.JSONDecodeError:
                            continue

            except httpx.ReadTimeout:
                yield f"data: {ApiStreamingResponse(type='finish', error='LLM yanıt zaman aşımına uğradı', isComplete=True).model_dump_json()}\n\n"
                return
            except Exception as e:
                logger.error(f"Streaming hatası: {e}")
                yield f"data: {ApiStreamingResponse(type='finish', error=str(e), isComplete=True).model_dump_json()}\n\n"
                return

        yield f"data: {ApiStreamingResponse(type='finish', isComplete=True).model_dump_json()}\n\n"


chat_service = ChatService()
