"""
AI Chat Service — İki mod:
 - DSPy (structural_intent): yapısal modeli oluştur/düzenle + analiz sorgula
 - OpenAI-compatible streaming: genel soru/cevap fallback
"""
import json
import logging
from collections.abc import AsyncGenerator

import httpx

from config import app_config
from models.dto import ApiStreamingResponse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_FALLBACK = """Sen StructAI, yapısal mühendislik konusunda uzmanlaşmış bir AI asistanısın.
Görevlerin:
- SAP2000, ETABS, RISA-3D, STAAD Pro model dosyalarını analiz etmek
- TBDY 2018, ASCE 7-22, Eurocode 8, ACI 318 yönetmeliklerine göre kontrol yapmak
- Kesit optimizasyonu, deprem analizi, yük kontrolleri yapmak
- Yapısal mühendislik sorularını yanıtlamak

Yanıtlarını Türkçe ver. Markdown formatı kullan."""


class ChatService:

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        file_context: str | None = None,
        analysis_summary: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Ana chat akışı — provider seçimine göre yönlendirir.

        DSPy yolu: niyet çıkarımı + deterministik model üretimi/edit.
        OpenAI yolu: genel sohbet (fallback).
        """
        api_key = app_config.llm_api_key
        if not api_key:
            yield _finish(error="LLM API key ayarlanmamış.")
            return

        provider = app_config.llm_provider

        if provider == "replicate":
            async for chunk in self._stream_dspy_intent(
                messages, file_context, analysis_summary,
            ):
                yield chunk
        else:
            llm_messages = [{"role": "system", "content": SYSTEM_PROMPT_FALLBACK}]
            if file_context:
                llm_messages.append({
                    "role": "system",
                    "content": f"Kullanıcının aktif .s2k içeriği (kısaltılmış):\n```\n{file_context[:50000]}\n```",
                })
            if analysis_summary:
                llm_messages.append({
                    "role": "system",
                    "content": f"Son analiz sonucu özeti:\n{analysis_summary}",
                })
            llm_messages.extend(messages)
            llm_model = model or app_config.default_llm_model
            async for chunk in self._stream_openai_compatible(
                provider, api_key, llm_model, llm_messages,
            ):
                yield chunk

    # -------------------------------------------------------------------
    # DSPy intent + structural_generator flow
    # -------------------------------------------------------------------
    async def _stream_dspy_intent(
        self,
        messages: list[dict],
        file_context: str | None,
        analysis_summary: str | None,
    ) -> AsyncGenerator[str, None]:
        """AI intent çıkarır → backend generator/editor çağırır.

        Bu akış:
          1. Son kullanıcı mesajı + mevcut dosya + analiz özeti → DSPy
          2. DSPy intent + chat_response döndürür
          3. Intent 'create_rc_frame' ise generator çağırılır, file_update event yollanır
          4. Intent 'edit_model' → (MVP: şu an not supported, ileride delta editor)
          5. Intent 'query' → sadece chat_response döner
        """
        import asyncio
        try:
            import dspy
            from signature.structural_intent import StructuralChatV2
            from services.structural_generator import (
                RCFrameParams,
                generate_rc_frame,
                validate_generated_model,
                GenerationError,
            )

            history_msgs = [m for m in messages if m["role"] != "system"]
            user_question = history_msgs[-1]["content"] if history_msgs else ""

            current_file = file_context[:30000] if file_context else "Dosya yok"
            analysis_ctx = analysis_summary or "Analiz yapılmamış"

            predict = dspy.Predict(StructuralChatV2)

            def run_predict():
                return predict(
                    user_message=user_question,
                    current_file=current_file,
                    analysis_summary=analysis_ctx,
                )

            # Placeholder delta (UI'da "düşünüyor" göstermesi için)
            yield _delta("")

            result = await asyncio.to_thread(run_predict)
            intent = result.intent
            chat_response: str = result.chat_response or ""

            logger.info(
                "DSPy intent: action=%s rc_params=%s edit=%s",
                intent.action,
                intent.rc_frame_params.model_dump() if intent.rc_frame_params else None,
                intent.edit_description,
            )

            # Chat yanıtını chunk'la akıt
            async for piece in _streamed(chat_response):
                yield piece

            # Intent'e göre dosya etkisi
            if intent.action == "create_rc_frame":
                params = _params_from_intent(intent.rc_frame_params)
                try:
                    s2k_text = generate_rc_frame(params)
                    report = validate_generated_model(s2k_text)
                except GenerationError as exc:
                    yield _delta(f"\n\n⚠️ Model üretimi başarısız: {exc}")
                else:
                    summary_line = (
                        f"\n\n### ✅ Model oluşturuldu\n"
                        f"- Grid: **{params.bays_x}x{params.bays_y}**, "
                        f"**{params.stories} kat**\n"
                        f"- Düğüm: {report.n_nodes}, Frame: {report.n_frames}\n"
                        f"- Beton: **C{params.fck_mpa}**, "
                        f"Kolon: **{int(params.col_size*100)}x{int(params.col_size*100)}**, "
                        f"Kiriş: **{int(params.beam_width*100)}x{int(params.beam_height*100)}**\n"
                    )
                    yield _delta(summary_line)
                    # Dosya güncelleme event'i → frontend dosyayı overwrite edecek
                    yield _file_update(s2k_text)
                    logger.info(
                        "create_rc_frame: %d düğüm, %d frame üretildi (%d char)",
                        report.n_nodes, report.n_frames, len(s2k_text),
                    )

            elif intent.action == "edit_model":
                # MVP: şu an edit modu kapalı; kullanıcıya nazikçe bildir.
                note = (
                    "\n\n_Not: Mevcut modelde inline düzenleme yakında geliyor. "
                    "Şimdilik yeni bir oluşturma isteği verebilirsin._"
                )
                yield _delta(note)

            # "query" için ek bir şey yapmaya gerek yok — sadece response yeterli.

        except Exception as e:
            logger.error(f"DSPy intent hatası: {e}", exc_info=True)
            yield _delta(f"\n\nAI analiz hatası: {e}")

        yield _finish()

    # -------------------------------------------------------------------
    # OpenAI-compatible streaming (Gemini, OpenRouter) — fallback
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
                        yield _finish(error=f"LLM API hatası ({response.status_code}): {error_body.decode()}")
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
                                yield _delta(content)
                        except json.JSONDecodeError:
                            continue

            except httpx.ReadTimeout:
                yield _finish(error="LLM yanıt zaman aşımına uğradı")
                return
            except Exception as e:
                logger.error(f"Streaming hatası: {e}")
                yield _finish(error=str(e))
                return

        yield _finish()


# --------------------------------------------------------------------- helpers
def _delta(content: str) -> str:
    return f"data: {ApiStreamingResponse(type='delta', content=content, isComplete=False).model_dump_json()}\n\n"


def _file_update(content: str) -> str:
    return f"data: {ApiStreamingResponse(type='file_update', content=content, isComplete=False).model_dump_json()}\n\n"


def _finish(error: str | None = None) -> str:
    return f"data: {ApiStreamingResponse(type='finish', error=error, isComplete=True).model_dump_json()}\n\n"


async def _streamed(text: str, chunk_size: int = 80):
    """Uzun cevabı chunk'lara böl — UI'da akıyor gibi görünsün."""
    import asyncio
    for i in range(0, len(text), chunk_size):
        yield _delta(text[i:i + chunk_size])
        await asyncio.sleep(0.015)


def _params_from_intent(p) -> "RCFrameParams":  # type: ignore[name-defined]
    """RCFrameParamsDto → RCFrameParams (None'ları atlar, default'a düşer)."""
    from services.structural_generator import RCFrameParams
    if p is None:
        return RCFrameParams()
    kwargs: dict = {}
    mapping = {
        "bays_x": "bays_x", "bays_y": "bays_y", "stories": "stories",
        "bay_dx_m": "bay_dx", "bay_dy_m": "bay_dy", "story_h_m": "story_h",
        "col_size_m": "col_size", "beam_w_m": "beam_width", "beam_h_m": "beam_height",
        "fck_mpa": "fck_mpa", "dead_q_knm": "dead_q", "live_q_knm": "live_q",
    }
    for dto_field, param_field in mapping.items():
        val = getattr(p, dto_field, None)
        if val is not None:
            kwargs[param_field] = val
    return RCFrameParams(**kwargs)


chat_service = ChatService()
