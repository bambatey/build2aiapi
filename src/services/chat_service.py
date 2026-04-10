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

            # Son kullanıcı mesajını al
            history_msgs = [m for m in messages if m["role"] != "system"]
            user_question = history_msgs[-1]["content"] if history_msgs else ""

            # Dosya içeriğini kısalt (token limiti için)
            file_content = file_context[:30000] if file_context else "Dosya yüklenmemiş"

            predict = dspy.Predict(StructuralChatSignature)

            def run_predict():
                return predict(
                    user_question=user_question,
                    file_content=file_content,
                )

            # "Düşünüyorum..." göster
            yield f"data: {ApiStreamingResponse(type='delta', content='', isComplete=False).model_dump_json()}\n\n"

            result = await asyncio.to_thread(run_predict)

            # Structured response'u parse et
            response: StructuralChatResponse = result.response

            # Debug log
            logger.info(f"DSPy yanıt alındı — answer: {len(response.answer)} char")
            logger.info(f"DSPy updated_tables: {len(response.updated_tables)} char")
            logger.info(f"DSPy change_summary: '{response.change_summary}'")

            # Ana yanıt
            full_content = response.answer

            # İçeriği chunk'lar halinde gönder
            chunk_size = 50
            for i in range(0, len(full_content), chunk_size):
                chunk = full_content[i:i + chunk_size]
                yield f"data: {ApiStreamingResponse(type='delta', content=chunk, isComplete=False).model_dump_json()}\n\n"
                await asyncio.sleep(0.02)

            # Dosya güncellemesi varsa — tabloları orijinal dosyaya merge et
            if response.updated_tables and response.updated_tables.strip() and file_content != "Dosya yüklenmemiş":
                merged = self._merge_tables(file_content, response.updated_tables)

                if response.change_summary:
                    summary_content = f"\n\n### Dosya Güncellendi\n{response.change_summary}\n"
                    yield f"data: {ApiStreamingResponse(type='delta', content=summary_content, isComplete=False).model_dump_json()}\n\n"

                yield f"data: {ApiStreamingResponse(type='file_update', content=merged, isComplete=False).model_dump_json()}\n\n"
                logger.info(f"Dosya merge edildi: {len(file_content)} → {len(merged)} char")

        except Exception as e:
            logger.error(f"DSPy hatası: {e}", exc_info=True)
            yield f"data: {ApiStreamingResponse(type='delta', content=f'DSPy analiz hatası: {str(e)}', isComplete=False).model_dump_json()}\n\n"

        yield f"data: {ApiStreamingResponse(type='finish', isComplete=True).model_dump_json()}\n\n"

    @staticmethod
    def _merge_tables(original: str, updated_tables: str) -> str:
        """
        AI'ın ürettiği TABLE bloklarını orijinal dosyaya merge eder.
        - Aynı isimli tablo varsa → AI'ın satırlarını mevcut tabloya EKLE (append)
        - Yeni tablo ise → END TABLE DATA'dan önce ekle
        - Duplicate satırları engelle (aynı Joint= veya Frame= varsa ekleme)
        """
        import re

        def extract_table_rows(text: str) -> dict[str, list[str]]:
            """Metinden TABLE adı → veri satırları dict'i çıkar."""
            tables: dict[str, list[str]] = {}
            current_name = None

            for line in text.split('\n'):
                match = re.match(r'^\s*TABLE:\s*"([^"]+)"', line)
                if match:
                    current_name = match.group(1)
                    if current_name not in tables:
                        tables[current_name] = []
                    continue

                if current_name and line.strip() and not line.strip().startswith('$'):
                    if line.strip() == 'END TABLE DATA':
                        current_name = None
                        continue
                    tables[current_name].append(line.rstrip())

            return tables

        def get_row_key(line: str) -> str | None:
            """Satırdan unique key çıkar (Joint=X, Frame=X gibi)."""
            m = re.match(r'\s*(Joint|Frame|Area|LoadPat|Material|SectionName)=(\S+)', line)
            if m:
                return f"{m.group(1)}={m.group(2)}"
            return None

        new_tables = extract_table_rows(updated_tables)
        if not new_tables:
            return original

        # Orijinal dosyadaki mevcut satır key'lerini tabloya göre topla
        orig_tables = extract_table_rows(original)

        # Orijinal dosyayı satır satır işle, ilgili tablo sonuna yeni satırları ekle
        result_lines = []
        original_lines = original.split('\n')
        appended_tables: set[str] = set()
        current_table: str | None = None
        i = 0

        while i < len(original_lines):
            line = original_lines[i]
            table_match = re.match(r'^\s*TABLE:\s*"([^"]+)"', line)

            if table_match:
                current_table = table_match.group(1)
                result_lines.append(line)
                i += 1
                continue

            # Yeni tablo başlıyor veya END TABLE DATA — mevcut tabloya ekleme yap
            next_is_new_table = False
            if i + 1 < len(original_lines):
                next_is_new_table = bool(re.match(r'^\s*TABLE:\s*"', original_lines[i + 1]))

            is_empty_before_next = (
                line.strip() == '' and
                i + 1 < len(original_lines) and
                (next_is_new_table or original_lines[i + 1].strip() == 'END TABLE DATA')
            )

            if (is_empty_before_next or line.strip() == 'END TABLE DATA') and current_table and current_table in new_tables and current_table not in appended_tables:
                # Mevcut tablodaki key'leri topla
                existing_keys = set()
                for orig_row in orig_tables.get(current_table, []):
                    key = get_row_key(orig_row)
                    if key:
                        existing_keys.add(key)

                # AI'ın yeni satırlarını ekle (duplicate olmayanları)
                new_rows = new_tables[current_table]
                added = 0
                for new_row in new_rows:
                    row_key = get_row_key(new_row)
                    if row_key and row_key in existing_keys:
                        continue  # Zaten var, ekleme
                    result_lines.append(new_row)
                    added += 1

                appended_tables.add(current_table)
                logger.info(f"Merge: {current_table} tablosuna {added} satır eklendi")

            if line.strip() == 'END TABLE DATA':
                # Tamamen yeni tabloları ekle
                for name, rows in new_tables.items():
                    if name not in appended_tables and name not in orig_tables:
                        result_lines.append('')
                        result_lines.append(f'TABLE:  "{name}"')
                        for row in rows:
                            result_lines.append(row)
                        result_lines.append('')
                        appended_tables.add(name)
                        logger.info(f"Merge: Yeni tablo eklendi: {name} ({len(rows)} satır)")

                current_table = None

            result_lines.append(line)
            i += 1

        return '\n'.join(result_lines)

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
