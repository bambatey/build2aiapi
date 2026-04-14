"""TABLE: blok ayıklayıcı.

.s2k dosyası şu yapıda:

    TABLE:  "JOINT COORDINATES"
       Joint=1   CoordSys=GLOBAL   ...
       Joint=2   ...

    TABLE:  "CONNECTIVITY - FRAME"
       Frame=1   JointI=1   JointJ=2
    END TABLE DATA

Uzun satırlar SAP2000'de ``_`` ile çok satıra bölünür:

    SectionName=70*80   Material=c35   ...   WMod=1 _
         GUID=88a27f4b-...   Notes="Added 16.11.2025 22:50:07"

Bu blokta önce fiziksel satırlar mantıksal satıra birleştirilir, sonra
``TABLE:`` başlıklarına göre gruplanır.
"""

from __future__ import annotations

import re

from .tokens import parse_row

_TABLE_RE = re.compile(r'^TABLE:\s*"([^"]+)"')
# Trailing whitespace + underscore = continuation marker
_CONT_RE = re.compile(r"\s+_\s*$")


def _join_continuations(text: str) -> list[str]:
    """Fiziksel satırları mantıksal satırlara birleştirir.

    Bir satır ``... _`` ile biterse bir sonraki satırla (strip edilmiş hali)
    birleştirilir. Birden fazla devam olabilir.
    """
    logical: list[str] = []
    buffer: str | None = None
    for raw in text.splitlines():
        # Trailing \r temizle
        line = raw.rstrip()
        if _CONT_RE.search(line):
            chunk = _CONT_RE.sub("", line)
            buffer = chunk if buffer is None else buffer + " " + chunk.lstrip()
            continue
        if buffer is not None:
            logical.append(buffer + " " + line.lstrip())
            buffer = None
        else:
            logical.append(line)
    if buffer is not None:
        logical.append(buffer)
    return logical


def extract_tables(text: str) -> dict[str, list[dict[str, str]]]:
    """Dosya metnini TABLE adlarına göre satır listelerine ayır.

    Dönüş: ``{"JOINT COORDINATES": [{"Joint": "1", ...}, ...], ...}``
    """
    tables: dict[str, list[dict[str, str]]] = {}
    current_name: str | None = None

    for line in _join_continuations(text):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$"):   # yorum satırı
            continue
        if stripped == "END TABLE DATA":
            current_name = None
            continue

        m = _TABLE_RE.match(stripped)
        if m:
            current_name = m.group(1)
            tables.setdefault(current_name, [])
            continue

        if current_name is None:
            continue

        row = parse_row(stripped)
        if row:
            tables[current_name].append(row)

    return tables
