"""SAP2000 .s2k satır tokenizer.

Her satır "Key=Value  Key=Value ..." biçiminde. Değerler tırnaklı ya da boşluksuz
olabilir. SAP2000 Windows yerel ayarı ondalıkta virgül üretebilir ("3,8" →
3.8) — sayıya çevirirken defensif davranırız.

Referans: build2ai/app/utils/s2kParser.ts (aynı format).
"""

from __future__ import annotations

import math
import re

_ROW_RE = re.compile(r'(\w+)=("([^"]*)"|(\S+))')


def parse_row(line: str) -> dict[str, str]:
    """Bir satırı Key=Value sözlüğüne çevir.

    Tırnak içindeki boşluklar korunur, tırnak ayıklanır.
    """
    out: dict[str, str] = {}
    for m in _ROW_RE.finditer(line):
        key = m.group(1)
        quoted = m.group(3)
        bare = m.group(4)
        out[key] = quoted if quoted is not None else bare
    return out


def to_float(v: str | None) -> float:
    """Güvenli float dönüşümü. Virgül ondalığı noktaya çevirir. None → NaN."""
    if v is None:
        return math.nan
    try:
        return float(v.replace(",", "."))
    except (TypeError, ValueError):
        return math.nan


def to_int(v: str | None) -> int | None:
    """Güvenli int dönüşümü. Sayısal olmayan değer için None döner."""
    if v is None:
        return None
    try:
        return int(float(v.replace(",", ".")))
    except (TypeError, ValueError):
        return None
