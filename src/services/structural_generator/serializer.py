"""TABLE bloklarını tekrar .s2k metnine serialize eder.

``parser.tables.extract_tables`` 'ın TERSİ. Editor pipeline'ı için temel:

    extract_tables(s2k_text) → modify dict → serialize_tables(dict) → new s2k

Çıktı SAP2000'in beklediği formata yakın:
    TABLE:  "ADI"
       Key=Value   Key=Value   ...
       Key=Value   Key=Value   ...

    TABLE:  "DİĞER"
       ...

    END TABLE DATA

Tablo sırası mümkün olduğunca anlamlı tutulur (PROGRAM CONTROL → MATERIAL →
SECTION → JOINT → FRAME → ASSIGNMENT → RESTRAINT → LOAD → COMBINATION).
"""

from __future__ import annotations


# Tablo serileştirme sırası — okunabilirlik için
_PREFERRED_ORDER = [
    "PROGRAM CONTROL",
    "MATERIAL PROPERTIES 01 - GENERAL",
    "MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES",
    "FRAME SECTION PROPERTIES 01 - GENERAL",
    "AREA SECTION PROPERTIES",
    "JOINT COORDINATES",
    "POINT COORDINATES",
    "CONNECTIVITY - FRAME",
    "CONNECTIVITY - LINE",
    "CONNECTIVITY - AREA",
    "FRAME SECTION ASSIGNMENTS",
    "AREA SECTION ASSIGNMENTS",
    "JOINT RESTRAINT ASSIGNMENTS",
    "LOAD PATTERN DEFINITIONS",
    "MASS SOURCE",
    "JOINT LOADS - FORCE",
    "FRAME LOADS - DISTRIBUTED",
    "AREA LOADS - UNIFORM TO FRAME",
    "AUTO SEISMIC LOADS TO JOINTS",
    "AUTO WIND LOADS TO JOINTS",
    "COMBINATION DEFINITIONS",
]


def _format_value(v: str) -> str:
    """Boşluk içeren değerleri tırnak içine al."""
    if " " in v and not (v.startswith('"') and v.endswith('"')):
        return f'"{v}"'
    return v


def _format_row(row: dict[str, str]) -> str:
    """Tek bir satır: 3 boşluk girinti + Key=Value boşluk Key=Value..."""
    return "   " + "   ".join(f"{k}={_format_value(str(v))}" for k, v in row.items())


def serialize_tables(tables: dict[str, list[dict[str, str]]]) -> str:
    """Tablolardan .s2k metnini geri üret."""
    out: list[str] = []
    out.append("$ Build2AI generated/edited file")
    out.append("")

    # Önce preferred order'daki tabloları yaz (varsa), sonra geri kalanları
    seen: set[str] = set()
    for name in _PREFERRED_ORDER:
        if name in tables and tables[name]:
            _write_table(out, name, tables[name])
            seen.add(name)
    for name, rows in tables.items():
        if name in seen or not rows:
            continue
        _write_table(out, name, rows)

    out.append("END TABLE DATA")
    out.append("")
    return "\n".join(out)


def _write_table(out: list[str], name: str, rows: list[dict[str, str]]) -> None:
    out.append(f'TABLE:  "{name}"')
    for row in rows:
        out.append(_format_row(row))
    out.append("")
