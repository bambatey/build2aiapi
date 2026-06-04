"""Eleman tabloları — kolon listesi ve kiriş listesi.

Sayfa boyutu: değişken (satır sayısına göre). Modelspace içinde sağda kat
planının yanına yerleşir.

Sütunlar (kolon tablosu):
  No | Kesit  | t2 (cm) | t3 (cm) | Adet
Sütunlar (kiriş tablosu):
  No | Kesit  | b (cm)  | h (cm)  | Adet
"""

from __future__ import annotations

from dataclasses import dataclass

from ezdxf.enums import TextEntityAlignment as TA
from ezdxf.layouts import Modelspace

from .geometry import StoryGeom

ROW_H = 0.45
TITLE_H = 0.32
HEADER_H = 0.24
CELL_H = 0.20


@dataclass(frozen=True)
class ColumnSpec:
    """Tablonun bir kolonu — başlık + genişlik (m)."""
    title: str
    width: float


COL_TABLE_COLS = (
    ColumnSpec("No", 0.7),
    ColumnSpec("Kesit", 2.2),
    ColumnSpec("b (cm)", 1.3),
    ColumnSpec("h (cm)", 1.3),
    ColumnSpec("Adet", 1.0),
)
BEAM_TABLE_COLS = (
    ColumnSpec("No", 0.7),
    ColumnSpec("Kesit", 2.4),
    ColumnSpec("b (cm)", 1.3),
    ColumnSpec("Adet", 1.0),
)


def draw_schedule(
    msp: Modelspace,
    story: StoryGeom,
    origin: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Bir kat için kolon ve kiriş tablolarını alt alta çiz."""
    ox, oy = origin

    # Sayım: kesit adına göre group
    col_counts: dict[str, tuple[float, float, int]] = {}
    for c in story.columns:
        key = c.section_name
        prev = col_counts.get(key)
        col_counts[key] = (c.t2, c.t3, (prev[2] if prev else 0) + 1)

    beam_counts: dict[str, tuple[float, int]] = {}
    for b in story.beams:
        key = b.section_name
        prev = beam_counts.get(key)
        beam_counts[key] = (b.width, (prev[1] if prev else 0) + 1)

    # KOLON tablosu (üstte)
    rows_col = []
    for i, (sec, (t2, t3, n)) in enumerate(sorted(col_counts.items()), start=1):
        rows_col.append([
            str(i), sec,
            f"{int(round(t2 * 100))}", f"{int(round(t3 * 100))}",
            str(n),
        ])
    h_col = _draw_table(msp, ox, oy, "KOLON LİSTESİ", COL_TABLE_COLS, rows_col)

    # KİRİŞ tablosu (kolon tablosunun altında)
    beam_y = oy - h_col - 0.40
    rows_beam = []
    for i, (sec, (b, n)) in enumerate(sorted(beam_counts.items()), start=1):
        rows_beam.append([
            str(i), sec,
            f"{int(round(b * 100))}", str(n),
        ])
    h_beam = _draw_table(msp, ox, beam_y, "KİRİŞ LİSTESİ",
                         BEAM_TABLE_COLS, rows_beam)

    total_h = h_col + 0.40 + h_beam
    total_w = max(sum(c.width for c in COL_TABLE_COLS),
                  sum(c.width for c in BEAM_TABLE_COLS))
    return (ox, oy - total_h, ox + total_w, oy)


def _draw_table(
    msp: Modelspace,
    ox: float, oy: float,
    title: str,
    cols: tuple[ColumnSpec, ...],
    rows: list[list[str]],
) -> float:
    """Bir tablo çiz, dönüş: kapladığı yükseklik (m)."""
    total_w = sum(c.width for c in cols)
    # Başlık satırı
    title_y = oy - ROW_H
    msp.add_lwpolyline(
        [(ox, title_y), (ox + total_w, title_y),
         (ox + total_w, oy), (ox, oy)],
        close=True,
        dxfattribs={"layer": "TABLE"},
    )
    msp.add_text(
        title,
        dxfattribs={"layer": "TABLE", "height": TITLE_H, "style": "STRUCTAI"},
    ).set_placement((ox + total_w / 2.0, oy - ROW_H / 2.0), align=TA.MIDDLE_CENTER)

    # Sütun başlığı satırı
    header_y = title_y - ROW_H
    cx = ox
    for c in cols:
        msp.add_lwpolyline(
            [(cx, header_y), (cx + c.width, header_y),
             (cx + c.width, title_y), (cx, title_y)],
            close=True,
            dxfattribs={"layer": "TABLE"},
        )
        msp.add_text(
            c.title,
            dxfattribs={"layer": "TABLE", "height": HEADER_H,
                        "style": "STRUCTAI"},
        ).set_placement(
            (cx + c.width / 2.0, title_y - ROW_H / 2.0),
            align=TA.MIDDLE_CENTER,
        )
        cx += c.width

    # Veri satırları
    ry = header_y
    for row in rows:
        next_ry = ry - ROW_H
        cx = ox
        for c, cell in zip(cols, row, strict=False):
            msp.add_lwpolyline(
                [(cx, next_ry), (cx + c.width, next_ry),
                 (cx + c.width, ry), (cx, ry)],
                close=True,
                dxfattribs={"layer": "TABLE"},
            )
            msp.add_text(
                cell,
                dxfattribs={"layer": "TABLE", "height": CELL_H,
                            "style": "STRUCTAI"},
            ).set_placement(
                (cx + c.width / 2.0, ry - ROW_H / 2.0),
                align=TA.MIDDLE_CENTER,
            )
            cx += c.width
        ry = next_ry

    # Tablo dışı 1px frame
    bot = oy - ROW_H * (2 + len(rows))
    msp.add_lwpolyline(
        [(ox, bot), (ox + total_w, bot),
         (ox + total_w, oy), (ox, oy)],
        close=True,
        dxfattribs={"layer": "TABLE"},
    )
    return ROW_H * (2 + len(rows))
