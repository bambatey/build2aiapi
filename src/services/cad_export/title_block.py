"""TR statik proje antedi — İller Bankası tip şartnamesi referansı.

Antet model-space içinde çizilir. Paperspace layout ile her sayfaya ayrı
viewport koymak yerine pragmatik yaklaşım: her kat planının sağ-alt
köşesinde antet bloğu.

Boyutlar (m, ölçek 1/50 paperspace karşılığı parantezde):
  Antet kutusu: 12 × 7  (240×140 mm)
  Satır yüksekliği: 0.6  (12 mm)
"""

from __future__ import annotations

from ezdxf.enums import TextEntityAlignment as TA
from ezdxf.layouts import Modelspace

from .project_info import ProjectInfo

_ALIGN_MAP = {
    "BOTTOM_LEFT": TA.LEFT,
    "MIDDLE_CENTER": TA.MIDDLE_CENTER,
    "MIDDLE_RIGHT": TA.MIDDLE_RIGHT,
    "MIDDLE_LEFT": TA.MIDDLE_LEFT,
}

ANTET_W = 12.0          # m, modelspace
ANTET_H = 7.0
ROW_H = 0.60
TEXT_LARGE = 0.36
TEXT_MED = 0.22
TEXT_SMALL = 0.16


def draw_title_block(
    msp: Modelspace,
    info: ProjectInfo,
    origin: tuple[float, float],
    story_label: str,
    sheet_no: str,
    total_sheets: str,
) -> tuple[float, float, float, float]:
    """Antet çiz, kapladığı bbox'ı döndür."""
    ox, oy = origin

    # Dış çerçeve
    _rect(msp, ox, oy, ANTET_W, ANTET_H, layer="TITLE-BLOCK")

    # Üst başlık bandı
    _rect(msp, ox, oy + ANTET_H - 1.20, ANTET_W, 1.20, layer="TITLE-BLOCK")
    _text(msp, info.firm_name or "STRUCTAI",
          (ox + 0.30, oy + ANTET_H - 0.45), height=TEXT_LARGE, layer="TITLE-BLOCK")
    _text(msp, "STATİK BETONARME PROJE",
          (ox + 0.30, oy + ANTET_H - 0.95), height=TEXT_MED, layer="TITLE-BLOCK")

    # Sağ üst — sheet kind + numara
    _text(msp, info.sheet_kind,
          (ox + ANTET_W - 0.30, oy + ANTET_H - 0.45),
          height=TEXT_LARGE, layer="TITLE-BLOCK", align="MIDDLE_RIGHT")
    _text(msp, f"Pafta: {sheet_no} / {total_sheets}",
          (ox + ANTET_W - 0.30, oy + ANTET_H - 0.95),
          height=TEXT_SMALL, layer="TITLE-BLOCK", align="MIDDLE_RIGHT")

    # Lokasyon bandı
    _hline(msp, ox, oy + ANTET_H - 2.40, ANTET_W)
    y = oy + ANTET_H - 1.65
    _label(msp, "PROJE", info.project_name, ox + 0.30, y)
    y -= ROW_H
    _label(msp, "KAT", story_label, ox + 0.30, y)

    # Lokasyon
    _hline(msp, ox, oy + ANTET_H - 3.60, ANTET_W)
    y = oy + ANTET_H - 2.85
    _label(msp, "İL / İLÇE",
           f"{info.city} / {info.district}", ox + 0.30, y)
    y -= ROW_H
    _label(msp, "MAH / ADA / PARSEL",
           f"{info.neighborhood}  {info.ada} / {info.parsel}", ox + 0.30, y)

    # Mühendis
    _hline(msp, ox, oy + ANTET_H - 4.80, ANTET_W)
    y = oy + ANTET_H - 4.05
    _label(msp, "PROJE MÜHENDİSİ", info.engineer_name, ox + 0.30, y)
    y -= ROW_H
    _label(msp, "İMO SİCİL / İTB NO",
           f"{info.engineer_chamber_no} / {info.engineer_itb_no}",
           ox + 0.30, y)

    # Teknik parametreler
    _hline(msp, ox, oy + ANTET_H - 6.00, ANTET_W)
    y = oy + ANTET_H - 5.25
    _label(msp, "BETON / ÇELİK",
           f"{info.concrete_class} / {info.steel_class}",
           ox + 0.30, y)
    y -= ROW_H
    _label(msp,
           "I / R / Zemin",
           f"{info.building_importance} / {info.building_behavior_R} / {info.soil_class}",
           ox + 0.30, y)

    # Tarih + ölçek
    y = oy + 0.45
    _text(msp, f"TARİH: {info.drawing_date.strftime('%d.%m.%Y')}",
          (ox + 0.30, y), height=TEXT_MED, layer="TITLE-BLOCK")
    _text(msp, f"ÖLÇEK: {info.scale}",
          (ox + ANTET_W - 0.30, y),
          height=TEXT_MED, layer="TITLE-BLOCK", align="MIDDLE_RIGHT")
    _text(msp, info.sheet_size,
          (ox + ANTET_W / 2.0, y),
          height=TEXT_MED, layer="TITLE-BLOCK", align="MIDDLE_CENTER")

    return (ox, oy, ox + ANTET_W, oy + ANTET_H)


# ----------------------------------------------------------------- alt yardımcı
def _rect(msp: Modelspace, x: float, y: float, w: float, h: float,
          layer: str) -> None:
    msp.add_lwpolyline(
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        close=True,
        dxfattribs={"layer": layer},
    )


def _hline(msp: Modelspace, x: float, y: float, w: float) -> None:
    msp.add_line((x, y), (x + w, y), dxfattribs={"layer": "TITLE-BLOCK"})


def _text(msp: Modelspace, content: str, pos: tuple[float, float],
          height: float, layer: str, align: str = "BOTTOM_LEFT") -> None:
    if not content:
        return
    t = msp.add_text(
        content,
        dxfattribs={"layer": layer, "height": height, "style": "STRUCTAI"},
    )
    if align == "BOTTOM_LEFT":
        t.set_placement(pos)
    else:
        t.set_placement(pos, align=_ALIGN_MAP.get(align, TA.LEFT))


def _label(msp: Modelspace, key: str, value: str,
           x: float, y: float) -> None:
    """Antet içi 'KEY  : VALUE' satırı — sol etiket küçük, sağda büyük değer."""
    _text(msp, key, (x, y + 0.05), height=TEXT_SMALL, layer="TITLE-BLOCK")
    _text(msp, value or "—", (x + 3.5, y - 0.05),
          height=TEXT_MED, layer="TITLE-BLOCK")
