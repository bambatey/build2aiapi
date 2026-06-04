"""Top-level DXF exporter — ModelDTO + ProjectInfo → DXF bytes.

Tek DXF dosyası, modelspace içinde her kat için bir "sayfa" bloğu:
  pafta = kat planı (sol) + eleman tablosu (sağ) + antet (sağ alt)

Sayfalar düşey eksende üst üste yerleştirilir (gap = 5m). Operatör DXF'i
açtığında her sayfayı seçip ayrı kağıda atayabilir; ilerideki sürümlerde
paperspace LAYOUT'ları otomatik üretilecek.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import ezdxf

from ..structural_analysis.model.dto import ModelDTO
from .blocks import setup_blocks
from .geometry import build_story_geom, detect_stories
from .plan_view import draw_plan
from .project_info import ProjectInfo
from .schedule_table import draw_schedule
from .styles import DIM_STYLE, new_document
from .title_block import ANTET_H, ANTET_W, draw_title_block


PAGE_GAP = 5.0          # m, ardışık paftalar arası boşluk (Y ekseninde)
TABLE_OFFSET_X = 3.0    # m, plan ile tablo arası boşluk
ANTET_OFFSET_Y = 1.5    # m, tablo ile antet arası boşluk


@dataclass(frozen=True)
class DxfExportResult:
    dxf_bytes: bytes
    sheet_count: int
    bbox: tuple[float, float, float, float]


def export_model_to_dxf(
    model: ModelDTO,
    info: ProjectInfo | None = None,
) -> DxfExportResult:
    """Çoklu kat kalıp planı DXF üret.

    Sayfa sayısı = detected story sayısı. Boş bir model gelirse bile DXF
    geçerli olur (bir uyarı sayfasıyla).
    """
    info = info or ProjectInfo()

    doc = new_document()
    setup_blocks(doc)
    msp = doc.modelspace()

    stories = detect_stories(model)
    sheet_count = len(stories)

    current_y = 0.0
    global_bbox = [float("inf"), float("inf"), float("-inf"), float("-inf")]

    for i, story in enumerate(stories, start=1):
        geom = build_story_geom(model, story)

        # 1) Plan view — sol-alt köşesi (0, current_y)
        plan_bbox = draw_plan(
            msp, geom, origin=(0.0, current_y), dimstyle=DIM_STYLE,
        )
        plan_w = plan_bbox[2] - plan_bbox[0]
        plan_h = plan_bbox[3] - plan_bbox[1]

        # 2) Eleman tablosu — plan'ın sağında, üst hizalı
        table_x = plan_bbox[2] + TABLE_OFFSET_X
        table_y = plan_bbox[3]
        table_bbox = draw_schedule(
            msp, geom, origin=(table_x, table_y),
        )
        table_h = table_bbox[3] - table_bbox[1]

        # 3) Antet — tablonun altına
        antet_x = table_x
        antet_y = table_bbox[1] - ANTET_OFFSET_Y - ANTET_H
        title_bbox = draw_title_block(
            msp, info,
            origin=(antet_x, antet_y),
            story_label=story.label,
            sheet_no=str(i),
            total_sheets=str(sheet_count),
        )

        # Sayfa frame'i (operatör için pafta sınırı görünür olsun)
        sheet_bbox = (
            plan_bbox[0] - 0.5,
            min(plan_bbox[1], title_bbox[1]) - 0.5,
            max(plan_bbox[2], table_bbox[2], title_bbox[2]) + 0.5,
            plan_bbox[3] + 0.5,
        )
        msp.add_lwpolyline(
            [(sheet_bbox[0], sheet_bbox[1]), (sheet_bbox[2], sheet_bbox[1]),
             (sheet_bbox[2], sheet_bbox[3]), (sheet_bbox[0], sheet_bbox[3])],
            close=True,
            dxfattribs={"layer": "TITLE-BLOCK"},
        )

        sheet_h = sheet_bbox[3] - sheet_bbox[1]
        current_y = sheet_bbox[1] - PAGE_GAP

        # Global bbox güncelle
        global_bbox[0] = min(global_bbox[0], sheet_bbox[0])
        global_bbox[1] = min(global_bbox[1], sheet_bbox[1])
        global_bbox[2] = max(global_bbox[2], sheet_bbox[2])
        global_bbox[3] = max(global_bbox[3], sheet_bbox[3])

    # 4) DXF byte olarak yaz (ezdxf write_str)
    buffer = io.StringIO()
    doc.write(buffer)
    dxf_str = buffer.getvalue()
    return DxfExportResult(
        dxf_bytes=dxf_str.encode("utf-8"),
        sheet_count=sheet_count,
        bbox=tuple(global_bbox),
    )
