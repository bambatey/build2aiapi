"""Tekrarlanan semboller için DXF blok tanımları.

Bloklar `INSERT` (block reference) ile çağrılır — dosya boyutu küçük,
operatör DXF içinde sembolü ortak düzenleyebilir.
"""

from __future__ import annotations

from ezdxf.document import Drawing


GRID_BUBBLE = "GRID_BUBBLE"
SECTION_MARK = "SECTION_MARK"
ELEV_MARK = "ELEV_MARK"


def setup_blocks(doc: Drawing) -> None:
    """Tüm tekrar blokları doc.blocks'a kur."""
    if GRID_BUBBLE not in doc.blocks:
        _make_grid_bubble(doc)
    if SECTION_MARK not in doc.blocks:
        _make_section_mark(doc)
    if ELEV_MARK not in doc.blocks:
        _make_elev_mark(doc)


def _make_grid_bubble(doc: Drawing) -> None:
    """Aks kabarcığı: r=0.40m çember, içine etiket başka bir entity olarak konur.

    Etiket bloğun parçası DEĞİL — insert edildikten sonra etiket text'i
    üstüne add_text ile yazılır (her aks farklı harf/sayı).
    """
    blk = doc.blocks.new(name=GRID_BUBBLE)
    blk.add_circle(center=(0, 0), radius=0.40, dxfattribs={"layer": "GRID-BUBBLE"})


def _make_section_mark(doc: Drawing) -> None:
    """Kesit işareti — kalın çizgi + ucunda ok, harf sembolü.

    Görsel: ───►A  (kesit hattı + ucundaki ok ve etiket).
    """
    blk = doc.blocks.new(name=SECTION_MARK)
    # Ok ucu: 30cm uzunluk, 10cm genişlik
    blk.add_solid(
        points=[(0, 0), (-0.30, 0.10), (-0.30, -0.10), (0, 0)],
        dxfattribs={"layer": "SECTION-MARK"},
    )


def _make_elev_mark(doc: Drawing) -> None:
    """Kot işareti — yukarı bakan üçgen (0.4m × 0.2m).

    Kot değeri (örn +3.00) blok dışında ayrı text ile yazılır.
    """
    blk = doc.blocks.new(name=ELEV_MARK)
    blk.add_lwpolyline(
        [(-0.20, 0.0), (0.0, -0.20), (0.20, 0.0), (-0.20, 0.0)],
        dxfattribs={"layer": "ELEV-MARK"},
        close=True,
    )
    # İç tarama yerine hatch: solid fill
    hatch = blk.add_hatch(color=7, dxfattribs={"layer": "ELEV-MARK"})
    hatch.paths.add_polyline_path(
        [(-0.20, 0.0), (0.0, -0.20), (0.20, 0.0), (-0.20, 0.0)],
        is_closed=True,
    )
