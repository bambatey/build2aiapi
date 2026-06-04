"""Per-story kalıp planı çizici.

Girdi: StoryGeom + offset (modelspace içinde nereye yerleşecek).
Çıktı: msp'ye eklenmiş entity'ler (akslar, bubble, kolonlar, kirişler,
ölçü çizgileri, kat başlığı, kot işareti).

Çizimler metre cinsinden, ölçek 1/50 hedefli (paperspace viewport ile).
"""

from __future__ import annotations

import math

from ezdxf.enums import TextEntityAlignment as TA
from ezdxf.layouts import Modelspace

from .blocks import GRID_BUBBLE
from .geometry import BeamGeom, ColumnGeom, GridAxes, StoryGeom

# Plan görünüm marjları (m) — aksların kenardan ne kadar dışarı taşacağı.
GRID_BUBBLE_OFFSET = 1.20         # Aks bubble'ı bbox'tan bu kadar dışarıda
GRID_EXTEND = 0.80                 # Aks çizgisi bbox'tan bu kadar uzar
DIM_OFFSET = 2.20                  # Ölçü çizgisinin bbox'tan mesafesi


def draw_plan(
    msp: Modelspace,
    story: StoryGeom,
    origin: tuple[float, float],
    dimstyle: str,
) -> tuple[float, float, float, float]:
    """Bu kat için tüm plan-view entity'lerini çiz.

    `origin` modelspace içinde plan'ın sol-alt köşesini belirler.
    Return: (x_min, y_min, x_max, y_max) — plan + marj dahil bbox.
    """
    ox, oy = origin
    # Plan koordinatları offset'lenmiş; helper tüm noktaları çevirir.
    def t(x: float, y: float) -> tuple[float, float]:
        return (x + ox, y + oy)

    grid = story.grid

    # Plan iç bbox (akslara göre)
    if not grid.x_positions or not grid.y_positions:
        gx_min = gy_min = gx_max = gy_max = 0.0
    else:
        gx_min, gx_max = min(grid.x_positions), max(grid.x_positions)
        gy_min, gy_max = min(grid.y_positions), max(grid.y_positions)

    # 1) Akslar
    _draw_axes(msp, grid, ox, oy, gx_min, gx_max, gy_min, gy_max)

    # 2) Döşemeler (önce, kolonlar üstünde kalsın)
    for slab in story.slabs:
        _draw_slab(msp, slab, t)

    # 3) Kirişler
    for beam in story.beams:
        _draw_beam(msp, beam, t)

    # 4) Kolonlar (en üst — yüksek görünürlük)
    for col in story.columns:
        _draw_column(msp, col, t)

    # 5) Otomatik ölçüler — aks aralıkları
    _draw_axis_dimensions(msp, grid, ox, oy, dimstyle,
                          gx_min, gx_max, gy_min, gy_max)

    # 6) Kat başlığı + kot
    title_x, title_y = t(gx_min, gy_max + DIM_OFFSET + 1.40)
    msp.add_text(
        f"{story.story.label.upper()} KALIP PLANI",
        dxfattribs={"layer": "TEXT", "height": 0.45, "style": "STRUCTAI"},
    ).set_placement((title_x, title_y))
    msp.add_text(
        f"Kot: +{story.story.top_z:.2f}",
        dxfattribs={"layer": "TEXT", "height": 0.28, "style": "STRUCTAI"},
    ).set_placement((title_x, title_y - 0.65))

    # 7) Final bbox (kotaj çizgileri dahil)
    margin = DIM_OFFSET + GRID_BUBBLE_OFFSET + 1.0
    x_min = ox + gx_min - margin
    x_max = ox + gx_max + margin
    y_min = oy + gy_min - margin
    y_max = oy + gy_max + margin + 2.5     # title üst marjı
    return (x_min, y_min, x_max, y_max)


# ----------------------------------------------------------------- alt çizici
def _draw_axes(
    msp: Modelspace,
    grid: GridAxes,
    ox: float, oy: float,
    gx_min: float, gx_max: float,
    gy_min: float, gy_max: float,
) -> None:
    """Dikey X aksları + yatay Y aksları + bubble'lar + harf/sayı etiketleri."""
    # X aksları (düşey çizgiler)
    for xp, label in zip(grid.x_positions, grid.x_labels, strict=False):
        msp.add_line(
            (ox + xp, oy + gy_min - GRID_EXTEND),
            (ox + xp, oy + gy_max + GRID_EXTEND),
            dxfattribs={"layer": "GRID"},
        )
        # Üst bubble
        bx, by = ox + xp, oy + gy_max + GRID_EXTEND + GRID_BUBBLE_OFFSET
        msp.add_blockref(GRID_BUBBLE, insert=(bx, by))
        msp.add_text(
            label,
            dxfattribs={"layer": "GRID-TEXT", "height": 0.35, "style": "STRUCTAI"},
        ).set_placement((bx, by), align=TA.MIDDLE_CENTER)
        # Alt bubble
        bx2, by2 = ox + xp, oy + gy_min - GRID_EXTEND - GRID_BUBBLE_OFFSET
        msp.add_blockref(GRID_BUBBLE, insert=(bx2, by2))
        msp.add_text(
            label,
            dxfattribs={"layer": "GRID-TEXT", "height": 0.35, "style": "STRUCTAI"},
        ).set_placement((bx2, by2), align=TA.MIDDLE_CENTER)

    # Y aksları (yatay çizgiler)
    for yp, label in zip(grid.y_positions, grid.y_labels, strict=False):
        msp.add_line(
            (ox + gx_min - GRID_EXTEND, oy + yp),
            (ox + gx_max + GRID_EXTEND, oy + yp),
            dxfattribs={"layer": "GRID"},
        )
        # Sol bubble
        bx, by = ox + gx_min - GRID_EXTEND - GRID_BUBBLE_OFFSET, oy + yp
        msp.add_blockref(GRID_BUBBLE, insert=(bx, by))
        msp.add_text(
            label,
            dxfattribs={"layer": "GRID-TEXT", "height": 0.35, "style": "STRUCTAI"},
        ).set_placement((bx, by), align=TA.MIDDLE_CENTER)
        # Sağ bubble
        bx2, by2 = ox + gx_max + GRID_EXTEND + GRID_BUBBLE_OFFSET, oy + yp
        msp.add_blockref(GRID_BUBBLE, insert=(bx2, by2))
        msp.add_text(
            label,
            dxfattribs={"layer": "GRID-TEXT", "height": 0.35, "style": "STRUCTAI"},
        ).set_placement((bx2, by2), align=TA.MIDDLE_CENTER)


def _draw_column(
    msp: Modelspace, col: ColumnGeom, t,
) -> None:
    """Kolon — t2×t3 dikdörtgen, merkez (col.x, col.y), local angle rotasyonu."""
    cx, cy = t(col.x, col.y)
    half_w, half_d = col.t2 / 2.0, col.t3 / 2.0
    # Lokal eksen rotasyonu (radyan)
    theta = math.radians(col.rotation_deg or 0.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = [
        (-half_w, -half_d), (half_w, -half_d),
        (half_w, half_d), (-half_w, half_d),
    ]
    rotated = [(cx + cos_t * px - sin_t * py, cy + sin_t * px + cos_t * py)
               for px, py in corners]
    msp.add_lwpolyline(
        rotated, close=True,
        dxfattribs={"layer": "COLUMN"},
    )
    # İç tarama (kolon dolu görünsün)
    hatch = msp.add_hatch(color=1, dxfattribs={"layer": "COLUMN-HATCH"})
    hatch.set_pattern_fill("ANSI31", scale=0.05)
    hatch.paths.add_polyline_path(rotated, is_closed=True)
    # Etiket: S1: 70×80
    label_text = f"{col.label}: {int(round(col.t2 * 100))}×{int(round(col.t3 * 100))}"
    msp.add_text(
        label_text,
        dxfattribs={"layer": "TEXT", "height": 0.18, "style": "STRUCTAI"},
    ).set_placement(
        (cx, cy - half_d - 0.30), align=TA.MIDDLE_CENTER,
    )


def _draw_beam(msp: Modelspace, beam: BeamGeom, t) -> None:
    """Kiriş — centerline + iki kenar çizgisi (genişlik=b)."""
    (x1, y1), (x2, y2) = t(*beam.p1), t(*beam.p2)
    # Centerline (kesik çizgi tercih edilir ama Continuous kalıyor — operatör değiştirebilir)
    msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": "BEAM"})
    # Kenar çizgiler
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    nx, ny = -dy / L, dx / L                 # birim normal vektör
    off = beam.width / 2.0
    msp.add_line(
        (x1 + nx * off, y1 + ny * off),
        (x2 + nx * off, y2 + ny * off),
        dxfattribs={"layer": "BEAM-EDGE"},
    )
    msp.add_line(
        (x1 - nx * off, y1 - ny * off),
        (x2 - nx * off, y2 - ny * off),
        dxfattribs={"layer": "BEAM-EDGE"},
    )
    # Etiket — orta nokta, kiriş paralelinde
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    angle_deg = math.degrees(math.atan2(dy, dx))
    if angle_deg > 90 or angle_deg < -90:
        angle_deg += 180.0                   # metin baş aşağı olmasın
    label_text = f"{beam.label} ({int(round(beam.width * 100))}×?)"
    # Width × depth yazmak için sec.id'yi kullansak daha iyi olur ama
    # current dto sadece width tutuyor — beam height parser'da yok.
    msp.add_text(
        beam.label,
        dxfattribs={"layer": "TEXT", "height": 0.16, "rotation": angle_deg,
                    "style": "STRUCTAI"},
    ).set_placement((mx, my + 0.12), align=TA.MIDDLE_CENTER)


def _draw_slab(msp: Modelspace, slab, t) -> None:
    """Döşeme — poligon kenarı + light hatch."""
    pts = [t(x, y) for x, y in slab.polygon]
    if len(pts) < 3:
        return
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "SLAB"})
    # Hatch (hafif, line pattern)
    try:
        hatch = msp.add_hatch(color=8, dxfattribs={"layer": "SLAB-HATCH"})
        hatch.set_pattern_fill("ANSI31", scale=0.20)
        hatch.paths.add_polyline_path(pts, is_closed=True)
    except Exception:
        # Pattern desteklenmiyorsa atlanır.
        pass
    # Etiket merkeze
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    label_text = f"{slab.label} t={int(round(slab.thickness * 100))}cm"
    msp.add_text(
        label_text,
        dxfattribs={"layer": "TEXT", "height": 0.14, "style": "STRUCTAI"},
    ).set_placement((cx, cy), align=TA.MIDDLE_CENTER)


def _draw_axis_dimensions(
    msp: Modelspace,
    grid: GridAxes,
    ox: float, oy: float,
    dimstyle: str,
    gx_min: float, gx_max: float,
    gy_min: float, gy_max: float,
) -> None:
    """Akslar arası otomatik ölçü çizgisi — alt ve sol kenarda."""
    # Alt yatay ölçü zinciri (X aksları arası)
    base_y = oy + gy_min - DIM_OFFSET
    xs = sorted(grid.x_positions)
    for x1, x2 in zip(xs, xs[1:], strict=False):
        try:
            dim = msp.add_aligned_dim(
                p1=(ox + x1, oy + gy_min - GRID_EXTEND),
                p2=(ox + x2, oy + gy_min - GRID_EXTEND),
                distance=-(DIM_OFFSET - GRID_EXTEND),
                dimstyle=dimstyle,
                dxfattribs={"layer": "DIMENSION"},
            )
            dim.render()
        except Exception:
            # Bazı ezdxf versiyonlarında dim.render() farklı — sessiz geç.
            pass

    # Sol dikey ölçü zinciri (Y aksları arası)
    ys = sorted(grid.y_positions)
    for y1, y2 in zip(ys, ys[1:], strict=False):
        try:
            dim = msp.add_aligned_dim(
                p1=(ox + gx_min - GRID_EXTEND, oy + y1),
                p2=(ox + gx_min - GRID_EXTEND, oy + y2),
                distance=-(DIM_OFFSET - GRID_EXTEND),
                dimstyle=dimstyle,
                dxfattribs={"layer": "DIMENSION"},
            )
            dim.render()
        except Exception:
            pass
