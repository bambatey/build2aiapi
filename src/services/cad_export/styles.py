"""Layer, text ve dimension stilleri.

TR statik proje konvansiyonu — `docs/architecture/06-cad-export.md §6.7` layer
tablosuna uygun. ACI renk kodları:
  1=Red, 2=Yellow, 3=Green, 4=Cyan, 5=Blue, 6=Magenta, 7=White, 8=DarkGray
"""

from __future__ import annotations

from dataclasses import dataclass

import ezdxf
from ezdxf.document import Drawing


# ---------------------------------------------------------------------- layers
@dataclass(frozen=True)
class LayerSpec:
    name: str
    color: int                  # ACI 1-255 (7=white, 0=ByBlock, 256=ByLayer)
    linetype: str = "Continuous"
    description: str = ""


LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec("GRID",         4,  "CENTER", "Aks çizgileri"),
    LayerSpec("GRID-BUBBLE",  4,  "Continuous", "Aks kabarcıkları"),
    LayerSpec("GRID-TEXT",    4,  "Continuous", "Aks etiketleri (A, B, 1, 2)"),
    LayerSpec("COLUMN",       1,  "Continuous", "Kolon poligonları (dolu çizgi)"),
    LayerSpec("COLUMN-HATCH", 1,  "Continuous", "Kolon iç tarama"),
    LayerSpec("BEAM",         3,  "Continuous", "Kiriş orta çizgisi"),
    LayerSpec("BEAM-EDGE",    3,  "Continuous", "Kiriş kenar (genişlik)"),
    LayerSpec("SLAB",         2,  "Continuous", "Döşeme kenarları"),
    LayerSpec("SLAB-HATCH",   8,  "Continuous", "Döşeme taraması (kalınlık ile)"),
    LayerSpec("DIMENSION",    5,  "Continuous", "Ölçü çizgileri ve metni"),
    LayerSpec("TEXT",         7,  "Continuous", "Genel metin / etiketler"),
    LayerSpec("ELEV-MARK",    7,  "Continuous", "Kot işareti (+0.00)"),
    LayerSpec("TITLE-BLOCK",  7,  "Continuous", "Antet çerçevesi ve metinleri"),
    LayerSpec("TABLE",        7,  "Continuous", "Eleman tablosu"),
    LayerSpec("SECTION-MARK", 6,  "Continuous", "Kesit işaretleri"),
)


# Text style — metric units, ölçek bağımsız (paperspace mm yerine
# modelspace metre kullanıyoruz, metin yüksekliklerini gerçek dünya cm).
TEXT_STYLE = "STRUCTAI"     # özel font tanımı (Arial veya benzeri)
DIM_STYLE = "STRUCTAI_DIM"


def setup_styles(doc: Drawing) -> None:
    """Doc'a layer, linetype, text ve dim stilleri kur."""
    _ensure_linetypes(doc)
    _ensure_layers(doc)
    _ensure_text_style(doc)
    _ensure_dim_style(doc)


def _ensure_linetypes(doc: Drawing) -> None:
    """Standart AutoCAD linetype'ları doc'a ekle.

    ezdxf base R2018 template'i Continuous + birkaç temel linetype içeriyor.
    Yine de garanti için kontrol et.
    """
    needed = ["Continuous", "CENTER", "DASHED", "HIDDEN"]
    existing = {lt.dxf.name.upper() for lt in doc.linetypes}
    for lt_name in needed:
        if lt_name.upper() in existing:
            continue
        # ezdxf yardımcısı: AutoCAD'in standart linetype tanımları
        try:
            doc.linetypes.add(
                name=lt_name,
                pattern=_LINETYPE_PATTERNS.get(lt_name, "A,1.0"),
                description=lt_name,
            )
        except Exception:
            # Linetype zaten varsa veya pattern syntax desteklenmiyorsa sessiz geç.
            pass


# AutoCAD acad.lin tipi pattern stringleri (basit yaklaşım).
_LINETYPE_PATTERNS = {
    "CENTER": "Center,____ _ ____ _ ____ _ ____ _ ____ _ ___",
    "DASHED": "Dashed,__ __ __ __ __ __ __ __ __ __ __ __ __ __",
    "HIDDEN": "Hidden,__ __ __ __ __ __ __ __ __ __ __ __ __ __",
}


def _ensure_layers(doc: Drawing) -> None:
    for spec in LAYERS:
        if spec.name in doc.layers:
            continue
        layer = doc.layers.add(spec.name)
        layer.color = spec.color
        try:
            layer.dxf.linetype = spec.linetype
        except Exception:
            layer.dxf.linetype = "Continuous"
        if spec.description:
            layer.description = spec.description


def _ensure_text_style(doc: Drawing) -> None:
    """STRUCTAI text style — sans-serif, Türkçe karakter destekli."""
    if TEXT_STYLE in doc.styles:
        return
    # ttf font dosyası rezolüsyonu — AutoCAD/BricsCAD sistemde arar.
    # arial.ttf TR karakterleri destekler.
    doc.styles.add(TEXT_STYLE, font="arial.ttf")


def _ensure_dim_style(doc: Drawing) -> None:
    """Otomatik ölçü için stil — metre cinsinden, 2 ondalık, cm okuma."""
    if DIM_STYLE in doc.dimstyles:
        return
    dimstyle = doc.dimstyles.add(DIM_STYLE)
    # Metre cinsinden model, ölçü 1.00 yerine 100 (cm) gösterilsin.
    dimstyle.dxf.dimlfac = 100.0    # linear scale (m → cm)
    dimstyle.dxf.dimdec = 0          # ondalık basamak yok (cm tam sayı)
    dimstyle.dxf.dimrnd = 0.0
    dimstyle.dxf.dimasz = 0.15       # ok ucu boyutu (m). 1/50'de ~3mm.
    dimstyle.dxf.dimtxt = 0.18       # metin yüksekliği (m). 1/50'de ~3.6mm.
    dimstyle.dxf.dimexe = 0.05       # uzatma çizgisi taşma
    dimstyle.dxf.dimexo = 0.10       # uzatma çizgisi başlangıç offset
    dimstyle.dxf.dimdli = 0.40       # ardışık ölçü çizgisi mesafesi
    dimstyle.dxf.dimtad = 1          # metin üstte (ölçü çizgisinin)
    dimstyle.dxf.dimgap = 0.06
    dimstyle.dxf.dimblk = "ARCHTICK"  # mimari tip ok ucu (eğik çizgi)
    dimstyle.dxf.dimblk1 = "ARCHTICK"
    dimstyle.dxf.dimblk2 = "ARCHTICK"
    try:
        dimstyle.dxf.dimtxsty = TEXT_STYLE
    except Exception:
        pass


def new_document() -> Drawing:
    """Yeni bir DXF dokümanı yarat ve tüm stilleri kur.

    Versiyon R2018 — AutoCAD 2018+ ve LibreCAD/BricsCAD/FreeCAD ile uyumlu.
    insunits=6 (metre) — analiz modelinin doğal birimi.
    """
    doc = ezdxf.new(dxfversion="R2018", setup=True)
    doc.header["$INSUNITS"] = 6              # metre
    doc.header["$MEASUREMENT"] = 1           # metric
    setup_styles(doc)
    return doc
