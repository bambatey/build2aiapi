"""Model → çizim geometrisi dönüşümleri.

- Story (kat) gruplama: düğümlerin Z koordinatından
- Element sınıflandırması: kolon (düşey) / kiriş (yatay) / brace (eğik)
- Aks (grid) türetimi: kolon XY pozisyonlarından unique x,y dizileri
- Kolon plan poligonu: t2 × t3 dikdörtgen, lokal eksen rotasyonu ile

ConcCol/ConcBeam flag'lerine güvenmiyoruz — kullanıcının SAP modelinde
yanlış işaretli olabiliyor. Geometrik orientasyon tek doğru kaynak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..structural_analysis.model.dto import (
    FrameElementDTO,
    FrameSectionDTO,
    ModelDTO,
    ShellElementDTO,
    ShellSectionDTO,
)


# Tolerans değerleri (metre). SAP2000 modelinde küçük numerik gürültü olabilir.
Z_CLUSTER_TOL = 0.30        # Aynı kata ait kabul edilen Z yüksekliği farkı
GRID_CLUSTER_TOL = 0.20     # Aynı aksta sayılan XY pozisyon farkı
COLUMN_DZ_RATIO = 0.7       # |dz|/L > bu oran → kolon
BEAM_DZ_RATIO = 0.2          # |dz|/L < bu oran → kiriş


# --------------------------------------------------------------- veri yapıları
@dataclass(frozen=True)
class StoryDef:
    """Bir kat — base/top Z aralığı + etiket.

    base_z = döşemenin/zemin'in olduğu Z (kolon altı).
    top_z = bu katın döşemesi (kolon üstü).
    Etiket TR konvansiyonu: Zemin, 1, 2, 3, ...
    """
    index: int                      # 0 = en alt kat (genelde zemin)
    base_z: float                   # m
    top_z: float                    # m
    label: str                      # "Zemin Kat", "1. Kat", ...

    @property
    def height(self) -> float:
        return self.top_z - self.base_z


@dataclass
class ColumnGeom:
    """Plan-view'da çizilecek tek kolon."""
    section_name: str
    x: float
    y: float
    t2: float                       # plan-X genişlik (m)
    t3: float                       # plan-Y derinlik (m)
    rotation_deg: float = 0.0       # lokal eksen dönüşü
    label: str = ""                 # S1, S2, ...
    element_ids: list[int] = field(default_factory=list)


@dataclass
class BeamGeom:
    """Plan-view'da çizilecek tek kiriş."""
    section_name: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    width: float                    # b (m) — plan'da kiriş eninin değeri
    label: str = ""                 # K1, K2, ...
    element_ids: list[int] = field(default_factory=list)


@dataclass
class SlabGeom:
    """Plan-view'da çizilecek tek döşeme paneli."""
    section_name: str
    polygon: list[tuple[float, float]]
    thickness: float                # m
    label: str = ""                 # D1, D2, ...
    element_ids: list[int] = field(default_factory=list)


@dataclass
class GridAxes:
    """Akslar — X aksları sol→sağ, Y aksları alt→üst."""
    x_positions: list[float]        # m, sırasıyla A, B, C...
    y_positions: list[float]        # m, sırasıyla 1, 2, 3...
    x_labels: list[str]
    y_labels: list[str]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) — akslar+marj yok."""
        return (
            min(self.x_positions, default=0.0),
            min(self.y_positions, default=0.0),
            max(self.x_positions, default=0.0),
            max(self.y_positions, default=0.0),
        )


@dataclass
class StoryGeom:
    """Bir kata ait tüm plan-view geometrisi — çizicilerin tek girdisi."""
    story: StoryDef
    columns: list[ColumnGeom]
    beams: list[BeamGeom]
    slabs: list[SlabGeom]
    grid: GridAxes


# ----------------------------------------------------------- story gruplaması
def detect_stories(model: ModelDTO) -> list[StoryDef]:
    """Düğüm Z koordinatlarından kat seviyelerini türet.

    Algoritma: tüm node z'leri tolerans ile cluster → kat seviyeleri.
    İki kat seviyesi arası bir "kat" (zemin = altta).
    """
    z_values = sorted({_round(n.z, Z_CLUSTER_TOL) for n in model.nodes.values()})
    if len(z_values) < 2:
        # Tek seviye varsa fictitious bir kat oluştur (kalıp planı hala üretilir).
        z = z_values[0] if z_values else 0.0
        return [StoryDef(index=0, base_z=z, top_z=z, label="Zemin Kat")]

    stories: list[StoryDef] = []
    for i in range(len(z_values) - 1):
        base, top = z_values[i], z_values[i + 1]
        label = _story_label(i, len(z_values) - 1)
        stories.append(StoryDef(index=i, base_z=base, top_z=top, label=label))
    return stories


def _story_label(index: int, total: int) -> str:
    """0 → 'Zemin Kat', 1 → '1. Kat', ..."""
    if index == 0:
        return "Zemin Kat"
    return f"{index}. Kat"


def _round(value: float, tol: float) -> float:
    """Tolerans-snap: tol = 0.30 ise 0.13 → 0, 3.61 → 3.60."""
    return round(value / tol) * tol


# ---------------------------------------------- element classification helpers
def classify_frame(model: ModelDTO, fid: int) -> str | None:
    """'column' | 'beam' | 'brace' | None.

    Geometrik kural: |dz| / L oranı.
    """
    el = model.frame_elements.get(fid)
    if el is None or len(el.nodes) < 2:
        return None
    n1, n2 = model.nodes.get(el.nodes[0]), model.nodes.get(el.nodes[1])
    if n1 is None or n2 is None:
        return None
    dx, dy, dz = n2.x - n1.x, n2.y - n1.y, n2.z - n1.z
    L = math.sqrt(dx * dx + dy * dy + dz * dz)
    if L < 1e-6:
        return None
    ratio = abs(dz) / L
    if ratio > COLUMN_DZ_RATIO:
        return "column"
    if ratio < BEAM_DZ_RATIO:
        return "beam"
    return "brace"


# ---------------------------------------------- kolon plan poligonu çıkar
def build_columns_at_story(
    model: ModelDTO, story: StoryDef
) -> list[ColumnGeom]:
    """Bu kata ait kolonları topla — base_z'den geçen düşey frame'ler.

    Aynı XY'de farklı kolon kesitleri olabilir (alt kat 70x80, üst 50x60).
    Plan'da kolonun bu kattaki kesiti çizilir.
    """
    cols: list[ColumnGeom] = []
    # Group XY → list of (element, section, top_z)
    bucket: dict[tuple[float, float], list[tuple[int, FrameSectionDTO]]] = {}

    for fid, el in model.frame_elements.items():
        if not isinstance(el, FrameElementDTO):
            continue
        if classify_frame(model, fid) != "column":
            continue
        n1 = model.nodes.get(el.nodes[0])
        n2 = model.nodes.get(el.nodes[1])
        if n1 is None or n2 is None:
            continue
        z_low, z_high = sorted([n1.z, n2.z])
        # Bu kolon bu katı kesiyor mu?
        if not (z_low <= story.base_z + Z_CLUSTER_TOL and z_high >= story.top_z - Z_CLUSTER_TOL):
            continue
        sec = model.sections.get(el.section_id)
        if not isinstance(sec, FrameSectionDTO):
            continue
        # XY pozisyon — kolonun XY ortası
        x = (n1.x + n2.x) / 2.0
        y = (n1.y + n2.y) / 2.0
        key = (_round(x, GRID_CLUSTER_TOL), _round(y, GRID_CLUSTER_TOL))
        bucket.setdefault(key, []).append((fid, sec))

    # Her XY için en büyük kesitli kolonu temsilci seç (üst üste binme önlenir).
    for label_index, (key, entries) in enumerate(sorted(bucket.items()), start=1):
        # XY median (snap'in geri çevrimi yerine ham koordinat)
        # Doğru XY için ilk element'in ortasını kullan.
        fid0, _ = entries[0]
        el0 = model.frame_elements[fid0]
        n1 = model.nodes[el0.nodes[0]]
        n2 = model.nodes[el0.nodes[1]]
        x_real = (n1.x + n2.x) / 2.0
        y_real = (n1.y + n2.y) / 2.0
        # En büyük A'lı kesiti temsilci seç
        sec = max((s for _, s in entries), key=lambda s: s.A)
        # Rectangular değilse ya da t2/t3 yoksa A'dan b=h=sqrt(A) tahmin
        t2, t3 = sec.t2, sec.t3
        if t2 <= 0 or t3 <= 0:
            t2 = t3 = math.sqrt(max(sec.A, 1e-6))
        cols.append(ColumnGeom(
            section_name=sec.id,
            x=x_real,
            y=y_real,
            t2=t2,
            t3=t3,
            rotation_deg=sum(el.local_axis_angle for el in (model.frame_elements[e[0]] for e in entries)) / len(entries),
            label=f"S{label_index}",
            element_ids=[fid for fid, _ in entries],
        ))
    return cols


# ---------------------------------------------- kiriş plan çizgisi çıkar
def build_beams_at_story(
    model: ModelDTO, story: StoryDef
) -> list[BeamGeom]:
    """Bu kata ait kirişleri topla — top_z seviyesindeki yatay frame'ler."""
    beams: list[BeamGeom] = []
    label_index = 0
    seen: set[tuple[tuple[float, float], tuple[float, float], str]] = set()

    for fid, el in model.frame_elements.items():
        if not isinstance(el, FrameElementDTO):
            continue
        if classify_frame(model, fid) != "beam":
            continue
        n1 = model.nodes.get(el.nodes[0])
        n2 = model.nodes.get(el.nodes[1])
        if n1 is None or n2 is None:
            continue
        z_avg = (n1.z + n2.z) / 2.0
        # Bu kat'ın top_z seviyesindeki kiriş — döşeme altına asılan.
        if abs(z_avg - story.top_z) > Z_CLUSTER_TOL:
            continue
        sec = model.sections.get(el.section_id)
        if not isinstance(sec, FrameSectionDTO):
            continue
        # Kiriş genişliği = t2 (SAP'ta lokal 2 ekseni — plan'da b). t3 = h (depth).
        width = sec.t2 if sec.t2 > 0 else math.sqrt(max(sec.A, 1e-6))
        p1 = (round(n1.x, 3), round(n1.y, 3))
        p2 = (round(n2.x, 3), round(n2.y, 3))
        # Aynı kiriş çift kez çizilmesin (id seti'nde tut)
        key = (min(p1, p2), max(p1, p2), sec.id)
        if key in seen:
            continue
        seen.add(key)
        label_index += 1
        beams.append(BeamGeom(
            section_name=sec.id,
            p1=p1,
            p2=p2,
            width=width,
            label=f"K{label_index}",
            element_ids=[fid],
        ))
    return beams


# ---------------------------------------------- döşeme plan çıkar
def build_slabs_at_story(
    model: ModelDTO, story: StoryDef
) -> list[SlabGeom]:
    """Bu katın döşemesi — top_z seviyesindeki shell elementler."""
    slabs: list[SlabGeom] = []
    label_index = 0
    for sid, el in model.shell_elements.items():
        if not isinstance(el, ShellElementDTO):
            continue
        nodes = [model.nodes.get(nid) for nid in el.nodes]
        if any(n is None for n in nodes):
            continue
        z_avg = sum(n.z for n in nodes) / len(nodes)
        if abs(z_avg - story.top_z) > Z_CLUSTER_TOL:
            continue
        sec = model.sections.get(el.section_id)
        thickness = sec.thickness if isinstance(sec, ShellSectionDTO) else 0.0
        polygon = [(round(n.x, 3), round(n.y, 3)) for n in nodes]
        label_index += 1
        slabs.append(SlabGeom(
            section_name=el.section_id or "",
            polygon=polygon,
            thickness=thickness,
            label=f"D{label_index}",
            element_ids=[sid],
        ))
    return slabs


# ------------------------------------------------- aks (grid) türetimi
def derive_grid(columns: list[ColumnGeom]) -> GridAxes:
    """Kolon XY pozisyonlarından unique aksları çıkar.

    X aksları → A, B, C, ...
    Y aksları → 1, 2, 3, ...
    GRID_CLUSTER_TOL ile aynı aksa düşen kolonlar birleştirilir.
    """
    xs = _cluster_positions([c.x for c in columns])
    ys = _cluster_positions([c.y for c in columns])
    x_labels = [_alpha_label(i) for i in range(len(xs))]
    y_labels = [str(i + 1) for i in range(len(ys))]
    return GridAxes(
        x_positions=xs,
        y_positions=ys,
        x_labels=x_labels,
        y_labels=y_labels,
    )


def _cluster_positions(values: list[float]) -> list[float]:
    """Yakın değerleri grupla ve gruptan median al."""
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters: list[list[float]] = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= GRID_CLUSTER_TOL:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _alpha_label(index: int) -> str:
    """0 → 'A', 25 → 'Z', 26 → 'AA', ..."""
    label = ""
    n = index
    while True:
        label = chr(ord("A") + n % 26) + label
        n = n // 26 - 1
        if n < 0:
            break
    return label


# ---------------------------------- top-level: kat geometrisinin tamamı
def build_story_geom(model: ModelDTO, story: StoryDef) -> StoryGeom:
    cols = build_columns_at_story(model, story)
    beams = build_beams_at_story(model, story)
    slabs = build_slabs_at_story(model, story)
    grid = derive_grid(cols)
    return StoryGeom(story=story, columns=cols, beams=beams, slabs=slabs, grid=grid)
