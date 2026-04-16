"""Parametrik betonarme çerçeve (RC Frame) .s2k üreticisi.

Kullanıcı "3x4 2 katlı" dediğinde ``RCFrameParams`` nesnesi oluşturulur
ve ``generate_rc_frame`` fonksiyonu tam bir SAP2000 .s2k dosyası üretir.

Üretilen dosya:
    - PROGRAM CONTROL (kN-m-C birim sistemi)
    - MATERIAL PROPERTIES (C30/37 beton varsayılan)
    - FRAME SECTION PROPERTIES (kare kolon + dikdörtgen kiriş)
    - JOINT COORDINATES (grid — temel + her kat kesişim noktası)
    - CONNECTIVITY - FRAME (kolonlar + X-yönü kirişler + Y-yönü kirişler)
    - FRAME SECTION ASSIGNMENTS (kolon/kiriş kesit ataması)
    - JOINT RESTRAINT ASSIGNMENTS (taban ankastre)
    - LOAD PATTERN DEFINITIONS (DEAD + LIVE)
    - FRAME LOADS - DISTRIBUTED (kirişlere yayılı DEAD + LIVE)
    - COMBINATION DEFINITIONS (1.4D + 1.6L)

Sonuç ``parseS2K`` ile parse edilebilir ve pipeline ile analiz edilebilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import Any

from .sections_rc import (
    FrameSection,
    concrete_default,
    rect_section,
    square_column,
)


# ---------------------------------------------------------------- parametreler
@dataclass
class RCFrameParams:
    """RC çerçeve üreticiye verilen parametreler (tümü opsiyonel)."""

    # --- grid (açıklık sayıları ve uzunlukları)
    bays_x: int = 3              # X yönünde açıklık sayısı
    bays_y: int = 4              # Y yönünde açıklık sayısı
    stories: int = 2             # Kat sayısı
    bay_dx: float = 6.0          # X açıklık uzunluğu (m)
    bay_dy: float = 5.0          # Y açıklık uzunluğu (m)
    story_h: float = 3.0         # Kat yüksekliği (m)

    # --- kesitler
    col_size: float = 0.40       # Kare kolon boyutu (m)  → 40x40
    beam_width: float = 0.30     # Kiriş genişliği (m)   → 30
    beam_height: float = 0.60    # Kiriş yüksekliği (m)  → 60
    # (İleride: farklı kat farklı kesit, farklı yönlerde farklı kiriş vb.)

    # --- malzeme
    fck_mpa: int = 30            # C30/37 varsayılan

    # --- yükler (her kiriş için kN/m, aşağı yönde)
    dead_q: float = 5.0          # G — sabit yük
    live_q: float = 2.0          # Q — hareketli yük

    # --- yardımcı bilgi (üretilen dosyanın başlık yorumuna yazılır)
    project_name: str = "Yeni Proje"
    notes: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "bays_x": self.bays_x, "bays_y": self.bays_y, "stories": self.stories,
            "bay_dx_m": self.bay_dx, "bay_dy_m": self.bay_dy, "story_h_m": self.story_h,
            "col_size_m": self.col_size,
            "beam_w_m": self.beam_width, "beam_h_m": self.beam_height,
            "fck_mpa": self.fck_mpa,
            "dead_q_knm": self.dead_q, "live_q_knm": self.live_q,
        }


# -------------------------------------------------------------------- grid helper
@dataclass
class _Node:
    id: int
    x: float
    y: float
    z: float


@dataclass
class _Frame:
    id: int
    i: int
    j: int
    kind: str   # "column" | "beam_x" | "beam_y"


def _build_geometry(p: RCFrameParams) -> tuple[list[_Node], list[_Frame]]:
    """Grid noktaları + kolon/kiriş eleman listesi üret.

    Koordinat sistemi: X-yan yan, Y-uzun yan, Z-yukarı. Taban z=0.
    Düğüm numaralandırma: kat×grid × (bays_y+1) × (bays_x+1).
    Kolonlar her düğümden bir üst kata, X-kirişleri bay_x doğrultusunda,
    Y-kirişleri bay_y doğrultusunda.
    """
    nx = p.bays_x + 1   # X yönü düğüm sayısı
    ny = p.bays_y + 1   # Y yönü düğüm sayısı
    nz = p.stories + 1  # Z yönü seviye sayısı (taban + her kat)

    nodes: list[_Node] = []
    # düğüm id = 1 + ((kz * ny) + iy) * nx + ix  → ama linear bir sayacı
    # kullanmak daha güvenli
    id_of: dict[tuple[int, int, int], int] = {}
    counter = 1
    for kz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                nodes.append(_Node(
                    id=counter,
                    x=ix * p.bay_dx,
                    y=iy * p.bay_dy,
                    z=kz * p.story_h,
                ))
                id_of[(ix, iy, kz)] = counter
                counter += 1

    frames: list[_Frame] = []
    fid = 1

    # Kolonlar: her grid noktasından bir üst kata
    for kz in range(nz - 1):
        for iy in range(ny):
            for ix in range(nx):
                a = id_of[(ix, iy, kz)]
                b = id_of[(ix, iy, kz + 1)]
                frames.append(_Frame(id=fid, i=a, j=b, kind="column"))
                fid += 1

    # X-yönü kirişler (bay_dx boyunca): sadece üst katlarda (kz >= 1)
    for kz in range(1, nz):
        for iy in range(ny):
            for ix in range(nx - 1):
                a = id_of[(ix, iy, kz)]
                b = id_of[(ix + 1, iy, kz)]
                frames.append(_Frame(id=fid, i=a, j=b, kind="beam_x"))
                fid += 1

    # Y-yönü kirişler (bay_dy boyunca): sadece üst katlarda
    for kz in range(1, nz):
        for iy in range(ny - 1):
            for ix in range(nx):
                a = id_of[(ix, iy, kz)]
                b = id_of[(ix, iy + 1, kz)]
                frames.append(_Frame(id=fid, i=a, j=b, kind="beam_y"))
                fid += 1

    return nodes, frames


def _base_joint_ids(p: RCFrameParams) -> list[int]:
    """Taban düğümlerinin id'leri (ankastre için)."""
    nx = p.bays_x + 1
    ny = p.bays_y + 1
    return list(range(1, nx * ny + 1))


# ----------------------------------------------------------- .s2k yazar yardımcıları
def _fmt_num(v: float, prec: int = 6) -> str:
    """SAP'ın sevdiği kısa ondalık biçim (trailing sıfırları kırp)."""
    s = f"{v:.{prec}f}"
    # Ondalıklı ise trailing 0'ları at, son "." kalırsa onu da at
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _kv(**kwargs: Any) -> str:
    """Key=Value string'i — değer tipine göre formatla."""
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, float):
            vs = _fmt_num(v)
        elif isinstance(v, bool):
            vs = "Yes" if v else "No"
        else:
            vs = str(v)
        # Değerde boşluk varsa tırnak içine al
        if " " in vs:
            vs = f'"{vs}"'
        parts.append(f"{k}={vs}")
    return "   " + "   ".join(parts)


def _table(out: StringIO, name: str, rows: list[str]) -> None:
    out.write(f'\nTABLE:  "{name}"\n')
    for row in rows:
        out.write(row + "\n")


# --------------------------------------------------------------- asıl üretici
def generate_rc_frame(params: RCFrameParams | None = None) -> str:
    """Parametrelere göre tam SAP2000 .s2k dosya içeriği üretir."""
    p = params or RCFrameParams()

    # -- malzeme
    concrete = concrete_default(p.fck_mpa)

    # -- kesitler
    col = square_column("COL", concrete.id, p.col_size)
    beam = rect_section("BEAM", concrete.id, p.beam_width, p.beam_height)

    # -- geometri
    nodes, frames = _build_geometry(p)

    out = StringIO()
    # Dosya başlık — bilgi amaçlı (parser yorum satırlarını yoksayar)
    out.write(f'$ File C:\\{p.project_name}.$2k saved {p.project_name}\n')
    out.write("$ PROGRAM:  SAP2000   VERSION:  24.0.0 (build2ai generator)\n")
    if p.notes:
        out.write(f"$ Notes: {p.notes}\n")

    # ================================================================ PROGRAM CONTROL
    _table(out, "PROGRAM CONTROL", [
        _kv(
            ProgramName="SAP2000", Version="24.0.0", ProgLevel="Ultimate",
            LicenseNum="1", LicenseOS="Yes", LicenseSC="Yes", LicenseBR="Yes",
            LicenseHT="No", CurrUnits="KN, m, C", SteelCode="AISC360-16",
            ConcCode="TS500-2000", AlumCode="AA-ASD 2000", ColdCode="AISI-ASD96",
            BridgeCode="AASHTO LRFD 2007", TimberCode="AF&PA NDS 2005",
            RegenHinge="Yes",
        )
    ])

    # =========================================================== MATERIAL PROPERTIES
    _table(out, "MATERIAL PROPERTIES 01 - GENERAL", [
        _kv(
            Material=concrete.id, Type="Concrete", Grade=f"f'c {p.fck_mpa} MPa",
            SymType="Isotropic", TempDepend="No", Color="Gray8Dark",
            Notes="Auto-generated (build2ai)",
        )
    ])
    _table(out, "MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES", [
        _kv(
            Material=concrete.id,
            UnitWeight=concrete.unit_weight,
            UnitMass=concrete.unit_mass,
            E1=concrete.E,
            U12=concrete.nu,
            A1=concrete.alpha,
        )
    ])

    # ============================================================= FRAME SECTIONS
    _section_rows = []
    for s in (col, beam):
        _section_rows.append(_kv(
            SectionName=s.id, Material=s.material_id,
            Shape="Rectangular", t3=s.t3, t2=s.t2,
            Area=s.A, TorsConst=s.J, I33=s.I33, I22=s.I22,
            AS2=s.A * (5.0 / 6.0), AS3=s.A * (5.0 / 6.0),
            S33=s.I33 / max(s.t3 / 2.0, 1e-9),
            S22=s.I22 / max(s.t2 / 2.0, 1e-9),
            Z33=s.t2 * (s.t3 ** 2) / 4.0,
            Z22=s.t3 * (s.t2 ** 2) / 4.0,
            R33=(s.I33 / max(s.A, 1e-9)) ** 0.5,
            R22=(s.I22 / max(s.A, 1e-9)) ** 0.5,
            ConcCol="Yes" if s.id == "COL" else "No",
            ConcBeam="No" if s.id == "COL" else "Yes",
            Color="Cyan",
            TotalWt=0.0, TotalMass=0.0,
            FromFile="No", AMod=1.0, A2Mod=1.0, A3Mod=1.0,
            JMod=1.0, I2Mod=1.0, I3Mod=1.0, MMod=1.0, WMod=1.0,
        ))
    _table(out, "FRAME SECTION PROPERTIES 01 - GENERAL", _section_rows)

    # =========================================================== JOINT COORDINATES
    _joint_rows = []
    for n in nodes:
        _joint_rows.append(_kv(
            Joint=n.id, CoordSys="GLOBAL", CoordType="Cartesian",
            XorR=n.x, Y=n.y, Z=n.z,
            SpecialJt="No", GlobalX=n.x, GlobalY=n.y, GlobalZ=n.z,
        ))
    _table(out, "JOINT COORDINATES", _joint_rows)

    # =========================================================== CONNECTIVITY FRAME
    _conn_rows = []
    for f in frames:
        _conn_rows.append(_kv(
            Frame=f.id, JointI=f.i, JointJ=f.j,
            IsCurved="No", Length=0.0, CentroidX=0.0, CentroidY=0.0, CentroidZ=0.0,
            GUID="", Notes="",
        ))
    _table(out, "CONNECTIVITY - FRAME", _conn_rows)

    # ====================================================== FRAME SECTION ASSIGNMENTS
    _assign_rows = []
    for f in frames:
        sec = "COL" if f.kind == "column" else "BEAM"
        _assign_rows.append(_kv(
            Frame=f.id, SectionType="Frame",
            AutoSelect="N.A.",
            AnalSect=sec, DesignSect=sec,
            MatProp="Default",
        ))
    _table(out, "FRAME SECTION ASSIGNMENTS", _assign_rows)

    # ======================================================= JOINT RESTRAINT ASSIGNMENTS
    _restraint_rows = []
    for nid in _base_joint_ids(p):
        _restraint_rows.append(_kv(
            Joint=nid, U1=True, U2=True, U3=True, R1=True, R2=True, R3=True,
        ))
    _table(out, "JOINT RESTRAINT ASSIGNMENTS", _restraint_rows)

    # =========================================================== LOAD PATTERNS
    _table(out, "LOAD PATTERN DEFINITIONS", [
        _kv(LoadPat="DEAD", DesignType="DEAD", SelfWtMult=1.0),
        _kv(LoadPat="LIVE", DesignType="LIVE", SelfWtMult=0.0),
    ])

    # =========================================================== FRAME LOADS - DIST
    # Kirişlere yayılı DEAD (p.dead_q) + LIVE (p.live_q) yükleri (Gravity yönünde)
    _dist_rows = []
    for f in frames:
        if f.kind not in ("beam_x", "beam_y"):
            continue
        for pat, mag in (("DEAD", p.dead_q), ("LIVE", p.live_q)):
            if mag == 0.0:
                continue
            _dist_rows.append(_kv(
                LoadPat=pat, Frame=f.id, CoordSys="GLOBAL",
                Type="Force", Dir="Gravity",
                RelDistA=0.0, RelDistB=1.0,
                AbsDistA=0.0, AbsDistB=0.0,
                FOverLA=mag, FOverLB=mag,
            ))
    if _dist_rows:
        _table(out, "FRAME LOADS - DISTRIBUTED", _dist_rows)

    # =========================================================== COMBINATIONS
    # Basit TS 500 temel kombinasyonu: 1.4G + 1.6Q
    _table(out, "COMBINATION DEFINITIONS", [
        _kv(
            ComboName="1.4D+1.6L", ComboType="Linear Add", AutoDesign="No",
            CaseType="Linear Static", CaseName="DEAD", ScaleFactor=1.4,
        ),
        _kv(
            ComboName="1.4D+1.6L", ComboType="Linear Add", AutoDesign="No",
            CaseType="Linear Static", CaseName="LIVE", ScaleFactor=1.6,
        ),
    ])

    out.write("\nEND TABLE DATA\n")
    return out.getvalue()
