"""Betonarme (RC) için default malzeme ve kesit hesapları.

TBDY 2018 ve TS 500 konvansiyonlarına yakın değerler:

- Beton: C30/37 varsayılan (fck=30 MPa, fcm=38, Ec=32 GPa ≈ 32e6 kN/m²)
- Poisson oranı: 0.20
- Birim kütle: 2.5 t/m³ = 2500 kg/m³ = 25 kN/m³
- Isıl genleşme: 1.0e-5 / °C

Kesit geometrik özellikleri:
- Kare kolon (b x b):      A = b², I = b⁴/12, J ≈ 0.141 × b⁴ (dikdörtgen yakın-küp)
- Dikdörtgen kiriş (b x h): A = b·h, Iz = b·h³/12 (kuvvetli), Iy = h·b³/12 (zayıf)
  Torsion: J = β·b³·h (b<h için β ≈ 0.33 - 0.21·b/h·(1 - b⁴/(12·h⁴)))
"""

from __future__ import annotations

from dataclasses import dataclass


# ------------------------------------------------------------------ malzemeler
@dataclass(frozen=True)
class ConcreteMaterial:
    """Beton malzeme özeti (SAP2000 kN-m-C birim sisteminde)."""
    id: str
    fck_mpa: int      # karakteristik silindir dayanımı (MPa)
    E: float          # elastisite modülü (kN/m²) — SAP formatında
    nu: float         # Poisson oranı
    unit_mass: float  # birim kütle (ton/m³) — SAP "UnitMass"
    unit_weight: float  # birim hacim ağırlığı (kN/m³) — SAP "UnitWeight"
    alpha: float      # ısıl genleşme katsayısı (1/°C)


def concrete_default(fck_mpa: int = 30) -> ConcreteMaterial:
    """TS 500 / EC2 uyumlu default betonarme malzeme.

    ``fck`` MPa olarak girilir; SAP için kN/m² birimine çevrilir.
    E değeri TS 500 Tablo 3.3'e yakın pratik değerlerle eşlenir.
    """
    # TS 500'e göre Ec ≈ 3250·sqrt(fck) + 14000 (MPa). Pratik tablolar:
    e_table_mpa = {
        20: 28000, 25: 30000, 30: 32000, 35: 33000, 40: 34000,
        45: 36000, 50: 37000,
    }
    e_mpa = e_table_mpa.get(fck_mpa, 32000)
    return ConcreteMaterial(
        id=f"C{fck_mpa}",
        fck_mpa=fck_mpa,
        E=e_mpa * 1000.0,        # MPa → kN/m²  (1 MPa = 1000 kN/m²)
        nu=0.20,
        unit_mass=2.5,           # ton/m³
        unit_weight=24.525,      # 2.5 × 9.81 kN/m³
        alpha=1.0e-5,
    )


# ------------------------------------------------------------------- kesitler
@dataclass(frozen=True)
class FrameSection:
    id: str
    material_id: str
    shape: str           # "Rectangular"
    t2: float            # 2-yönü boyut (m) — genelde kiriş genişliği / kolon x
    t3: float            # 3-yönü boyut (m) — genelde kiriş yüksekliği / kolon y
    A: float
    I22: float           # zayıf eksen etrafı atalet
    I33: float           # kuvvetli eksen etrafı atalet
    J: float             # burulma atalet momenti


def _rect_torsion_const(b: float, h: float) -> float:
    """Dikdörtgen kesit burulma sabiti J.

    Saint-Venant yaklaşımı: J = β·b³·h  (b ≤ h)
    β faktörü h/b oranına göre (Roark's Table 20-3 yaklaşık):

        h/b:   1.0   1.5   2.0   2.5   3.0   4.0   5.0   ∞
        β:    0.141 0.196 0.229 0.249 0.263 0.281 0.291 0.333
    """
    a, c = min(b, h), max(b, h)
    ratio = c / a
    betas = [
        (1.0, 0.141), (1.5, 0.196), (2.0, 0.229), (2.5, 0.249),
        (3.0, 0.263), (4.0, 0.281), (5.0, 0.291), (1e9, 0.333),
    ]
    # Lineer interpolasyon
    for (r1, b1), (r2, b2) in zip(betas, betas[1:]):
        if ratio <= r2:
            t = (ratio - r1) / max(r2 - r1, 1e-9)
            beta = b1 + (b2 - b1) * max(0.0, min(1.0, t))
            return beta * (a ** 3) * c
    return 0.333 * (a ** 3) * c


def rect_section(
    section_id: str,
    material_id: str,
    b: float,
    h: float,
) -> FrameSection:
    """Dikdörtgen kesit (b x h): SAP2000 konvansiyonu
    t2 = 2-yönü boyut (enlem), t3 = 3-yönü boyut (derinlik/yükseklik).
    Bizim konvansiyon: b = kiriş genişliği (t2), h = yükseklik (t3).
    I33 kuvvetli eksen (b·h³/12), I22 zayıf eksen (h·b³/12).
    """
    A = b * h
    I22 = h * (b ** 3) / 12.0
    I33 = b * (h ** 3) / 12.0
    J = _rect_torsion_const(b, h)
    return FrameSection(
        id=section_id,
        material_id=material_id,
        shape="Rectangular",
        t2=b,
        t3=h,
        A=A,
        I22=I22,
        I33=I33,
        J=J,
    )


def square_column(section_id: str, material_id: str, size: float) -> FrameSection:
    """Kare kolon yardımcısı (size x size)."""
    return rect_section(section_id, material_id, size, size)
