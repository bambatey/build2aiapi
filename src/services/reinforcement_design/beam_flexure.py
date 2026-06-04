"""Kiriş eğilme donatısı tasarımı — TS 500 §7.4.

Kapsam (MVP):
- Tek yönlü (basit) eğilme, dikdörtgen kesit
- Pozitif veya negatif moment için tek taraflı donatı (singly reinforced)
- Çift donatılı tasarım: gerekirse `requires_double_reinforcement=True` döner;
  hesaplanır ama compression çubuğu için ayrı algoritma (TS 500 §7.4.2)
  Faz 2'nin ikinci PR'ına bırakılıyor.

Notasyon (TS 500):
  K_d = M_d / (b × d² × f_cd)                    boyutsuz moment
  ω   = 1 − √(1 − 2 × K_d)                       mekanik donatı oranı
  ρ   = ω × (f_cd / f_yd)                        geometrik donatı oranı
  A_s = ρ × b × d                                gerekli donatı alanı (mm²)

Sınırlar:
  ρ_min = 0.8 × f_ctd / f_yd                     TS 500 §7.4.1
  ρ_max = 0.85 × ρ_b                             sünek tasarım için
         ρ_b = 0.85 × β_1 × (f_cd / f_yd) × (ε_cu / (ε_cu + ε_yd))

  β_1 = 0.85                                     f_ck ≤ 28 MPa için
  β_1 = 0.85 − 0.0075 × (f_ck − 28)              28 < f_ck ≤ 56 MPa
  β_1 = 0.65                                     f_ck > 56 MPa
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .bar_sizes import BarSelection, select_bars
from .materials import MaterialProperties
from .types import DesignError, RectangularSection


# ----------------------------------------------------------------- veri yapıları
@dataclass(frozen=True)
class BeamFlexureInput:
    """Bir kiriş kesitinde eğilme tasarımı için tek girdi.

    M_design_kNm: |M| design momenti (kNm cinsinden, işaret önemsiz —
    pozitif/negatif tarafı çağrı yerinde ayrılır).
    """
    section: RectangularSection
    materials: MaterialProperties
    M_design_kNm: float

    # Bar seçimi sınırları (opsiyonel override)
    min_bar_diameter_mm: int = 12
    max_bar_diameter_mm: int = 25
    min_bar_count: int = 2
    max_bar_count: int = 8


@dataclass(frozen=True)
class BarLayout:
    """Kesitin bir yüzeyindeki donatı yerleşimi."""
    selection: BarSelection
    A_s_provided_mm2: float

    @property
    def label(self) -> str:
        return self.selection.label


@dataclass
class BeamFlexureResult:
    """Tasarım sonucu — endpoint serialize ederken JSON'a basılır."""
    A_s_required_mm2: float
    A_s_min_mm2: float
    A_s_max_mm2: float
    rho_required: float
    rho_min: float
    rho_max: float
    K_d: float
    omega: float
    bars: BarLayout | None
    requires_double_reinforcement: bool
    warnings: list[str] = field(default_factory=list)


# -------------------------------------------------------- ana algoritma
def design_beam_flexure(spec: BeamFlexureInput) -> BeamFlexureResult:
    """Tek yönlü dikdörtgen kiriş eğilme donatısı hesabı.

    Birim dönüşüm: M (kNm) × 10⁶ = M (Nmm).
    """
    m = spec.materials
    sec = spec.section
    warnings: list[str] = []

    if sec.b_mm <= 0 or sec.h_mm <= 0:
        raise DesignError("Kesit b ve h pozitif olmalı.")
    d = sec.d_mm
    if d <= 0:
        raise DesignError(
            f"Yararlı yükseklik d ≤ 0 (h={sec.h_mm}, cover={sec.cover_mm}). "
            "Kesit veya pas payı değerlerini kontrol edin."
        )

    M_d_Nmm = abs(spec.M_design_kNm) * 1e6
    b, f_cd, f_yd = sec.b_mm, m.f_cd, m.f_yd

    # K_d, ω, ρ
    K_d = M_d_Nmm / (b * d * d * f_cd)
    if K_d >= 0.5:
        raise DesignError(
            f"K_d = {K_d:.3f} ≥ 0.5 — kesit eğilme momentini taşıyamaz."
        )

    # ρ_b ve ρ_max (sünek tasarım sınırı)
    beta_1 = _beta_1(m.f_ck)
    epsilon_cu, epsilon_yd = m.epsilon_cu, m.epsilon_yd
    rho_b = (
        0.85 * beta_1 * (f_cd / f_yd) * (epsilon_cu / (epsilon_cu + epsilon_yd))
    )
    rho_max = 0.85 * rho_b
    K_d_max = _K_d_from_rho(rho_max, f_cd, f_yd)

    if K_d > K_d_max:
        # Tek tarafdı tasarım yetmez — çift donatılı gerek
        requires_double = True
        # Burada tek-tarafdı tasarımın "max" donatı miktarını veriyoruz, ama
        # bu reel A_s'yi karşılamaz. Çağrı yeri çift donatı modülüne yönlenmeli.
        A_s_req = rho_max * b * d
        warnings.append(
            f"K_d = {K_d:.3f} > K_d,max = {K_d_max:.3f} — çift donatılı tasarım gerekli "
            "(compression bar). Şu an MVP tek tarafı veriyor."
        )
    else:
        requires_double = False

    omega = 1.0 - math.sqrt(max(0.0, 1.0 - 2.0 * min(K_d, K_d_max)))
    rho = omega * (f_cd / f_yd)
    A_s_calc = rho * b * d

    # ρ_min (TS 500 §7.4.1)
    rho_min = 0.8 * m.f_ctd / m.f_yd
    A_s_min = rho_min * b * d
    A_s_max = rho_max * b * d

    A_s_req = max(A_s_calc, A_s_min)
    if rho < rho_min:
        warnings.append(
            f"ρ = {rho:.5f} < ρ_min = {rho_min:.5f} — minimum donatı kuralı uygulandı."
        )

    # Bar seçimi (count × Ø)
    sel = select_bars(
        A_s_req,
        min_diameter_mm=spec.min_bar_diameter_mm,
        max_diameter_mm=spec.max_bar_diameter_mm,
        min_count=spec.min_bar_count,
        max_count=spec.max_bar_count,
    )
    if sel is None:
        warnings.append(
            "Uygun tek çaplı bar kombinasyonu bulunamadı; çoklu çaplı detaylama gerekli."
        )
        bars = None
    else:
        bars = BarLayout(selection=sel, A_s_provided_mm2=sel.area_mm2)

    return BeamFlexureResult(
        A_s_required_mm2=A_s_req,
        A_s_min_mm2=A_s_min,
        A_s_max_mm2=A_s_max,
        rho_required=A_s_req / (b * d),
        rho_min=rho_min,
        rho_max=rho_max,
        K_d=K_d,
        omega=omega,
        bars=bars,
        requires_double_reinforcement=requires_double,
        warnings=warnings,
    )


# ----------------------------------------------------------------- helpers
def _beta_1(f_ck_MPa: float) -> float:
    """TS 500 §7.1 — eşdeğer dikdörtgen blok derinliği katsayısı.

    f_ck'a göre lineer azalır.
    """
    if f_ck_MPa <= 28.0:
        return 0.85
    if f_ck_MPa <= 56.0:
        return 0.85 - 0.0075 * (f_ck_MPa - 28.0)
    return 0.65


def _K_d_from_rho(rho: float, f_cd: float, f_yd: float) -> float:
    """Verili ρ için K_d. Sünek sınır kontrolünde kullanılır."""
    omega = rho * f_yd / f_cd
    return omega - omega * omega / 2.0
