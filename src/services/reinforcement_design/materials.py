"""Beton + çelik sınıfları — TS 500 ve TBDY 2018 değerleri.

TS 500 Tablo 3.1 (beton karakteristik dayanımları), Tablo 3.2 (çelik
karakteristik dayanımları). Hesap dayanımları γ_c = 1.5, γ_s = 1.15 ile.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ----------------------------------------------------- material safety factors
# TS 500 §6.2.5
GAMMA_C = 1.5      # beton kısmi güvenlik
GAMMA_S = 1.15     # çelik kısmi güvenlik


class ConcreteGrade(str, Enum):
    """TS 500 Tablo 3.1 — karakteristik silindir basınç dayanımı f_ck (MPa)."""
    C16_20 = "C16/20"
    C20_25 = "C20/25"
    C25_30 = "C25/30"
    C30_37 = "C30/37"
    C35_45 = "C35/45"
    C40_50 = "C40/50"
    C45_55 = "C45/55"
    C50_60 = "C50/60"


class SteelGrade(str, Enum):
    """TS 500 + TS 708 — çelik karakteristik akma dayanımı f_yk (MPa)."""
    S220 = "S220"      # düz yüzeyli — eski, artık nadir
    S420 = "S420"      # eski standart, hâlâ yaygın
    B420C = "B420C"    # TS 708 yeni notasyon
    B500C = "B500C"    # TBDY 2018 yüksek süneklik tercihi


# TS 500 §3.2 — silindir/kübik karakteristik dayanım değerleri (MPa)
_CONCRETE_FCK: dict[ConcreteGrade, float] = {
    ConcreteGrade.C16_20: 16.0,
    ConcreteGrade.C20_25: 20.0,
    ConcreteGrade.C25_30: 25.0,
    ConcreteGrade.C30_37: 30.0,
    ConcreteGrade.C35_45: 35.0,
    ConcreteGrade.C40_50: 40.0,
    ConcreteGrade.C45_55: 45.0,
    ConcreteGrade.C50_60: 50.0,
}

_STEEL_FYK: dict[SteelGrade, float] = {
    SteelGrade.S220: 220.0,
    SteelGrade.S420: 420.0,
    SteelGrade.B420C: 420.0,
    SteelGrade.B500C: 500.0,
}


@dataclass(frozen=True)
class MaterialProperties:
    """Bir tasarım çağrısı için sabitlenmiş malzeme dayanımları.

    Hesap dayanımları (f_cd, f_yd) MPa cinsinden tutulur. Kullanıcı
    SI birim sisteminde (N, mm) çalışmak için tüm değerleri MPa = N/mm².
    """
    concrete: ConcreteGrade
    steel: SteelGrade

    # TS 500 §3.2 hesap dayanımları (cached property gibi davranır)
    @property
    def f_ck(self) -> float:
        """Karakteristik silindir basınç dayanımı (MPa)."""
        return _CONCRETE_FCK[self.concrete]

    @property
    def f_cd(self) -> float:
        """Beton hesap basınç dayanımı (MPa). f_cd = f_ck / γ_c."""
        return self.f_ck / GAMMA_C

    @property
    def f_ctk(self) -> float:
        """Karakteristik eksen çekme dayanımı (MPa). TS 500 §3.2.3."""
        return 0.35 * (self.f_ck ** 0.5)

    @property
    def f_ctd(self) -> float:
        """Beton hesap çekme dayanımı (MPa). f_ctd = f_ctk / γ_c."""
        return self.f_ctk / GAMMA_C

    @property
    def E_c(self) -> float:
        """Beton sekant elastisite modülü (MPa). TS 500 §3.2.5."""
        return 3250.0 * (self.f_ck ** 0.5) + 14000.0

    @property
    def f_yk(self) -> float:
        """Çelik karakteristik akma dayanımı (MPa)."""
        return _STEEL_FYK[self.steel]

    @property
    def f_yd(self) -> float:
        """Çelik hesap akma dayanımı (MPa). f_yd = f_yk / γ_s."""
        return self.f_yk / GAMMA_S

    @property
    def E_s(self) -> float:
        """Çelik elastisite modülü (MPa) — sabit 200 GPa."""
        return 200_000.0

    @property
    def epsilon_cu(self) -> float:
        """Maksimum beton birim kısalması — TS 500 §7.1: 0.003."""
        return 0.003

    @property
    def epsilon_yd(self) -> float:
        """Akma birim uzaması = f_yd / E_s."""
        return self.f_yd / self.E_s
