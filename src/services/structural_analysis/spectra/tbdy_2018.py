"""TBDY 2018 (Türkiye Bina Deprem Yönetmeliği 2018) elastik tasarım spektrumu.

Referans: TBDY 2018 Madde 2.3 "Tasarım İvme Spektrumu".

Eğri:
    T ≤ TA:          Sae(T) = (0.4 + 0.6 T/TA) × SDS
    TA < T ≤ TB:     Sae(T) = SDS
    TB < T ≤ TL:     Sae(T) = SD1 / T
    T  > TL:         Sae(T) = SD1 × TL / T²

Tasarım ivmesi (R ile azaltılmış):
    Sar(T) = Sae(T) × I / Ra(T)
burada:
    Ra(T) = R / I   (T ≥ TB)
    Ra(T) = D + (R/I − D) × T/TB   (T < TB)   — TBDY Madde 4.10.1.1
    D = 1.5  (çoğu sistem için varsayılan)

SDS ve SD1:
    SDS = Ss × FS   (kısa periyot tasarım spektrumu)
    SD1 = S1 × F1   (1-saniye tasarım spektrumu)
    FS, F1 — zemin katsayıları (TBDY Madde 2.2 Tablo 2.1 / 2.2)
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Literal

SoilClass = Literal["ZA", "ZB", "ZC", "ZD", "ZE"]


# TBDY 2018 Tablo 2.1 — FS zemin katsayıları
# Kolonlar: Ss = [0.25, 0.50, 0.75, 1.00, 1.25, ≥1.50]
_FS_TABLE: dict[SoilClass, list[float]] = {
    "ZA": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    "ZB": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    "ZC": [1.3, 1.3, 1.2, 1.2, 1.2, 1.2],
    "ZD": [1.6, 1.4, 1.2, 1.1, 1.0, 1.0],
    "ZE": [2.4, 1.7, 1.3, 1.1, 0.9, 0.8],
}
_SS_KEYS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

# TBDY 2018 Tablo 2.2 — F1 zemin katsayıları
# Kolonlar: S1 = [0.10, 0.20, 0.30, 0.40, 0.50, ≥0.60]
_F1_TABLE: dict[SoilClass, list[float]] = {
    "ZA": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    "ZB": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    "ZC": [1.5, 1.5, 1.5, 1.5, 1.5, 1.4],
    "ZD": [2.4, 2.2, 2.0, 1.9, 1.8, 1.7],
    "ZE": [4.2, 3.3, 2.8, 2.4, 2.2, 2.0],
}
_S1_KEYS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]


def soil_coefficients(
    soil: SoilClass, Ss: float, S1: float
) -> tuple[float, float]:
    """TBDY 2018 Tablo 2.1 / 2.2 — (FS, F1) döndürür.

    Tablo sınırlarını aşan değerlerde en yakın sınır kullanılır; aradaki
    değerler lineer interpolasyon ile bulunur.
    """
    FS = _interp(Ss, _SS_KEYS, _FS_TABLE[soil])
    F1 = _interp(S1, _S1_KEYS, _F1_TABLE[soil])
    return FS, F1


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


@dataclass(frozen=True)
class TBDY2018Spectrum:
    """TBDY 2018 elastik + tasarım spektrumu.

    ``Sa_elastic(T)`` elastik spektrum (g cinsinden),
    ``Sa_design(T)`` R ile azaltılmış tasarım spektrumu.
    ``g`` parametresi ivme birim çevrimi için (varsayılan 9.80665 m/s²).
    """

    Ss: float
    S1: float
    soil: SoilClass = "ZC"
    R: float = 4.0
    I: float = 1.0
    D: float = 1.5          # TBDY denklem 4.1 için katsayı
    TL: float = 6.0         # Uzun periyot köşesi (s)
    g: float = 9.80665      # m/s²

    @property
    def FS(self) -> float:
        return soil_coefficients(self.soil, self.Ss, self.S1)[0]

    @property
    def F1(self) -> float:
        return soil_coefficients(self.soil, self.Ss, self.S1)[1]

    @property
    def SDS(self) -> float:
        return self.Ss * self.FS

    @property
    def SD1(self) -> float:
        return self.S1 * self.F1

    @property
    def TA(self) -> float:
        sds = self.SDS
        return 0.2 * self.SD1 / sds if sds > 0 else 0.0

    @property
    def TB(self) -> float:
        sds = self.SDS
        return self.SD1 / sds if sds > 0 else 0.0

    def Sa_elastic(self, T: float) -> float:
        """Elastik spektrum Sae(T), g cinsinden."""
        if T < 0:
            return 0.0
        TA, TB, TL = self.TA, self.TB, self.TL
        SDS, SD1 = self.SDS, self.SD1
        if T <= TA:
            if TA == 0:
                return SDS
            return (0.4 + 0.6 * T / TA) * SDS
        if T <= TB:
            return SDS
        if T <= TL:
            return SD1 / T
        return SD1 * TL / (T * T)

    def Ra(self, T: float) -> float:
        """Deprem yükü azaltma katsayısı (TBDY Denklem 4.1)."""
        TB = self.TB
        if T >= TB:
            return self.R / self.I
        # T < TB: lineer geçiş (D + (R/I − D) × T/TB)
        if TB == 0:
            return self.R / self.I
        return self.D + (self.R / self.I - self.D) * T / TB

    def Sa_design(self, T: float) -> float:
        """Tasarım ivmesi Sar(T), g cinsinden. = Sae(T) × I / Ra(T)."""
        ra = self.Ra(T)
        if ra == 0:
            return 0.0
        return self.Sa_elastic(T) * self.I / ra

    def Sa_design_ms2(self, T: float) -> float:
        """Tasarım ivmesi m/s² cinsinden."""
        return self.Sa_design(T) * self.g
