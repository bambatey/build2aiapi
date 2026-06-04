"""Standart inşaat demiri çapları + alan tablosu (TS 708).

Türkiye'de yaygın kullanılan donatı çapları. Hesapta küçükten büyüğe
gidip ilk yeterli kombi seçilir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# TS 708 / İMO konvansiyonel çelik donatı çapları (mm)
STANDARD_BAR_DIAMETERS_MM: tuple[int, ...] = (
    8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30, 32,
)


def bar_area_mm2(diameter_mm: int) -> float:
    """Tek bir donatının kesit alanı — π × Ø² / 4 (mm²)."""
    return math.pi * (diameter_mm ** 2) / 4.0


@dataclass(frozen=True)
class BarSelection:
    """Bir donatı seti — n adet Ø_d demir.

    `A_s` = n × A_single. `label` çizimde kullanılır ("3Ø16").
    """
    count: int
    diameter_mm: int

    @property
    def area_mm2(self) -> float:
        return self.count * bar_area_mm2(self.diameter_mm)

    @property
    def label(self) -> str:
        return f"{self.count}Ø{self.diameter_mm}"


def select_bars(
    required_area_mm2: float,
    *,
    min_diameter_mm: int = 12,
    max_diameter_mm: int = 25,
    min_count: int = 2,
    max_count: int = 8,
    preferred_max_count: int = 4,
) -> BarSelection | None:
    """A_s_req'i karşılayan en uygun tek-çaplı (count×Ø) kombiyi döndür.

    TR proje pratiği: aynı kesitte az ama büyük çap (3Ø16) > çok küçük
    çap (5Ø12) — kalıp/montaj kolaylığı + çelik standart kalemleri ile
    uyum. Bu nedenle skor:
        skor = excess + bar_count_penalty
        bar_count_penalty = max(0, count - preferred_max_count) × 50 mm²

    Excess: |A_provided - A_required|. Penalty küçük adetli kombileri
    (≤ preferred_max_count) çekici hâle getirir; gerekirse yine 5-8
    adetli kombi seçilebilir.
    """
    if required_area_mm2 <= 0:
        return BarSelection(count=min_count, diameter_mm=min_diameter_mm)

    best: BarSelection | None = None
    best_score = float("inf")

    for d in STANDARD_BAR_DIAMETERS_MM:
        if d < min_diameter_mm or d > max_diameter_mm:
            continue
        a1 = bar_area_mm2(d)
        n_needed = max(min_count, math.ceil(required_area_mm2 / a1))
        if n_needed > max_count:
            continue
        excess = n_needed * a1 - required_area_mm2
        count_penalty = max(0, n_needed - preferred_max_count) * 50.0
        score = excess + count_penalty
        if score < best_score:
            best = BarSelection(count=n_needed, diameter_mm=d)
            best_score = score

    return best
