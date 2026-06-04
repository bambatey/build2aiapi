"""Ortak veri yapıları."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RectangularSection:
    """Dikdörtgen kesit geometrisi — kiriş ve kolon için ortak.

    Boyutlar **mm** cinsinden (TS 500 hesabı SI MPa-mm).
    """
    b_mm: float                      # kesit genişliği (mm)
    h_mm: float                      # kesit toplam derinliği (mm)
    cover_mm: float = 25.0           # net pas payı (TS 500 §3.3.4: iç ortam ≥25mm)
    stirrup_diameter_mm: float = 8.0
    longitudinal_diameter_mm: float = 16.0  # bar selection için ön tahmin (d hesabı)

    @property
    def d_mm(self) -> float:
        """Yararlı yükseklik d = h − cover − Ø_stirrup − Ø_bar/2."""
        return (
            self.h_mm
            - self.cover_mm
            - self.stirrup_diameter_mm
            - self.longitudinal_diameter_mm / 2.0
        )


class DesignError(Exception):
    """Tasarım kuralı ihlali — kullanıcıya gösterilebilir mesaj."""
