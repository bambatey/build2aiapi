"""Betonarme donatı tasarımı modülü (Faz 2).

Kapsam:
  - Kiriş eğilme donatısı (TS 500 §7.4) — **MVP buradan başlıyor**
  - Kiriş kesme/etriye (TS 500 §7.5, TBDY 2018 §7.4.5) — sıradaki
  - Kolon eksenel + iki yönlü eğilme (TS 500 §7.3) — sonraki PR
  - Kolon etriye (TBDY 2018 §7.3.4) — sonraki PR
  - Plak/temel — sonraki fazlar

Hesap saf matematik library olarak yapılır; analiz pipeline'ı henüz
element forces (M, V, N) üretmediği için endpoint integrasyonu Faz 2'nin
ikinci adımı. Şu an "M_design ve V_design verili" senaryoda hesap yapılır.

Validation hedefi: STA4CAD / ProBina referans projeleriyle %2 tolerans
içinde sonuç vermek.
"""

from .beam_flexure import (
    BeamFlexureInput,
    BeamFlexureResult,
    BarLayout,
    design_beam_flexure,
)
from .materials import ConcreteGrade, SteelGrade, MaterialProperties
from .types import RectangularSection

__all__ = [
    "BarLayout",
    "BeamFlexureInput",
    "BeamFlexureResult",
    "ConcreteGrade",
    "MaterialProperties",
    "RectangularSection",
    "SteelGrade",
    "design_beam_flexure",
]
