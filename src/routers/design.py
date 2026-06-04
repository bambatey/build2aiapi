"""Reinforcement Design Router — TS 500 + TBDY 2018 donatı hesabı.

Faz 2 MVP (Standalone Calculator):
  POST /api/design/beam/flexure
    body: BeamFlexureRequest (kesit + malzeme + M_design)
    return: BeamFlexureResponse (A_s, bars, K_d, ω, uyarılar)

Element forces pipeline (analiz → her elemente M/V/N çıkarma) henüz yok.
O entegre olunca:
  POST /api/projects/{pid}/files/{fid}/analyses/{aid}/design/beams
    → her kiriş için otomatik tasarım, listede döner.

Şu an endpoint *calculator* modunda: kullanıcı M değerini direkt verir.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dependencies import get_uid
from models.dto import BusinessLogicDto
from services.reinforcement_design import (
    BeamFlexureInput,
    ConcreteGrade,
    MaterialProperties,
    RectangularSection,
    SteelGrade,
    design_beam_flexure,
)
from services.reinforcement_design.types import DesignError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/design", tags=["reinforcement_design"])


# --------------------------------------------------------------- request/response
class SectionDto(BaseModel):
    b_mm: float = Field(..., gt=0, le=5000, description="Kesit genişliği")
    h_mm: float = Field(..., gt=0, le=5000, description="Kesit derinliği")
    cover_mm: float = Field(25.0, ge=15, le=80, description="Net pas payı")
    stirrup_diameter_mm: float = Field(8.0, ge=6, le=14)
    longitudinal_diameter_mm: float = Field(16.0, ge=10, le=32,
                                             description="d hesabı için ön tahmin")


class BeamFlexureRequest(BaseModel):
    section: SectionDto
    concrete: ConcreteGrade = ConcreteGrade.C30_37
    steel: SteelGrade = SteelGrade.B500C
    M_design_kNm: float = Field(..., description="Tasarım momenti — işaret önemsiz")

    # Bar seçim sınırları (opsiyonel)
    min_bar_diameter_mm: int = Field(12, ge=8, le=32)
    max_bar_diameter_mm: int = Field(25, ge=10, le=32)
    min_bar_count: int = Field(2, ge=2, le=20)
    max_bar_count: int = Field(8, ge=2, le=20)


class BarLayoutDto(BaseModel):
    count: int
    diameter_mm: int
    A_s_provided_mm2: float
    label: str


class BeamFlexureResponse(BaseModel):
    A_s_required_mm2: float
    A_s_min_mm2: float
    A_s_max_mm2: float
    rho_required: float
    rho_min: float
    rho_max: float
    K_d: float
    omega: float
    bars: BarLayoutDto | None
    requires_double_reinforcement: bool
    warnings: list[str]

    # Türetilmiş — frontend kolaylığı
    d_mm: float
    concrete: str
    steel: str
    f_cd_MPa: float
    f_yd_MPa: float


# ----------------------------------------------------------------- endpoint
@router.post(
    "/beam/flexure",
    response_model=BusinessLogicDto[BeamFlexureResponse],
)
async def design_beam_flexure_endpoint(
    request: BeamFlexureRequest,
    uid: str = Depends(get_uid),
):
    """Tek bir kiriş kesiti için TS 500 §7.4 eğilme donatısı hesabı.

    Standalone calculator: M_design_kNm girdisi kullanıcıdan gelir.
    Analiz pipeline entegrasyonu sonraki PR'da (element forces eklenince).
    """
    sec = RectangularSection(
        b_mm=request.section.b_mm,
        h_mm=request.section.h_mm,
        cover_mm=request.section.cover_mm,
        stirrup_diameter_mm=request.section.stirrup_diameter_mm,
        longitudinal_diameter_mm=request.section.longitudinal_diameter_mm,
    )
    mat = MaterialProperties(concrete=request.concrete, steel=request.steel)
    spec = BeamFlexureInput(
        section=sec, materials=mat,
        M_design_kNm=request.M_design_kNm,
        min_bar_diameter_mm=request.min_bar_diameter_mm,
        max_bar_diameter_mm=request.max_bar_diameter_mm,
        min_bar_count=request.min_bar_count,
        max_bar_count=request.max_bar_count,
    )

    try:
        result = design_beam_flexure(spec)
    except DesignError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Beam flexure design failed")
        raise HTTPException(status_code=500, detail=f"Hesap hatası: {exc}") from exc

    bars_dto: BarLayoutDto | None = None
    if result.bars is not None:
        bars_dto = BarLayoutDto(
            count=result.bars.selection.count,
            diameter_mm=result.bars.selection.diameter_mm,
            A_s_provided_mm2=result.bars.A_s_provided_mm2,
            label=result.bars.selection.label,
        )

    response = BeamFlexureResponse(
        A_s_required_mm2=result.A_s_required_mm2,
        A_s_min_mm2=result.A_s_min_mm2,
        A_s_max_mm2=result.A_s_max_mm2,
        rho_required=result.rho_required,
        rho_min=result.rho_min,
        rho_max=result.rho_max,
        K_d=result.K_d,
        omega=result.omega,
        bars=bars_dto,
        requires_double_reinforcement=result.requires_double_reinforcement,
        warnings=result.warnings,
        d_mm=sec.d_mm,
        concrete=mat.concrete.value,
        steel=mat.steel.value,
        f_cd_MPa=mat.f_cd,
        f_yd_MPa=mat.f_yd,
    )
    return BusinessLogicDto(success=True, data=response)
