"""Yapısal analiz API DTO'ları.

Backend pipeline (``services.structural_analysis``) ile frontend arasındaki
sözleşme. METHOD.md §3-4 spec'ine uygun.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AnalysisOptionsDto(BaseModel):
    """POST /analyze body içinde gelen istek seçenekleri."""

    linear_static: bool = True
    modal: bool = False
    modal_n_modes: int = 12
    response_spectrum: bool = False
    spectrum_code: Literal["TBDY_2018", "EC8", "CUSTOM"] | None = None
    pdelta: bool = False
    auto_combinations: bool = True
    combination_code: Literal["TBDY_2018", "EC0", "ASCE7"] = "TBDY_2018"
    solver: Literal["direct", "iterative"] = "direct"
    output_detail: Literal["summary", "full"] = "full"

    # Seçim filtreleri — None = hepsi, boş liste = hiç.
    # İstek: SAP'taki "run checked load cases" mantığı.
    selected_load_cases: list[str] | None = None
    selected_combinations: list[str] | None = None


class AnalyzeRequestDto(BaseModel):
    """POST /api/projects/{pid}/files/{fid}/analyze body."""

    options: AnalysisOptionsDto = Field(default_factory=AnalysisOptionsDto)


class ModelSummaryDto(BaseModel):
    """Ayrıştırılan modelin özet istatistikleri."""

    n_nodes: int
    n_frame_elements: int
    n_shell_elements: int
    n_dofs_free: int
    n_dofs_total: int
    n_load_cases: int
    max_displacement: float


class AnalysisStatusDto(BaseModel):
    """POST response ve GET /analyses/{aid} yanıtı."""

    analysis_id: str
    file_id: str
    project_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: datetime
    completed_at: datetime | None = None
    duration_ms: int | None = None
    options: AnalysisOptionsDto
    summary: ModelSummaryDto | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class CaseSummaryDto(BaseModel):
    case_id: str
    max_abs_displacement: float
    n_nodes_with_reaction: int


class NodeDisplacementDto(BaseModel):
    node_id: int
    load_case: str
    ux: float
    uy: float
    uz: float
    rx: float
    ry: float
    rz: float


class ReactionDto(BaseModel):
    node_id: int
    load_case: str
    fx: float
    fy: float
    fz: float
    mx: float
    my: float
    mz: float


class AnalysisListItemDto(BaseModel):
    """GET /files/{fid}/analyses listesinin her bir öğesi."""

    analysis_id: str
    status: str
    created_at: datetime
    duration_ms: int | None = None
    summary: ModelSummaryDto | None = None


# ------------------------------------------------------------------ preview
class LoadCasePreviewDto(BaseModel):
    """GET /preview — Dosyadaki yük durumlarını özetler (analiz öncesi seçim)."""

    id: str
    type: str                           # "dead", "live", "earthquake_x", ...
    self_weight_factor: float
    n_point_loads: int
    n_distributed_loads: int


class CombinationPreviewDto(BaseModel):
    id: str
    factors: dict[str, float]           # {case_id: scale_factor}


class ModelPreviewDto(BaseModel):
    """Analiz başlatmadan önce kullanıcıya seçenekleri göstermek için."""

    n_nodes: int
    n_frame_elements: int
    n_shell_elements: int
    load_cases: list[LoadCasePreviewDto]
    combinations: list[CombinationPreviewDto]
    warnings: list[str] = Field(default_factory=list)


# -------------------------------------------------------------------- modes
class ModeDto(BaseModel):
    """Modal analiz — tek bir mod."""

    mode_no: int
    period: float                       # saniye
    frequency: float                    # Hz
    angular_frequency: float            # rad/s
    mass_participation: dict[str, float] = Field(default_factory=dict)
