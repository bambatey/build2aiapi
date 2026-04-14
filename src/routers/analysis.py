"""
Yapısal Analiz Router — Statik Lineer MVP + Modal + Kombinasyonlar

GET    /api/projects/{pid}/files/{fid}/preview                  → Model özeti + yük seçim seçenekleri
POST   /api/projects/{pid}/files/{fid}/analyze                  → Analiz tetikle (senkron)
GET    /api/projects/{pid}/files/{fid}/analyses                 → Dosyanın geçmişi
GET    /api/projects/{pid}/files/{fid}/analyses/{aid}           → Status + özet
GET    /api/projects/{pid}/files/{fid}/analyses/{aid}/displacements[?load_case=...]
GET    /api/projects/{pid}/files/{fid}/analyses/{aid}/reactions[?load_case=...]
GET    /api/projects/{pid}/files/{fid}/analyses/{aid}/summary
GET    /api/projects/{pid}/files/{fid}/analyses/{aid}/modes     → Modal mod şekilleri
DELETE /api/projects/{pid}/files/{fid}/analyses/{aid}
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status

from dependencies import get_uid
from models.analysis_dto import (
    AnalysisListItemDto,
    AnalysisStatusDto,
    AnalyzeRequestDto,
    CombinationPreviewDto,
    LoadCasePreviewDto,
    ModeDto,
    ModelPreviewDto,
    ModelSummaryDto,
    NodeDisplacementDto,
    ReactionDto,
)
from models.dto import BusinessLogicDto
from repositories import analysis_repository, file_repository
from services import storage_service
from services.structural_analysis.exceptions import StructuralAnalysisError
from services.structural_analysis.parser import parse_s2k
from services.structural_analysis.pipeline import (
    AnalysisOptions,
    SpectrumOptions,
    run_from_s2k,
)
from services.structural_analysis.results import analysis_to_persistable
from services.structural_analysis.validation import validate_model

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/projects/{project_id}/files/{file_id}",
    tags=["analysis"],
)


# ---------------------------------------------------------------- PREVIEW
@router.get(
    "/preview",
    response_model=BusinessLogicDto[ModelPreviewDto],
)
async def preview_model(
    project_id: str,
    file_id: str,
    uid: str = Depends(get_uid),
):
    """Dosyayı parse et, yük durumları + kombinasyonları + özet döndür.

    Frontend bu veriyle analiz-öncesi seçici modalını doldurur.
    """
    file_meta = await file_repository.get(uid, project_id, file_id)
    if not file_meta:
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")
    storage_path = file_meta.get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=400, detail="Dosya içeriği mevcut değil")

    text = await storage_service.download_file(storage_path)
    try:
        model = parse_s2k(text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Parse hatası: {exc}") from exc

    # Kısa validator uyarıları
    report = validate_model(model)
    warnings_list = [f"[{i.code}] {i.message}" for i in report.issues]

    cases = [
        LoadCasePreviewDto(
            id=c.id,
            type=c.type.value,
            self_weight_factor=c.self_weight_factor,
            n_point_loads=len(c.point_loads),
            n_distributed_loads=len(c.distributed_loads),
        )
        for c in model.load_cases.values()
    ]
    combos = [
        CombinationPreviewDto(id=c.id, factors=c.factors)
        for c in model.combinations
    ]

    return BusinessLogicDto(
        success=True,
        data=ModelPreviewDto(
            n_nodes=len(model.nodes),
            n_frame_elements=len(model.frame_elements),
            n_shell_elements=len(model.shell_elements),
            load_cases=cases,
            combinations=combos,
            warnings=warnings_list[:50],
        ),
    )


# --------------------------------------------------------------------- POST
@router.post(
    "/analyze",
    response_model=BusinessLogicDto[AnalysisStatusDto],
    status_code=status.HTTP_201_CREATED,
)
async def trigger_analysis(
    project_id: str,
    file_id: str,
    request: AnalyzeRequestDto,
    uid: str = Depends(get_uid),
):
    """Statik lineer analizi senkron olarak çalıştır ve sonucu kaydet."""
    file_meta = await file_repository.get(uid, project_id, file_id)
    if not file_meta:
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")
    storage_path = file_meta.get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=400, detail="Dosya içeriği mevcut değil")

    s2k_text = await storage_service.download_file(storage_path)

    # API options → iç dataclass
    sp_dto = request.options.spectrum_params
    spectrum_opts = SpectrumOptions(
        Ss=sp_dto.Ss, S1=sp_dto.S1, soil=sp_dto.soil,
        R=sp_dto.R, I=sp_dto.I,
        run_x=sp_dto.run_x, run_y=sp_dto.run_y,
    ) if sp_dto is not None else None

    pipeline_options = AnalysisOptions(
        selected_load_cases=request.options.selected_load_cases,
        selected_combinations=request.options.selected_combinations,
        run_modal=request.options.modal,
        modal_n_modes=request.options.modal_n_modes,
        run_response_spectrum=request.options.response_spectrum,
        spectrum=spectrum_opts,
    )

    started = time.perf_counter()
    try:
        result = run_from_s2k(s2k_text, pipeline_options)
    except StructuralAnalysisError as exc:
        logger.exception("Analiz başarısız (project=%s file=%s)", project_id, file_id)
        # Hata kaydını da Firestore'a yaz ki frontend görebilsin
        duration_ms = int((time.perf_counter() - started) * 1000)
        created = await analysis_repository.create(
            uid=uid, project_id=project_id, file_id=file_id,
            options=request.options.model_dump(),
            summary={}, cases={}, modes=[], warnings=[],
            duration_ms=duration_ms, status="failed", error=str(exc),
        )
        return BusinessLogicDto(
            success=False,
            error=str(exc),
            data=_to_status_dto(created, request),
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    persistable = analysis_to_persistable(result)

    created = await analysis_repository.create(
        uid=uid,
        project_id=project_id,
        file_id=file_id,
        options=request.options.model_dump(),
        summary=persistable["summary"],
        cases=persistable["cases"],
        modes=persistable["modes"],
        warnings=result.warnings,
        duration_ms=duration_ms,
        status="completed",
    )
    return BusinessLogicDto(success=True, data=_to_status_dto(created, request))


# --------------------------------------------------------------------- LIST
@router.get(
    "/analyses",
    response_model=BusinessLogicDto[list[AnalysisListItemDto]],
)
async def list_analyses(
    project_id: str,
    file_id: str,
    uid: str = Depends(get_uid),
):
    """Bu dosya için geçmiş analizleri listele (en yeni önce)."""
    records = await analysis_repository.list_by_file(uid, project_id, file_id)
    items = [
        AnalysisListItemDto(
            analysis_id=r["analysis_id"],
            status=r.get("status", "unknown"),
            created_at=r.get("created_at") or datetime.utcnow(),
            duration_ms=r.get("duration_ms"),
            summary=_summary_or_none(r.get("summary")),
        )
        for r in records
    ]
    return BusinessLogicDto(success=True, data=items)


# --------------------------------------------------------------- GET status
@router.get(
    "/analyses/{analysis_id}",
    response_model=BusinessLogicDto[AnalysisStatusDto],
)
async def get_analysis_status(
    project_id: str,
    file_id: str,
    analysis_id: str,
    uid: str = Depends(get_uid),
):
    record = await _require(uid, project_id, file_id, analysis_id)
    return BusinessLogicDto(success=True, data=_to_status_dto(record))


# ---------------------------------------------------------- GET summary
@router.get(
    "/analyses/{analysis_id}/summary",
    response_model=BusinessLogicDto[ModelSummaryDto],
)
async def get_analysis_summary(
    project_id: str,
    file_id: str,
    analysis_id: str,
    uid: str = Depends(get_uid),
):
    record = await _require(uid, project_id, file_id, analysis_id)
    summary = _summary_or_none(record.get("summary"))
    if summary is None:
        raise HTTPException(status_code=404, detail="Özet bulunamadı")
    return BusinessLogicDto(success=True, data=summary)


# ---------------------------------------------------- GET displacements
@router.get(
    "/analyses/{analysis_id}/displacements",
    response_model=BusinessLogicDto[list[NodeDisplacementDto]],
)
async def get_displacements(
    project_id: str,
    file_id: str,
    analysis_id: str,
    load_case: str | None = Query(None, description="Yalnızca bu yük durumu"),
    uid: str = Depends(get_uid),
):
    record = await _require(uid, project_id, file_id, analysis_id)
    out: list[NodeDisplacementDto] = []
    for case_id, case_data in (record.get("cases") or {}).items():
        if load_case and case_id != load_case:
            continue
        for d in case_data.get("displacements", []):
            out.append(NodeDisplacementDto(**d))
    return BusinessLogicDto(success=True, data=out)


# ---------------------------------------------------------- GET modes
@router.get(
    "/analyses/{analysis_id}/modes",
    response_model=BusinessLogicDto[list[ModeDto]],
)
async def get_modes(
    project_id: str,
    file_id: str,
    analysis_id: str,
    uid: str = Depends(get_uid),
):
    record = await _require(uid, project_id, file_id, analysis_id)
    modes = record.get("modes") or []
    return BusinessLogicDto(
        success=True,
        data=[
            ModeDto(
                mode_no=m["mode_no"],
                period=m["period"],
                frequency=m["frequency"],
                angular_frequency=m["angular_frequency"],
            )
            for m in modes
        ],
    )


# ------------------------------------------------------- GET reactions
@router.get(
    "/analyses/{analysis_id}/reactions",
    response_model=BusinessLogicDto[list[ReactionDto]],
)
async def get_reactions(
    project_id: str,
    file_id: str,
    analysis_id: str,
    load_case: str | None = Query(None),
    uid: str = Depends(get_uid),
):
    record = await _require(uid, project_id, file_id, analysis_id)
    out: list[ReactionDto] = []
    for case_id, case_data in (record.get("cases") or {}).items():
        if load_case and case_id != load_case:
            continue
        for r in case_data.get("reactions", []):
            out.append(ReactionDto(**r))
    return BusinessLogicDto(success=True, data=out)


# --------------------------------------------------------------- DELETE
@router.delete(
    "/analyses/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_analysis(
    project_id: str,
    file_id: str,
    analysis_id: str,
    uid: str = Depends(get_uid),
):
    await _require(uid, project_id, file_id, analysis_id)
    await analysis_repository.delete(uid, project_id, file_id, analysis_id)


# --------------------------------------------------------------- helpers
async def _require(uid: str, project_id: str, file_id: str, analysis_id: str) -> dict:
    record = await analysis_repository.get(uid, project_id, file_id, analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    return record


def _to_status_dto(
    record: dict, request: AnalyzeRequestDto | None = None
) -> AnalysisStatusDto:
    from models.analysis_dto import AnalysisOptionsDto
    options = (
        request.options if request is not None
        else AnalysisOptionsDto(**(record.get("options") or {}))
    )
    return AnalysisStatusDto(
        analysis_id=record["analysis_id"],
        file_id=record["file_id"],
        project_id=record["project_id"],
        status=record.get("status", "unknown"),
        created_at=record.get("created_at") or datetime.utcnow(),
        completed_at=record.get("completed_at"),
        duration_ms=record.get("duration_ms"),
        options=options,
        summary=_summary_or_none(record.get("summary")),
        warnings=record.get("warnings") or [],
        error=record.get("error"),
    )


def _summary_or_none(s: dict | None) -> ModelSummaryDto | None:
    if not s:
        return None
    try:
        return ModelSummaryDto(**s)
    except Exception:
        return None
