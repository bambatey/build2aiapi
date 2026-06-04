"""CAD Export Router — ModelDTO → DXF kalıp planı.

POST /api/projects/{pid}/files/{fid}/export/dxf
  body: ProjectInfoDto (opsiyonel — boş POST default antet ile çalışır)
  return: application/dxf octet-stream, multi-sheet kalıp planı

Faz 1 (Geometri-only). Faz 2'de aynı endpoint donatı parametreleri ile
genişletilecek (?include_reinforcement=true).
"""

from __future__ import annotations

import io
import logging
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dependencies import get_uid
from repositories import file_repository
from services import storage_service
from services.cad_export import ProjectInfo, export_model_to_dxf
from services.structural_analysis.parser import parse_s2k

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/projects/{project_id}/files/{file_id}",
    tags=["cad_export"],
)


# ---------------------------------------------------------------- request body
class ProjectInfoDto(BaseModel):
    """Frontend formundan gelen antet meta verisi. Tüm alanlar opsiyonel."""

    project_name: str = ""
    sheet_kind: str = "KALIP PLANI"
    scale: str = "1/50"
    drawing_date: date | None = None

    city: str = ""
    district: str = ""
    municipality: str = ""
    neighborhood: str = ""
    ada: str = ""
    parsel: str = ""

    engineer_name: str = ""
    engineer_chamber_no: str = ""
    engineer_itb_no: str = ""

    building_importance: str = ""
    building_behavior_R: str = ""
    seismic_zone: str = ""
    soil_class: str = ""
    concrete_class: str = "C30/37"
    steel_class: str = "B500C"

    firm_name: str = ""
    sheet_size: str = "A3"


def _dto_to_info(dto: ProjectInfoDto | None, fallback_name: str) -> ProjectInfo:
    """ProjectInfoDto (request) → ProjectInfo (cad_export iç tipi).

    Boş gelen alanlar default'a düşer; project_name boşsa dosya adından
    fallback alır.
    """
    if dto is None:
        return ProjectInfo(project_name=fallback_name or "İsimsiz Proje")
    payload = dto.model_dump(exclude_unset=False)
    # Boş project_name fallback
    if not payload.get("project_name"):
        payload["project_name"] = fallback_name or "İsimsiz Proje"
    # drawing_date None gelirse default = today
    if payload.get("drawing_date") is None:
        payload.pop("drawing_date", None)
    return ProjectInfo(**payload)


# -------------------------------------------------------------- POST endpoint
@router.post(
    "/export/dxf",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/dxf": {}, "image/vnd.dxf": {}},
            "description": "Multi-sheet kalıp planı DXF (R2018).",
        },
    },
)
async def export_dxf(
    project_id: str,
    file_id: str,
    info: ProjectInfoDto | None = None,
    uid: str = Depends(get_uid),
):
    """Aktif .s2k dosyasından multi-sheet kalıp planı DXF üret.

    Antet meta verisi opsiyoneldir — boş POST gönderilirse stub değerlerle
    çıkar (frontend antet formunu sonradan doldurabilir).
    """
    file_meta = await file_repository.get(uid, project_id, file_id)
    if not file_meta:
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")
    storage_path = file_meta.get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=400, detail="Dosya içeriği mevcut değil")

    try:
        s2k_text = await storage_service.download_file(storage_path)
        model = parse_s2k(s2k_text)
    except Exception as exc:
        logger.exception("DXF export — parse hatası")
        raise HTTPException(status_code=400, detail=f"Parse hatası: {exc}") from exc

    fallback_name = (file_meta.get("name") or "").rsplit(".", 1)[0]
    project_info = _dto_to_info(info, fallback_name)

    try:
        result = export_model_to_dxf(model, project_info)
    except Exception as exc:
        logger.exception("DXF export — çizim hatası")
        raise HTTPException(status_code=500, detail=f"DXF üretim hatası: {exc}") from exc

    filename = f"{fallback_name or 'kalip-plani'}.dxf"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Sheet-Count": str(result.sheet_count),
    }
    return StreamingResponse(
        io.BytesIO(result.dxf_bytes),
        media_type="application/dxf",
        headers=headers,
    )
