"""
Files Router
GET    /api/projects/{pid}/files                → Dosya listesi
POST   /api/projects/{pid}/files                → Dosya oluştur (JSON body)
POST   /api/projects/{pid}/files/upload         → Dosya yükle (multipart)
GET    /api/projects/{pid}/files/{fid}          → Dosya içeriği getir
PUT    /api/projects/{pid}/files/{fid}          → Dosya içeriği güncelle
DELETE /api/projects/{pid}/files/{fid}          → Dosya sil
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from dependencies import get_uid
from models.dto import (
    BusinessLogicDto,
    FileContentDto,
    FileCreateRequest,
    FileNodeDto,
    FileUpdateRequest,
)
from repositories import file_repository, project_repository
from services import storage_service

router = APIRouter(prefix="/api/projects/{project_id}/files", tags=["files"])


@router.get("", response_model=BusinessLogicDto)
async def list_files(project_id: str, uid: str = Depends(get_uid)):
    """Projedeki dosyaları listele."""
    files = await file_repository.list_by_project(uid, project_id)
    file_dtos = [
        FileNodeDto(
            id=f["id"],
            name=f["name"],
            type=f.get("type", "file"),
            path=f.get("path", ""),
            format=f.get("format"),
            size=f.get("size_bytes"),
            line_count=f.get("line_count"),
            last_modified=f.get("updated_at"),
            storage_path=f.get("storage_path"),
        )
        for f in files
    ]
    return BusinessLogicDto(success=True, data=file_dtos)


@router.post("", response_model=BusinessLogicDto, status_code=status.HTTP_201_CREATED)
async def create_file(
    project_id: str,
    request: FileCreateRequest,
    uid: str = Depends(get_uid),
):
    """Dosya oluştur (içerik JSON body'de)."""
    project = await project_repository.get(uid, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Proje bulunamadı")

    import uuid
    file_id = str(uuid.uuid4())

    content = request.content
    size_bytes = len(content.encode("utf-8"))
    line_count = content.count("\n") + 1 if content else 0

    # Firebase Storage'a yükle
    storage_path = await storage_service.upload_file(
        uid=uid,
        project_id=project_id,
        file_id=file_id,
        file_name=request.name,
        content=content,
    )

    # Firestore'a metadata yaz
    file_data = await file_repository.create(
        uid=uid,
        project_id=project_id,
        name=request.name,
        format=request.format,
        storage_path=storage_path,
        size_bytes=size_bytes,
        line_count=line_count,
    )

    await project_repository.increment_file_count(uid, project_id, 1)

    return BusinessLogicDto(
        success=True,
        data=FileNodeDto(
            id=file_data["id"],
            name=file_data["name"],
            type="file",
            path=file_data.get("path", ""),
            format=file_data.get("format"),
            size=size_bytes,
            line_count=line_count,
            last_modified=file_data.get("updated_at"),
            storage_path=storage_path,
        ),
    )


@router.post("/upload", response_model=BusinessLogicDto, status_code=status.HTTP_201_CREATED)
async def upload_file(
    project_id: str,
    file: UploadFile = File(...),
    uid: str = Depends(get_uid),
):
    """Dosya yükle (multipart form data) — .s2k, .e2k, .r3d, .std dosyaları."""
    project = await project_repository.get(uid, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Proje bulunamadı")

    # Dosya formatı kontrolü
    allowed_formats = {".s2k", ".e2k", ".r3d", ".std", ".tcl", ".inp"}
    file_ext = ""
    if file.filename:
        file_ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if file_ext not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı: {file_ext}. İzin verilenler: {allowed_formats}",
        )

    # Dosya boyutu kontrolü (10MB)
    content_bytes = await file.read()
    if len(content_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Dosya boyutu 10MB'ı aşamaz")

    content = content_bytes.decode("utf-8")
    size_bytes = len(content_bytes)
    line_count = content.count("\n") + 1

    import uuid
    file_id = str(uuid.uuid4())

    # Firebase Storage'a yükle
    storage_path = await storage_service.upload_file(
        uid=uid,
        project_id=project_id,
        file_id=file_id,
        file_name=file.filename or "unnamed",
        content=content,
    )

    # Firestore'a metadata yaz
    file_data = await file_repository.create(
        uid=uid,
        project_id=project_id,
        name=file.filename or "unnamed",
        format=file_ext,
        storage_path=storage_path,
        size_bytes=size_bytes,
        line_count=line_count,
    )

    await project_repository.increment_file_count(uid, project_id, 1)

    return BusinessLogicDto(
        success=True,
        data=FileNodeDto(
            id=file_data["id"],
            name=file_data["name"],
            type="file",
            path=file_data.get("path", ""),
            format=file_ext,
            size=size_bytes,
            line_count=line_count,
            last_modified=file_data.get("updated_at"),
            storage_path=storage_path,
        ),
    )


@router.get("/{file_id}", response_model=BusinessLogicDto)
async def get_file_content(
    project_id: str,
    file_id: str,
    uid: str = Depends(get_uid),
):
    """Dosya içeriğini getir (Firebase Storage'dan okur)."""
    file_meta = await file_repository.get(uid, project_id, file_id)
    if not file_meta:
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")

    storage_path = file_meta.get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=404, detail="Dosya içeriği bulunamadı")

    content = await storage_service.download_file(storage_path)

    return BusinessLogicDto(
        success=True,
        data=FileContentDto(
            id=file_id,
            name=file_meta["name"],
            content=content,
            size=file_meta.get("size_bytes", 0),
            line_count=file_meta.get("line_count", 0),
        ),
    )


@router.put("/{file_id}", response_model=BusinessLogicDto)
async def update_file_content(
    project_id: str,
    file_id: str,
    request: FileUpdateRequest,
    uid: str = Depends(get_uid),
):
    """Dosya içeriğini güncelle (editörden save)."""
    file_meta = await file_repository.get(uid, project_id, file_id)
    if not file_meta:
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")

    content = request.content
    size_bytes = len(content.encode("utf-8"))
    line_count = content.count("\n") + 1

    # Storage'a yeni içerik yaz
    storage_path = await storage_service.upload_file(
        uid=uid,
        project_id=project_id,
        file_id=file_id,
        file_name=file_meta["name"],
        content=content,
    )

    # Metadata güncelle
    updated = await file_repository.update_storage(
        uid=uid,
        project_id=project_id,
        file_id=file_id,
        storage_path=storage_path,
        size_bytes=size_bytes,
        line_count=line_count,
    )

    return BusinessLogicDto(
        success=True,
        data=FileContentDto(
            id=file_id,
            name=file_meta["name"],
            content=content,
            size=size_bytes,
            line_count=line_count,
        ),
    )


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    project_id: str,
    file_id: str,
    uid: str = Depends(get_uid),
):
    """Dosyayı sil (storage + firestore)."""
    file_meta = await file_repository.get(uid, project_id, file_id)
    if not file_meta:
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")

    # Storage'dan sil
    if file_meta.get("storage_path"):
        try:
            await storage_service.delete_file(file_meta["storage_path"])
        except Exception:
            pass

    # Firestore'dan sil
    await file_repository.delete(uid, project_id, file_id)
    await project_repository.increment_file_count(uid, project_id, -1)
