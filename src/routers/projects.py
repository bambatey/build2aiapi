"""
Projects Router
GET    /api/projects            → Kullanıcının tüm projeleri
POST   /api/projects            → Yeni proje oluştur
GET    /api/projects/{id}       → Proje detayı + dosya listesi
PUT    /api/projects/{id}       → Proje güncelle
DELETE /api/projects/{id}       → Proje sil
"""
from fastapi import APIRouter, Depends, HTTPException, status

from dependencies import get_uid
from models.dto import (
    BusinessLogicDto,
    FileNodeDto,
    ProjectCreateRequest,
    ProjectDetailDto,
    ProjectDto,
    ProjectUpdateRequest,
)
from repositories import file_repository, project_repository

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=BusinessLogicDto)
async def list_projects(uid: str = Depends(get_uid)):
    """Kullanıcının tüm projelerini listele."""
    projects = await project_repository.list_by_user(uid)
    project_dtos = [
        ProjectDto(
            id=p["id"],
            name=p["name"],
            format=p.get("format", ".s2k"),
            file_count=p.get("file_count", 0),
            last_modified=p.get("updated_at"),
            progress=p.get("progress", 0),
            tags=p.get("tags", []),
            created_at=p.get("created_at"),
            updated_at=p.get("updated_at"),
        )
        for p in projects
    ]
    return BusinessLogicDto(success=True, data=project_dtos)


@router.post("", response_model=BusinessLogicDto, status_code=status.HTTP_201_CREATED)
async def create_project(
    request: ProjectCreateRequest,
    uid: str = Depends(get_uid),
):
    """Yeni proje oluştur."""
    project = await project_repository.create(
        uid=uid,
        name=request.name,
        format=request.format,
        tags=request.tags,
    )
    return BusinessLogicDto(
        success=True,
        data=ProjectDto(
            id=project["id"],
            name=project["name"],
            format=project["format"],
            file_count=project.get("file_count", 0),
            progress=project.get("progress", 0),
            tags=project.get("tags", []),
            created_at=project.get("created_at"),
            updated_at=project.get("updated_at"),
        ),
    )


@router.get("/{project_id}", response_model=BusinessLogicDto)
async def get_project(project_id: str, uid: str = Depends(get_uid)):
    """Proje detayı + dosya listesi."""
    project = await project_repository.get(uid, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Proje bulunamadı")

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

    return BusinessLogicDto(
        success=True,
        data=ProjectDetailDto(
            id=project["id"],
            name=project["name"],
            format=project.get("format", ".s2k"),
            file_count=project.get("file_count", 0),
            last_modified=project.get("updated_at"),
            progress=project.get("progress", 0),
            tags=project.get("tags", []),
            created_at=project.get("created_at"),
            updated_at=project.get("updated_at"),
            files=file_dtos,
        ),
    )


@router.put("/{project_id}", response_model=BusinessLogicDto)
async def update_project(
    project_id: str,
    request: ProjectUpdateRequest,
    uid: str = Depends(get_uid),
):
    """Proje güncelle (isim, tags, progress vb.)."""
    project = await project_repository.get(uid, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Proje bulunamadı")

    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="Güncellenecek alan belirtilmedi")

    updated = await project_repository.update(uid, project_id, updates)
    return BusinessLogicDto(
        success=True,
        data=ProjectDto(
            id=updated["id"],
            name=updated["name"],
            format=updated.get("format", ".s2k"),
            file_count=updated.get("file_count", 0),
            last_modified=updated.get("updated_at"),
            progress=updated.get("progress", 0),
            tags=updated.get("tags", []),
            created_at=updated.get("created_at"),
            updated_at=updated.get("updated_at"),
        ),
    )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, uid: str = Depends(get_uid)):
    """Projeyi sil."""
    project = await project_repository.get(uid, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Proje bulunamadı")

    # Proje dosyalarını da sil (storage + firestore)
    files = await file_repository.list_by_project(uid, project_id)
    from services import storage_service
    for f in files:
        if f.get("storage_path"):
            try:
                await storage_service.delete_file(f["storage_path"])
            except Exception:
                pass  # Storage silme başarısız olsa da devam et
        await file_repository.delete(uid, project_id, f["id"])

    await project_repository.delete(uid, project_id)
