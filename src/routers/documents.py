"""
Documents Router
GET /api/documents → Doküman listesi (yönetmelikler, rehberler vb.)

Frontend'deki server/api/documents.get.ts endpoint'inin karşılığı.
"""
from fastapi import APIRouter, Depends, Query

from dependencies import get_uid
from models.dto import BusinessLogicDto, DocumentDto
from services.firebase_service import firebase_service

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=BusinessLogicDto)
async def list_documents(
    document_type: str = Query(default="1"),
    uid: str = Depends(get_uid),
):
    """Dokümanları listele (yönetmelikler, rehberler vb.)."""
    docs_ref = firebase_service.db.collection("documents")

    if document_type:
        docs_ref = docs_ref.where("document_type", "==", document_type)

    docs = docs_ref.stream()
    document_dtos = []
    for doc in docs:
        data = doc.to_dict()
        document_dtos.append(
            DocumentDto(
                id=doc.id,
                title=data.get("title", ""),
                content=data.get("content"),
                document_type=data.get("document_type", "1"),
                created_at=data.get("created_at"),
            )
        )

    return BusinessLogicDto(success=True, data=document_dtos)
