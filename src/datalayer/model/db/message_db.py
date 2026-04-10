from datetime import datetime
from typing import Optional
from uuid import UUID
import uuid

from sqlalchemy import Column
from sqlmodel import Field, SQLModel
from sqlalchemy.dialects.postgresql import JSONB


class MessageDB(SQLModel, table=True):
    __tablename__ = "message"
    __table_args__ = {"schema": "public"}

    message_id: UUID = Field(description="Message ID", primary_key=True)
    message_conversation_id: UUID = Field(description="Conversation ID")
    message_role: str = Field(description="Role")
    message_content: str = Field(description="Content")
    message_metadata: Optional[dict] = Field(sa_column=Column(JSONB))
    message_created_at: datetime = Field(description="Created At")
    conversation_analysis_id: Optional[UUID] = Field(description="Conversation Analysis ID", default_factory=None)
