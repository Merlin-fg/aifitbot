"""会话模型（对话分组）。"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field, Relationship


class Session(SQLModel, table=True):
    """聊天会话，每个用户可创建多个。"""
    __tablename__ = "sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="users.id")
    title: str = Field(default="新对话", max_length=100)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
