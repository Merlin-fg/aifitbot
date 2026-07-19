"""消息模型（单条对话记录）。"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field


class Message(SQLModel, table=True):
    """对话消息，属于某个会话。"""
    __tablename__ = "messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(index=True, foreign_key="sessions.id")
    role: str = Field(max_length=20)   # "user" 或 "assistant"
    content: str = Field()             # 消息正文
    references: Optional[str] = Field(default=None)  # RAG 引用来源（JSON）
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
