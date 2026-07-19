"""文档元数据模型。"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field


class Document(SQLModel, table=True):
    """知识库文档元数据。实际文件存储在 data/ 目录。"""
    __tablename__ = "documents"

    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(max_length=255)                    # 原始文件名
    stored_name: str = Field(max_length=255, unique=True)    # 实际存储的文件名（UUID）
    file_type: str = Field(max_length=20)                    # md / pdf / txt
    size_bytes: int = Field(default=0)                       # 文件大小
    chunk_count: int = Field(default=0)                      # 切分后的文本块数量
    status: str = Field(default="processing")                # processing / ready / error
    error_msg: Optional[str] = Field(default=None)           # 错误信息
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
