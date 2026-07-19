"""用户模型。"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import SQLModel, Field


class UserRole(str, Enum):
    """用户角色：admin 可管理知识库，user 仅可问答。"""
    ADMIN = "admin"
    USER = "user"


class User(SQLModel, table=True):
    """系统用户。"""
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=50)
    hashed_password: str = Field(max_length=128)
    role: UserRole = Field(default=UserRole.USER)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
