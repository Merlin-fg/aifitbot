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

    # ---- 用户档案（健身相关） ----
    height_cm: Optional[int] = Field(default=None)       # 身高(cm)
    weight_kg: Optional[float] = Field(default=None)      # 体重(kg)
    age: Optional[int] = Field(default=None)              # 年龄
    gender: Optional[str] = Field(default=None, max_length=10)   # 性别
    goal: Optional[str] = Field(default=None, max_length=50)     # 目标(增肌/减脂/塑形)
    equipment: Optional[str] = Field(default=None, max_length=200) # 可用器械
    injuries: Optional[str] = Field(default=None, max_length=200)  # 伤病情况

    def to_profile_text(self) -> str:
        """将档案转为注入 prompt 的文本。无档案返回空字符串。"""
        parts = []
        if self.age:        parts.append(f"{self.age}岁")
        if self.gender:     parts.append(self.gender)
        if self.height_cm:  parts.append(f"{self.height_cm}cm")
        if self.weight_kg:  parts.append(f"{self.weight_kg}kg")
        if self.goal:       parts.append(f"目标: {self.goal}")
        if self.equipment:  parts.append(f"可用器械: {self.equipment}")
        if self.injuries:   parts.append(f"伤病注意: {self.injuries}")
        return "用户档案：" + "，".join(parts) + "。" if parts else ""
