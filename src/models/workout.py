"""训练打卡模型——训练计划、动作、打卡记录。"""

from datetime import datetime, timezone, date
from typing import Optional

from sqlmodel import SQLModel, Field


class WorkoutPlan(SQLModel, table=True):
    """训练计划表。"""
    __tablename__ = "workout_plans"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    name: str = Field(max_length=100)                    # 计划名称
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class WorkoutExercise(SQLModel, table=True):
    """训练动作表（属于某个计划）。"""
    __tablename__ = "workout_exercises"

    id: Optional[int] = Field(default=None, primary_key=True)
    plan_id: int = Field(index=True, foreign_key="workout_plans.id")
    name: str = Field(max_length=100)                    # 动作名称
    sets: int = Field(default=3)                         # 目标组数
    reps: str = Field(default="8-12", max_length=20)     # 目标次数
    rest_sec: int = Field(default=90)                    # 组间休息(秒)
    notes: str = Field(default="", max_length=200)       # 动作要点
    order: int = Field(default=0)                        # 排序
    completed: bool = Field(default=False)               # 今日是否完成


class CheckIn(SQLModel, table=True):
    """打卡记录表。"""
    __tablename__ = "checkins"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    exercise_id: int = Field(index=True, foreign_key="workout_exercises.id")
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    sets_done: int = Field(default=0)                    # 实际完成组数
    notes: str = Field(default="", max_length=200)       # 备注
