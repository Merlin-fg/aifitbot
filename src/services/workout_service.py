"""训练打卡服务——JSON 解析 + 计划入库 + 统计查询。"""

import json
import re
from typing import Optional

from sqlmodel import Session

from src.repositories.workout_repo import WorkoutRepository
from src.utils.logger import logger


class WorkoutService:
    """打卡业务逻辑。"""

    def __init__(self, session: Session):
        self.repo = WorkoutRepository(session)

    def parse_and_import(self, user_id: int, ai_response: str) -> Optional[str]:
        """从 AI 回答中提取 JSON 训练计划并入库。

        Returns:
            计划名称或 None（解析失败）。
        """
        # 提取 JSON 块
        plan_data = self._extract_json(ai_response)
        if not plan_data:
            return None

        try:
            exercises = plan_data.get("exercises", [])
            if not exercises:
                return None

            plan_name = plan_data.get("plan_name", "训练计划")
            plan = self.repo.create_plan(user_id, plan_name)

            for i, ex in enumerate(exercises):
                self.repo.add_exercise(
                    plan_id=plan.id,
                    name=ex.get("name", ""),
                    sets=ex.get("sets", 3),
                    reps=ex.get("reps", "8-12"),
                    rest_sec=ex.get("rest_sec", 90),
                    notes=ex.get("notes", ""),
                    order=i,
                )

            logger.info(f"打卡计划已导入: {plan_name}, {len(exercises)} 个动作")
            return plan_name
        except Exception as e:
            logger.error(f"导入打卡计划失败: {e}")
            return None

    def _extract_json(self, text: str) -> Optional[dict]:
        """从文本中提取 JSON 训练计划。"""
        # STARTJSON...ENDJSON 标记格式（v2，不限是否跨行）
        match = re.search(r'STARTJSON\s*([\s\S]*?)\s*ENDJSON', text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        return None

    def get_today_data(self, user_id: int) -> dict:
        """获取今日打卡页面所需数据。"""
        exercises = self.repo.get_today_exercises(user_id)
        total = len(exercises)
        done = sum(1 for e in exercises if e.completed)
        return {
            "exercises": exercises,
            "total": total,
            "done": done,
            "progress": round(done / total * 100) if total > 0 else 0,
        }

    def get_stats(self, user_id: int) -> dict:
        """获取统计数据。"""
        checkin_dates = self.repo.get_checkin_dates(user_id, days=90)
        weekly_sets = self.repo.get_weekly_stats(user_id)
        body_parts = self.repo.get_body_part_distribution(user_id)
        return {
            "checkin_dates": [d.isoformat() for d in checkin_dates],
            "total_checkin_days": len(checkin_dates),
            "weekly_sets": weekly_sets,
            "body_parts": body_parts,
        }
