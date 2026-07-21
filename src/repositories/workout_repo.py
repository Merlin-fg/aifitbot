"""训练打卡数据访问层。"""

from datetime import datetime, date, timedelta, timezone
from typing import Optional, List

from sqlmodel import Session, select, func

from src.models.workout import WorkoutPlan, WorkoutExercise, CheckIn


class WorkoutRepository:
    """打卡数据 CRUD + 统计。"""

    def __init__(self, session: Session):
        self.session = session

    # ================================================================
    # 计划
    # ================================================================

    def create_plan(self, user_id: int, name: str) -> WorkoutPlan:
        plan = WorkoutPlan(user_id=user_id, name=name)
        self.session.add(plan)
        self.session.commit()
        self.session.refresh(plan)
        return plan

    def get_latest_plan(self, user_id: int) -> Optional[WorkoutPlan]:
        return self.session.exec(
            select(WorkoutPlan)
            .where(WorkoutPlan.user_id == user_id)
            .order_by(WorkoutPlan.created_at.desc())
        ).first()

    def get_plan(self, plan_id: int) -> Optional[WorkoutPlan]:
        return self.session.get(WorkoutPlan, plan_id)

    # ================================================================
    # 动作
    # ================================================================

    def add_exercise(self, plan_id: int, name: str, sets: int,
                     reps: str, rest_sec: int, notes: str = "",
                     order: int = 0) -> WorkoutExercise:
        ex = WorkoutExercise(
            plan_id=plan_id, name=name, sets=sets,
            reps=reps, rest_sec=rest_sec, notes=notes, order=order
        )
        self.session.add(ex)
        self.session.commit()
        self.session.refresh(ex)
        return ex

    def get_exercises(self, plan_id: int) -> List[WorkoutExercise]:
        return list(self.session.exec(
            select(WorkoutExercise)
            .where(WorkoutExercise.plan_id == plan_id)
            .order_by(WorkoutExercise.order.asc())
        ).all())

    def get_exercise(self, exercise_id: int) -> Optional[WorkoutExercise]:
        return self.session.get(WorkoutExercise, exercise_id)

    def toggle_complete(self, exercise_id: int, user_id: int) -> bool:
        ex = self.get_exercise(exercise_id)
        if not ex:
            return False
        # 验证 exercise 所属 plan 的 owner
        plan = self.get_plan(ex.plan_id)
        if not plan or plan.user_id != user_id:
            return False
        if ex.completed:
            ex.completed = False
            ci = self.session.exec(
                select(CheckIn).where(
                    CheckIn.exercise_id == exercise_id,
                    CheckIn.user_id == user_id,
                )
            ).first()
            if ci:
                self.session.delete(ci)
        else:
            ex.completed = True
            ci = CheckIn(user_id=user_id, exercise_id=exercise_id, sets_done=ex.sets)
            self.session.add(ci)
        self.session.add(ex)
        self.session.commit()
        return ex.completed

    def get_today_exercises(self, user_id: int) -> List[WorkoutExercise]:
        """获取今日训练动作（从最新计划，忽略日期只看是否完成）。"""
        plan = self.get_latest_plan(user_id)
        if not plan:
            return []
        return self.get_exercises(plan.id)

    def reset_daily(self, user_id: int):
        """重置当日完成状态（新的一天开始）。"""
        plan = self.get_latest_plan(user_id)
        if plan:
            for ex in self.get_exercises(plan.id):
                ex.completed = False
                self.session.add(ex)
            self.session.commit()

    # ================================================================
    # 统计
    # ================================================================

    def get_checkin_dates(self, user_id: int, days: int = 90) -> List[date]:
        """获取最近 N 天有打卡的日期列表。"""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        results = self.session.exec(
            select(CheckIn.completed_at)
            .where(CheckIn.user_id == user_id, CheckIn.completed_at >= since)
        ).all()
        return list(set(d.date() for d in results))

    def get_weekly_stats(self, user_id: int) -> List[dict]:
        """获取近 4 周每周总训练组数。"""
        stats = []
        for week_offset in range(3, -1, -1):
            end = datetime.now(timezone.utc) - timedelta(weeks=week_offset)
            start = end - timedelta(weeks=1)
            count = self.session.exec(
                select(func.count(CheckIn.id))
                .where(
                    CheckIn.user_id == user_id,
                    CheckIn.completed_at >= start,
                    CheckIn.completed_at < end,
                )
            ).one()
            stats.append({
                "week": f"W{4-week_offset}",
                "sets": count,
            })
        return stats

    def get_body_part_distribution(self, user_id: int) -> List[dict]:
        """获取本周训练部位分布（从打卡记录的动作名称推断）。"""
        since = datetime.now(timezone.utc) - timedelta(days=7)
        results = self.session.exec(
            select(WorkoutExercise.name)
            .join(CheckIn, CheckIn.exercise_id == WorkoutExercise.id)
            .where(CheckIn.user_id == user_id, CheckIn.completed_at >= since)
        ).all()

        body_parts = {"胸部": 0, "背部": 0, "肩部": 0, "手臂": 0, "臀腿": 0, "核心": 0, "其他": 0}
        keywords = {
            "胸部": ["卧推", "飞鸟", "夹胸", "俯卧撑", "胸"],
            "背部": ["引体", "划船", "下拉", "硬拉", "面拉", "背"],
            "肩部": ["推举", "侧平举", "前平举", "肩推", "阿诺德", "肩"],
            "手臂": ["弯举", "下压", "臂屈伸", "锤式", "二头", "三头", "臂"],
            "臀腿": ["深蹲", "弓箭步", "腿举", "弯举", "提踵", "臀", "腿"],
            "核心": ["卷腹", "平板支撑", "举腿", "转体", "腹", "核心"],
        }
        for name in results:
            found = False
            for part, kws in keywords.items():
                if any(kw in name for kw in kws):
                    body_parts[part] += 1
                    found = True
                    break
            if not found:
                body_parts["其他"] += 1

        return [{"part": k, "count": v} for k, v in body_parts.items() if v > 0]
