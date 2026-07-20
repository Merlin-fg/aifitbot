"""训练打卡路由。"""

from pathlib import Path

from fastapi import APIRouter, Depends, Request, Form
from fastapi.templating import Jinja2Templates
from sqlmodel import Session as DbSession

from src.database import get_session
from src.middleware.auth_middleware import get_current_user
from src.models.user import User
from src.services.workout_service import WorkoutService

router = APIRouter(tags=["打卡"])

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/workout")
def page_workout(
    request: Request,
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
):
    """打卡主页。"""
    ws = WorkoutService(db)
    today = ws.get_today_data(user.id)
    stats = ws.get_stats(user.id)
    return templates.TemplateResponse(
        request, "workout.html",
        {
            "request": request,
            "user": user,
            "today": today,
            "stats": stats,
        }
    )


@router.post("/api/workout/toggle")
def api_toggle(
    request: Request,
    exercise_id: int = Form(...),
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
):
    """切换动作完成状态 + 打卡记录。"""
    ws = WorkoutService(db)
    ws.repo.toggle_complete(exercise_id, user.id)
    today = ws.get_today_data(user.id)
    return templates.TemplateResponse(
        request=request, name="_workout_list.html",
        context={"request": request, "user": user, "today": today}
    )
