"""用户认证相关路由：注册、登录、修改密码。"""

from fastapi import APIRouter, Depends, HTTPException, status, Form
from sqlmodel import Session

from src.database import get_session
from src.services.auth_service import AuthService
from src.middleware.auth_middleware import get_current_user
from src.models.user import User

router = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post("/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    session: Session = Depends(get_session),
):
    """用户注册（表单提交）。"""
    username = username.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    if len(username) < 2:
        raise HTTPException(status_code=400, detail="用户名至少 2 个字符")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 个字符")
    if password != confirm_password:
        raise HTTPException(status_code=400, detail="两次输入的密码不一致")

    auth = AuthService(session)
    ok, msg = auth.register(username, password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    return {"message": msg}


@router.post("/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    """用户登录，返回 JWT access_token。"""
    auth = AuthService(session)
    ok, token, msg = auth.login(username, password)
    if not ok:
        raise HTTPException(status_code=401, detail=msg)

    return {"access_token": token, "token_type": "bearer", "message": msg}


@router.put("/password")
def change_password(
    old_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """修改当前登录用户的密码。"""
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码至少 6 个字符")
    if new_password != confirm_password:
        raise HTTPException(status_code=400, detail="两次输入的密码不一致")

    auth = AuthService(session)
    ok, msg = auth.change_password(user.id, old_password, new_password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    return {"message": msg}
