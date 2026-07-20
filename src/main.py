"""AIFitBot FastAPI 应用入口。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from src.config import APP_TITLE, APP_VERSION, ADMIN_USERNAME, ADMIN_PASSWORD, ACCESS_TOKEN_EXPIRE_MINUTES
from src.database import engine, init_db, get_session
from src.models.user import User, UserRole
from src.models.message import Message
from src.models.document import Document
from src.models.session import Session as SessionModel
from src.middleware.auth_middleware import get_optional_user, get_current_user, require_admin
from src.services.auth_service import AuthService
from src.routes import auth_routes, kb_routes, chat_routes
from src.repositories.user_repo import UserRepository
from src.repositories.document_repo import DocumentRepository
from src.utils.logger import logger
from sqlmodel import select, func


# ============================================================
# 应用生命周期
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化数据库并创建默认管理员。"""
    logger.info(f"正在启动 {APP_TITLE} v{APP_VERSION} ...")
    init_db()

    # 确保管理员账号存在（防多进程竞态）
    with Session(engine) as session:
        auth = AuthService(session)
        admin = auth.repo.get_by_username(ADMIN_USERNAME)
        if not admin:
            try:
                auth.repo.create(
                    username=ADMIN_USERNAME,
                    hashed_password=auth.hash_password(ADMIN_PASSWORD),
                    role=UserRole.ADMIN,
                )
                logger.info(f"已创建默认管理员: {ADMIN_USERNAME}")
            except Exception as e:
                if "UNIQUE constraint" in str(e) or "duplicate" in str(e).lower():
                    logger.info(f"管理员账号已存在（由其他进程创建）: {ADMIN_USERNAME}")
                else:
                    logger.error(f"创建管理员失败: {e}")
                    raise
        else:
            logger.info(f"管理员账号已存在: {ADMIN_USERNAME}")

    logger.info(f"启动完成，访问 http://localhost:8000")
    yield
    logger.info(f"{APP_TITLE} 已关闭")


# ============================================================
# 创建应用
# ============================================================
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    lifespan=lifespan,
)

# 静态文件
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Jinja2 模板
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# 注册 API 路由
app.include_router(auth_routes.router)
app.include_router(kb_routes.api_router)
app.include_router(chat_routes.router)


# ============================================================
# 页面路由
# ============================================================

@app.get("/")
def page_home(request: Request, user: User | None = Depends(get_optional_user)):
    """首页。"""
    return templates.TemplateResponse(request, "home.html", {"request": request, "user": user})


@app.get("/auth/login")
def page_login(request: Request, user: User | None = Depends(get_optional_user)):
    """登录页面。"""
    if user:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"request": request, "user": user})


@app.get("/auth/register")
def page_register(request: Request, user: User | None = Depends(get_optional_user)):
    """注册页面。"""
    if user:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "register.html", {"request": request, "user": user})


@app.post("/auth/login")
async def handle_login(
    request: Request,
    session: Session = Depends(get_session),
):
    """处理登录表单，成功后跳转到 /chat。"""
    form = await request.form()
    username = form.get("username", "").strip()
    password = form.get("password", "")

    auth = AuthService(session)
    ok, token, msg = auth.login(username, password)

    if not ok:
        return templates.TemplateResponse(
            request, "login.html", {"request": request, "user": None, "error": msg}
        )

    import os
    response = RedirectResponse("/chat", status_code=303)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=os.getenv("APP_ENV") == "production",
    )
    return response


@app.post("/auth/register")
async def handle_register(
    request: Request,
    session: Session = Depends(get_session),
):
    """处理注册表单，成功后跳转到登录页。"""
    form = await request.form()
    username = form.get("username", "").strip()
    password = form.get("password", "")
    confirm_password = form.get("confirm_password", "")

    if len(username) < 2:
        return templates.TemplateResponse(
            request, "register.html", {"request": request, "user": None, "error": "用户名至少 2 个字符"}
        )
    if len(password) < 6:
        return templates.TemplateResponse(
            request, "register.html", {"request": request, "user": None, "error": "密码至少 6 个字符"}
        )
    if password != confirm_password:
        return templates.TemplateResponse(
            request, "register.html", {"request": request, "user": None, "error": "两次输入的密码不一致"}
        )

    auth = AuthService(session)
    ok, msg = auth.register(username, password)

    if not ok:
        return templates.TemplateResponse(
            request, "register.html", {"request": request, "user": None, "error": msg}
        )

    return templates.TemplateResponse(
        request, "login.html", {"request": request, "user": None, "msg": "注册成功，请登录"}
    )


@app.get("/auth/logout")
def handle_logout():
    """退出登录，清除 Cookie 并返回首页。"""
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("access_token")
    return response


# ============================================================
# 对话页面（需登录）
# ============================================================

@app.get("/chat")
def page_chat(request: Request, user: User = Depends(get_current_user)):
    """对话主界面。"""
    return templates.TemplateResponse(request, "chat.html", {"request": request, "user": user})


# ============================================================
# 修改密码页面（需登录）
# ============================================================

@app.get("/auth/password")
def page_change_password(request: Request, user: User = Depends(get_current_user)):
    """修改密码页面。"""
    return templates.TemplateResponse(request, "change_password.html", {"request": request, "user": user})


@app.post("/auth/password")
async def handle_change_password(
    request: Request,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """处理修改密码表单。"""
    form = await request.form()
    old_password = form.get("old_password", "")
    new_password = form.get("new_password", "")
    confirm_password = form.get("confirm_password", "")

    if len(new_password) < 6:
        return templates.TemplateResponse(
            request, "change_password.html",
            {"request": request, "user": user, "error": "新密码至少 6 个字符"}
        )
    if new_password != confirm_password:
        return templates.TemplateResponse(
            request, "change_password.html",
            {"request": request, "user": user, "error": "两次输入的密码不一致"}
        )

    auth = AuthService(session)
    ok, msg = auth.change_password(user.id, old_password, new_password)

    if not ok:
        return templates.TemplateResponse(
            request, "change_password.html",
            {"request": request, "user": user, "error": msg}
        )

    return templates.TemplateResponse(
        request, "change_password.html",
        {"request": request, "user": user, "msg": "密码修改成功"}
    )


# ============================================================
# 知识库管理页面（仅管理员）
# ============================================================

@app.get("/admin/kb")
async def page_admin_kb(request: Request, user: User = Depends(get_current_user)):
    """知识库管理页面。"""
    require_admin(user)
    return templates.TemplateResponse(request, "admin_kb.html", {"request": request, "user": user})


@app.get("/admin/dashboard")
def page_admin_dashboard(
    request: Request,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """管理员仪表盘。"""
    require_admin(user)

    user_repo = UserRepository(session)
    doc_repo = DocumentRepository(session)

    stats = {
        "user_count": user_repo.user_count(),
        "doc_count": session.exec(select(func.count(Document.id))).one(),
        "session_count": session.exec(select(func.count(SessionModel.id))).one(),
        "message_count": session.exec(select(func.count(Message.id))).one(),
    }

    recent_users = user_repo.get_recent(5)

    return templates.TemplateResponse(
        request, "admin_dashboard.html",
        {"request": request, "user": user, "stats": stats, "recent_users": recent_users}
    )


# ============================================================
# 健康检查
# ============================================================
@app.get("/api/health")
def health_check():
    """健康检查端点。"""
    return {"status": "ok", "version": APP_VERSION}
