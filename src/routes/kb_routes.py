"""知识库管理路由（仅管理员）。"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from sqlmodel import Session

from src.database import get_session
from src.middleware.auth_middleware import get_current_user, require_admin
from src.models.user import User
from src.repositories.vector_repo import VectorRepository
from src.services.kb_service import KBService

# API 路由
api_router = APIRouter(prefix="/api/admin/kb", tags=["知识库管理"])

# 模板
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


def get_kb_service(session: Session = Depends(get_session)) -> KBService:
    """依赖注入：创建 KBService 实例。"""
    from src.rag import DashScopeEmbeddings
    embedding = DashScopeEmbeddings()
    vector_repo = VectorRepository(embedding=embedding)
    return KBService(session, vector_repo)


# ---- API 端点 ----

@api_router.post("/upload")
def api_upload(
    file: UploadFile = File(...),
    admin: User = Depends(require_admin),
    kb: KBService = Depends(get_kb_service),
):
    """上传文档并向量化。"""
    if not file.filename:
        raise HTTPException(400, "请选择文件")
    content = file.file.read()
    ok, msg = kb.upload_document(content, file.filename)
    if not ok:
        raise HTTPException(400, msg)
    return {"message": msg}


@api_router.get("/documents")
def api_list(
    request: Request,
    admin: User = Depends(require_admin),
    kb: KBService = Depends(get_kb_service),
):
    """文档列表（HTML 片段，供 HTMX 加载）。"""
    docs = kb.list_documents()
    return templates.TemplateResponse(
        request, "_doc_list.html", {"request": request, "user": admin, "docs": docs}
    )


@api_router.delete("/documents/{doc_id}")
def api_delete(
    doc_id: int,
    admin: User = Depends(require_admin),
    kb: KBService = Depends(get_kb_service),
):
    """删除文档。"""
    ok, msg = kb.delete_document(doc_id)
    if not ok:
        raise HTTPException(404, msg)
    return {"message": msg}


# ---- 页面路由（供 main.py 注册） ----

async def page_admin_kb(request: Request, user: User = Depends(require_admin)):
    """知识库管理页面（仅管理员可见）。"""
    return templates.TemplateResponse(request, "admin_kb.html", {"request": request, "user": user})
