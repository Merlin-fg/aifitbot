"""对话路由——多会话管理 + RAG 问答 + 消息持久化 + 流式输出。"""

import json
import html
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request, Form, Query
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session as DbSession

from src.database import get_session
from src.middleware.auth_middleware import get_current_user
from src.models.user import User
from src.repositories.vector_repo import VectorRepository
from src.repositories.session_repo import SessionRepository
from src.repositories.message_repo import MessageRepository
from src.services.rag_service import RAGService
from src.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["对话"])

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# 内存中暂存待生成的流式任务
_stream_tasks: dict[str, dict] = {}


def get_rag_service() -> RAGService:
    """依赖注入：RAGService。"""
    from src.bot import get_llm
    from src.rag import DashScopeEmbeddings
    llm = get_llm()
    embedding = DashScopeEmbeddings()
    vector_repo = VectorRepository(embedding=embedding)
    return RAGService(llm, vector_repo)


# ================================================================
# 会话管理 API
# ================================================================

@router.post("/session/new")
def session_new(
    title: str = Form(default="新对话"),
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
):
    """创建新会话，返回会话 ID。"""
    repo = SessionRepository(db)
    sess = repo.create(user.id, title)
    return {"session_id": sess.id, "title": sess.title}


@router.get("/session/list")
async def session_list(
    request: Request,
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
):
    """会话列表（HTML 片段）。"""
    repo = SessionRepository(db)
    sessions = repo.get_by_user(user.id)
    return templates.TemplateResponse(
        request, "_session_list.html",
        {"request": request, "user": user, "sessions": sessions}
    )


@router.put("/session/rename")
def session_rename(
    session_id: int = Form(...),
    title: str = Form(...),
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
):
    """重命名会话。"""
    repo = SessionRepository(db)
    sess = repo.get_by_id(session_id)
    if not sess or sess.user_id != user.id:
        return {"error": "会话不存在"}
    repo.update_title(session_id, title)
    return {"ok": True}


@router.delete("/session/delete")
def session_delete(
    session_id: int = Form(...),
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
):
    """删除会话及其所有消息。"""
    repo = SessionRepository(db)
    sess = repo.get_by_id(session_id)
    if not sess or sess.user_id != user.id:
        return {"error": "会话不存在"}
    # 先删消息
    msg_repo = MessageRepository(db)
    msg_repo.delete_by_session(session_id)
    # 再删会话
    repo.delete(session_id)
    return {"ok": True}


# ================================================================
# 对话消息 API
# ================================================================

@router.get("/messages/{session_id}")
async def load_messages(
    session_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
):
    """加载某会话的历史消息（HTML 片段）。"""
    sess_repo = SessionRepository(db)
    sess = sess_repo.get_by_id(session_id)
    if not sess or sess.user_id != user.id:
        return templates.TemplateResponse(
            request, "_chat_empty.html",
            {"request": request, "user": user}
        )

    msg_repo = MessageRepository(db)
    messages = msg_repo.get_by_session(session_id)

    # 在 Python 端渲染消息 HTML
    parts = []
    for msg in messages:
        if msg.role == "user":
            parts.append(f'<div class="message-user">{html.escape(msg.content)}</div>')
        else:
            refs_html = ""
            if msg.references:
                try:
                    refs = json.loads(msg.references)
                    ref_items = []
                    for i, ref in enumerate(refs, 1):
                        ref_items.append(
                            f'<div class="text-xs mb-1">'
                            f'<span class="font-medium">[{i}] {html.escape(ref["source"])}</span> '
                            f'<span class="text-gray-400">(相似度: {ref["score"]})</span>'
                            f'<p class="text-gray-500 mt-0.5">{html.escape(ref["content"])}...</p>'
                            f'</div>'
                        )
                    refs_html = (
                        '<details class="references-box mt-2"><summary class="cursor-pointer font-medium">'
                        f'📚 引用来源（{len(refs)} 条）</summary>'
                        f'{"".join(ref_items)}</details>'
                    )
                except Exception:
                    pass
            escaped_content = html.escape(msg.content).replace("\n", "<br>")
            parts.append(f'<div class="message-assistant">{escaped_content}{refs_html}</div>')

    html_content = "".join(parts)
    return templates.TemplateResponse(
        request, "_chat_messages.html",
        {"request": request, "user": user, "html_content": html_content}
    )


@router.post("/send")
async def chat_send(
    request: Request,
    question: str = Form(...),
    session_id: int = Form(default=0),
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
    rag: RAGService = Depends(get_rag_service),
):
    """发送消息 → RAG 检索 → 保存 → 返回 HTML 片段。"""
    safe_question = html.escape(question)

    # 获取或创建会话
    sess_repo = SessionRepository(db)
    if session_id:
        sess = sess_repo.get_by_id(session_id)
        if not sess or sess.user_id != user.id:
            sess = sess_repo.create(user.id, "新对话")
            session_id = sess.id
    else:
        sess = sess_repo.create(user.id, question[:30] if len(question) > 30 else question)
        session_id = sess.id

    # 保存用户消息
    msg_repo = MessageRepository(db)
    msg_repo.create(session_id, "user", question)

    # 加载历史构建多轮对话上下文
    history_msgs = msg_repo.get_by_session(session_id)
    history_text = ""
    for m in history_msgs[:-1]:
        role_label = "用户" if m.role == "user" else "教练"
        history_text += f"{role_label}：{m.content}\n"
    full_question = f"对话历史：\n{history_text}\n用户最新问题：{question}" if history_text.strip() else question

    # 调用 RAG
    result = rag.query(full_question)
    answer = result["answer"]
    references = result["references"]

    # 保存 AI 回复
    refs_json = json.dumps(references, ensure_ascii=False) if references else None
    msg_repo.create(session_id, "assistant", answer, refs_json)

    # 更新会话标题（取前30字）和更新时间
    if sess.title == "新对话":
        new_title = question[:30] + ("..." if len(question) > 30 else "")
        sess_repo.update_title(session_id, new_title)
    else:
        sess.updated_at = datetime.now(timezone.utc)
        db.add(sess)
        db.commit()

    # 构建引用 HTML
    refs_html = ""
    if references:
        ref_items = []
        for i, ref in enumerate(references, 1):
            ref_items.append(
                f'<div class="text-xs mb-1">'
                f'<span class="font-medium">[{i}] {html.escape(ref["source"])}</span> '
                f'<span class="text-gray-400">(相似度: {ref["score"]})</span>'
                f'<p class="text-gray-500 mt-0.5">{html.escape(ref["content"])}...</p>'
                f'</div>'
            )
        refs_html = (
            '<details class="references-box mt-2"><summary class="cursor-pointer font-medium">'
            f'📚 引用来源（{len(references)} 条）</summary>'
            f'{"".join(ref_items)}</details>'
        )

    escaped_answer = html.escape(answer).replace("\n", "<br>")

    return f"""
<div class="message-user">{safe_question}</div>
<div class="message-assistant">
    {escaped_answer}
    {refs_html}
</div>
"""


# ================================================================
# 流式输出（SSE）
# ================================================================

@router.post("/stream")
async def chat_stream(
    question: str = Form(...),
    session_id: int = Form(default=0),
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
    rag: RAGService = Depends(get_rag_service),
):
    """流式 RAG 问答——返回 SSE 事件流。"""
    # 获取或创建会话
    sess_repo = SessionRepository(db)
    if session_id:
        sess = sess_repo.get_by_id(session_id)
        if not sess or sess.user_id != user.id:
            sess = sess_repo.create(user.id, "新对话")
            session_id = sess.id
    else:
        sess = sess_repo.create(user.id, question[:30] if len(question) > 30 else question)
        session_id = sess.id

    # 保存用户消息
    msg_repo = MessageRepository(db)
    msg_repo.create(session_id, "user", question)

    # 加载当前会话的历史消息（用于多轮对话上下文）
    history_msgs = msg_repo.get_by_session(session_id)
    history_text = ""
    for m in history_msgs[:-1]:  # 排除刚保存的用户消息本身
        role_label = "用户" if m.role == "user" else "教练"
        history_text += f"{role_label}：{m.content}\n"

    # 构建带历史的查询
    if history_text.strip():
        full_question = f"对话历史：\n{history_text}\n用户最新问题：{question}"
    else:
        full_question = question

    # 更新标题
    if sess.title == "新对话":
        new_title = question[:30] + ("..." if len(question) > 30 else "")
        sess_repo.update_title(session_id, new_title)

    # 先检索（使用带历史的完整查询）
    retriever = rag.vector_repo.as_retriever(k=3)
    try:
        docs_with_scores = rag.vector_repo._get_store().similarity_search_with_score(full_question, k=3)
    except Exception:
        docs_with_scores = []

    references = []
    for doc, score in docs_with_scores:
        references.append({
            "source": doc.metadata.get("source", "未知"),
            "content": doc.page_content[:300],
            "score": round(float(score), 4),
        })

    # 构建知识库上下文
    context_parts = []
    for d, _ in docs_with_scores:
        context_parts.append(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
        )
    kb_context = "\n\n---\n\n".join(context_parts)
    if not kb_context:
        kb_context = "（知识库中暂无相关内容）"

    # 获取 LLM 和 prompt
    from src.bot import get_llm
    from src.services.rag_service import SYSTEM_PROMPT
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    async def generate():
        """SSE 事件生成器。"""
        full_answer = ""

        # 发送 session_id
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        try:
            chain = prompt | llm
            async for chunk in chain.astream({"context": kb_context, "input": full_question}):
                token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                    await asyncio.sleep(0.01)

            # 发送引用
            yield f"data: {json.dumps({'type': 'references', 'refs': references})}\n\n"

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

        finally:
            # 无论如何（正常结束/中断/报错）都保存已生成的内容
            if full_answer.strip():
                try:
                    refs_json = json.dumps(references, ensure_ascii=False) if references else None
                    msg_repo.create(session_id, "assistant", full_answer, refs_json)
                    sess.updated_at = datetime.now(timezone.utc)
                    db.add(sess)
                    db.commit()
                    logger.info(f"已保存消息 ({len(full_answer)} 字)")
                except Exception as e:
                    logger.error(f"保存消息失败: {e}")

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
