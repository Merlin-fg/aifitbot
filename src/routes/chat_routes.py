"""对话路由——多会话管理 + RAG 问答 + 消息持久化 + 流式输出 + 打卡导入。"""

import json
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session as DbSession
from langchain_core.messages import HumanMessage, AIMessage

from src.database import engine, get_session
from src.middleware.auth_middleware import get_current_user
from src.models.user import User
from src.models.session import Session as SessionModel
from src.repositories.vector_repo import VectorRepository
from src.repositories.session_repo import SessionRepository
from src.repositories.message_repo import MessageRepository
from src.services.rag_service import RAGService
from src.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["对话"])

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

def get_rag_service() -> RAGService:
    from src.bot import get_llm
    from src.rag import DashScopeEmbeddings
    llm = get_llm()
    embedding = DashScopeEmbeddings()
    vector_repo = VectorRepository(embedding=embedding)
    return RAGService(llm, vector_repo)

CONFIRM_WORDS = ["需要", "好的", "加入打卡", "可以", "行", "ok", "OK", "好"]


def try_import_workout(question: str, session_id: int, user_id: int,
                       db: DbSession) -> Optional[str]:
    """检测打卡确认词，尝试从历史 AI 回答导入训练计划。返回计划名或 None。"""
    if not any(w in question for w in CONFIRM_WORDS):
        return None
    msg_repo = MessageRepository(db)
    prev_msgs = msg_repo.get_by_session(session_id, limit=5)
    for pm in reversed(prev_msgs):
        if pm.role == "assistant" and pm.content:
            from src.services.workout_service import WorkoutService
            ws = WorkoutService(db)
            plan_name = ws.parse_and_import(user_id, pm.content)
            if plan_name:
                return plan_name
            break  # only check the most recent AI message
    return None


# ================================================================
# 会话管理 API
# ================================================================

@router.post("/session/new")
def session_new(
    title: str = Form(default="新对话"),
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
):
    repo = SessionRepository(db)
    sess = repo.create(user.id, title)
    return {"session_id": sess.id, "title": sess.title}


@router.get("/session/list")
async def session_list(
    request: Request,
    user: User = Depends(get_current_user),
    db: DbSession = Depends(get_session),
):
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
    repo = SessionRepository(db)
    sess = repo.get_by_id(session_id)
    if not sess or sess.user_id != user.id:
        return {"error": "会话不存在"}
    msg_repo = MessageRepository(db)
    msg_repo.delete_by_session(session_id)
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
    sess_repo = SessionRepository(db)
    sess = sess_repo.get_by_id(session_id)
    if not sess or sess.user_id != user.id:
        return templates.TemplateResponse(
            request, "_chat_empty.html",
            {"request": request, "user": user}
        )
    msg_repo = MessageRepository(db)
    messages = msg_repo.get_by_session(session_id)
    for msg in messages:
        if msg.references:
            try:
                msg.references = json.loads(msg.references)
            except Exception:
                msg.references = None
    return templates.TemplateResponse(
        request, "_chat_messages.html",
        {"request": request, "user": user, "messages": messages}
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

    # === 打卡确认拦截（在 RAG 之前） ===
    plan_name = try_import_workout(question, session_id, user.id, db)
    if plan_name:
        msg_repo.create(session_id, "assistant",
            f"已将「{plan_name}」加入训练打卡。点击查看打卡页面开始训练。")
        return f"""
<div class="message-user">{safe_question}</div>
<div class="message-assistant" style="text-align:center;">
    <div style="font-size:2.2rem;margin-bottom:8px;">✅</div>
    <p style="font-weight:700;">已加入训练打卡</p>
    <p style="color:var(--c-text2);font-size:var(--t-sm);">{html.escape(plan_name)}</p>
    <a href="/workout" class="btn btn-primary" style="margin-top:12px;">查看打卡</a>
</div>
"""

    # === 正常 RAG 流程 ===
    history_msgs = msg_repo.get_by_session(session_id, limit=13)
    chat_history: List = []
    for m in history_msgs[:-1]:
        if m.role == "user":
            chat_history.append(HumanMessage(content=m.content))
        else:
            chat_history.append(AIMessage(content=m.content))

    profile_text = user.to_profile_text()
    result = rag.query(question, chat_history=chat_history, profile_text=profile_text)
    answer = result["answer"]
    references = result["references"]

    refs_json = json.dumps(references, ensure_ascii=False) if references else None
    msg_repo.create(session_id, "assistant", answer, refs_json)

    if sess.title == "新对话":
        new_title = question[:30] + ("..." if len(question) > 30 else "")
        sess_repo.update_title(session_id, new_title)
    else:
        sess.updated_at = datetime.now(timezone.utc)
        db.add(sess)
        db.commit()

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

    import re as _re
    display_answer = _re.sub(r'STARTJSON[\s\S]*?ENDJSON', '', answer)
    display_answer = _re.sub(r'\[PLAN_JSON\][\s\S]*?\[/PLAN_JSON\]', '', display_answer)
    escaped_answer = html.escape(display_answer).replace("\n", "<br>")

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

    # === 打卡确认拦截（在 RAG 之前） ===
    plan_name = try_import_workout(question, session_id, user.id, db)
    if plan_name:
        msg_repo.create(session_id, "assistant",
            f"已将「{plan_name}」加入训练打卡。点击查看打卡页面开始训练。")

        async def confirm_stream():
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            for ch in f"✅ 已加入训练打卡：{plan_name}\n\n点击打卡页面开始训练吧！":
                yield f"data: {json.dumps({'type': 'token', 'text': ch})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            confirm_stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"}
        )

    # === 正常流式 RAG（使用完整 RAG 管道检索） ===
    history_msgs = msg_repo.get_by_session(session_id, limit=13)
    chat_history: List = []
    for m in history_msgs[:-1]:
        if m.role == "user":
            chat_history.append(HumanMessage(content=m.content))
        else:
            chat_history.append(AIMessage(content=m.content))

    if sess.title == "新对话":
        new_title = question[:30] + ("..." if len(question) > 30 else "")
        sess_repo.update_title(session_id, new_title)

    from src.services.rag_service import SYSTEM_PROMPT
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    llm = rag.llm
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    profile_block = f"\n\n## 用户档案\n{user.to_profile_text()}" if user.to_profile_text() else ""

    async def generate():
        import asyncio
        import time as _time
        _t_request = _time.perf_counter()
        full_answer = ""
        # Immediately send a thinking event so the browser knows the stream is alive
        yield f"data: {json.dumps({'type': 'thinking', 'text': '检索知识库中...'})}\n\n"

        # Run retrieval in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        ret = await loop.run_in_executor(None, rag.retrieve, question)
        _t_retrieval_done = _time.perf_counter()
        references = ret["references"]
        kb_context = ret["formatted_context"] or "（知识库中暂无相关内容）"
        quality = ret.get("quality", "ok")
        top_score = ret.get("top_score", 0.0)

        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        # 质量门控：拒答
        if quality == "reject":
            reject_msg = (
                "抱歉，当前知识库中暂无相关信息，建议查阅专业健身书籍或咨询持证教练。"
                "\n\n💡 提示：AI 健身教练目前的知识库主要涵盖训练动作、营养饮食、拉伸恢复和训练原理四大领域，"
                "你可以尝试换个方式提问。"
            )
            for ch in reject_msg:
                full_answer += ch
                yield f"data: {json.dumps({'type': 'token', 'text': ch})}\n\n"
            yield f"data: {json.dumps({'type': 'references', 'refs': []})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            # 保存消息
            try:
                from src.database import Session as NewDbSession
                with NewDbSession(engine) as save_db:
                    save_repo = MessageRepository(save_db)
                    save_repo.create(session_id, "assistant", full_answer, None)
                    s = save_db.get(SessionModel, session_id)
                    if s:
                        s.updated_at = datetime.now(timezone.utc)
                        save_db.add(s)
                        save_db.commit()
            except Exception as e:
                logger.error(f"保存消息失败: {e}")
            return

        try:
            chain = prompt | llm
            _t_prompt_ready = _time.perf_counter()
            _first_token = None
            async for chunk in chain.astream({
                "context": kb_context,
                "input": question,
                "chat_history": chat_history,
                "profile": profile_block,
            }):
                token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if token:
                    if _first_token is None:
                        _first_token = _time.perf_counter()
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            _t_generation_done = _time.perf_counter()
            _t_first = _first_token - _t_prompt_ready if _first_token else 0
            _t_gen = _t_generation_done - _first_token if _first_token else _t_generation_done - _t_prompt_ready
            _t_total = _t_generation_done - _t_request
            logger.info(
                f"[TIMING] generate() 分步耗时: "
                f"请求→检索完成={_t_retrieval_done-_t_request:.2f}s | "
                f"检索→Prompt就绪={_t_prompt_ready-_t_retrieval_done:.2f}s | "
                f"Prompt→首Token={_t_first:.2f}s | "
                f"首Token→生成结束={_t_gen:.2f}s ({len(full_answer)} chars) | "
                f"总计={_t_total:.2f}s"
            )

            # 弱关联时追加免责
            if quality == "weak":
                disclaimer = (
                    f"\n\n---\n⚠️ 知识库中与此问题的相关度较低（rerank score: {top_score:.2f}），"
                    "以上回答结合了有限知识片段，仅供参考。"
                )
                for ch in disclaimer:
                    full_answer += ch
                    yield f"data: {json.dumps({'type': 'token', 'text': ch})}\n\n"

            yield f"data: {json.dumps({'type': 'references', 'refs': references})}\n\n"

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

        finally:
            if full_answer.strip():
                try:
                    from src.database import Session as NewDbSession
                    with NewDbSession(engine) as save_db:
                        save_repo = MessageRepository(save_db)
                        save_repo.create(session_id, "assistant", full_answer,
                                         json.dumps(references, ensure_ascii=False) if references else None)
                        s = save_db.get(SessionModel, session_id)
                        if s:
                            s.updated_at = datetime.now(timezone.utc)
                            save_db.add(s)
                            save_db.commit()
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
