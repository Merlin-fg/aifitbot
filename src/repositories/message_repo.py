"""消息数据访问层。"""

from typing import Optional

from sqlmodel import Session as DbSession, select
from src.models.message import Message


class MessageRepository:
    """消息表 CRUD。"""

    def __init__(self, session: DbSession):
        self.session = session

    def create(self, session_id: int, role: str, content: str,
               references: Optional[str] = None) -> Message:
        """创建新消息。"""
        msg = Message(session_id=session_id, role=role, content=content,
                       references=references)
        self.session.add(msg)
        self.session.commit()
        self.session.refresh(msg)
        return msg

    def get_by_session(self, session_id: int, limit: Optional[int] = None) -> list[Message]:
        """获取某个会话的消息（按时间正序）。可限制最近 N 条。"""
        statement = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc() if limit else Message.created_at.asc())
        )
        results = list(self.session.exec(statement).all())
        if limit:
            results = list(reversed(results[:limit]))  # 取最近 N 条，再翻回正序
        return results

    def delete_by_session(self, session_id: int):
        """删除某会话的所有消息。"""
        messages = self.get_by_session(session_id)
        for msg in messages:
            self.session.delete(msg)
        self.session.commit()
