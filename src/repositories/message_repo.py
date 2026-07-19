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

    def get_by_session(self, session_id: int) -> list[Message]:
        """获取某个会话的所有消息（按时间正序）。"""
        statement = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
        )
        return list(self.session.exec(statement).all())

    def delete_by_session(self, session_id: int):
        """删除某会话的所有消息。"""
        messages = self.get_by_session(session_id)
        for msg in messages:
            self.session.delete(msg)
        self.session.commit()
