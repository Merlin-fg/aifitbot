"""会话管理数据访问层。"""

from typing import Optional
from datetime import datetime, timezone

from sqlmodel import Session as DbSession, select
from src.models.session import Session


class SessionRepository:
    """会话表 CRUD 操作。"""

    def __init__(self, session: DbSession):
        self.session = session

    def create(self, user_id: int, title: str = "新对话") -> Session:
        """为用户创建新会话。"""
        sess = Session(user_id=user_id, title=title)
        self.session.add(sess)
        self.session.commit()
        self.session.refresh(sess)
        return sess

    def get_by_user(self, user_id: int) -> list[Session]:
        """获取用户的所有会话，按更新时间倒序。"""
        statement = (
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.updated_at.desc())
        )
        return list(self.session.exec(statement).all())

    def get_by_id(self, session_id: int) -> Optional[Session]:
        """按 ID 获取会话。"""
        return self.session.get(Session, session_id)

    def update_title(self, session_id: int, title: str) -> bool:
        """重命名会话。"""
        sess = self.get_by_id(session_id)
        if not sess:
            return False
        sess.title = title
        sess.updated_at = datetime.now(timezone.utc)
        self.session.add(sess)
        self.session.commit()
        return True

    def delete(self, session_id: int) -> bool:
        """删除会话。"""
        sess = self.get_by_id(session_id)
        if not sess:
            return False
        self.session.delete(sess)
        self.session.commit()
        return True
