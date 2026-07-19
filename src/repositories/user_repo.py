"""用户数据访问层。"""

from typing import Optional

from sqlmodel import Session, select
from src.models.user import User, UserRole


class UserRepository:
    """用户表 CRUD 操作。"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, username: str, hashed_password: str, role: UserRole = UserRole.USER) -> User:
        """创建新用户。"""
        user = User(username=username, hashed_password=hashed_password, role=role)
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def get_by_username(self, username: str) -> Optional[User]:
        """按用户名查找用户。"""
        statement = select(User).where(User.username == username)
        return self.session.exec(statement).first()

    def get_by_id(self, user_id: int) -> Optional[User]:
        """按 ID 查找用户。"""
        return self.session.get(User, user_id)

    def update_password(self, user_id: int, new_hashed_password: str) -> bool:
        """更新用户密码。返回是否成功。"""
        user = self.get_by_id(user_id)
        if not user:
            return False
        user.hashed_password = new_hashed_password
        self.session.add(user)
        self.session.commit()
        return True

    def user_count(self) -> int:
        """返回用户总数。"""
        from sqlmodel import func
        return self.session.exec(select(func.count(User.id))).one()

    def get_recent(self, limit: int = 5) -> list[User]:
        """获取最近注册的用户。"""
        statement = select(User).order_by(User.created_at.desc()).limit(limit)
        return list(self.session.exec(statement).all())
