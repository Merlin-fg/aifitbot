"""认证服务：注册、登录、密码修改、JWT 签发与验证。"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from sqlmodel import Session

from src.config import JWT_SECRET, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from src.models.user import User
from src.repositories.user_repo import UserRepository
from src.utils.logger import logger


class AuthService:
    """用户认证与 JWT（JSON Web Token，一种防篡改的身份令牌）管理。"""

    def __init__(self, session: Session):
        """初始化认证服务。

        Args:
            session: SQLModel 数据库会话，用于操作用户表。
        """
        self.repo = UserRepository(session)

    # ---- 密码工具 ----
    @staticmethod
    def _normalize_password(password: str) -> bytes:
        """预处理密码：超 72 字节时先用 SHA-256 哈希缩短，避免 bcrypt 自身截断破坏 UTF-8 多字节字符。"""
        pw_bytes = password.encode("utf-8")
        if len(pw_bytes) > 72:
            import hashlib
            return hashlib.sha256(pw_bytes).hexdigest().encode("utf-8")
        return pw_bytes

    @classmethod
    def hash_password(cls, password: str) -> str:
        """将明文密码哈希化。"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(cls._normalize_password(password), salt).decode("utf-8")

    @classmethod
    def verify_password(cls, plain_password: str, hashed_password: str) -> bool:
        """验证明文密码是否与哈希匹配。"""
        hashed_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(cls._normalize_password(plain_password), hashed_bytes)

    # ---- 注册 ----
    def register(self, username: str, password: str) -> tuple[bool, str]:
        """注册新用户。

        Returns:
            (success, message)
        """
        if self.repo.get_by_username(username):
            return False, "用户名已存在"

        hashed = self.hash_password(password)
        self.repo.create(username, hashed)
        logger.info(f"新用户注册: {username}")
        return True, "注册成功"

    # ---- 登录 ----
    def login(self, username: str, password: str) -> tuple[bool, Optional[str], str]:
        """验证登录并签发 JWT。

        Returns:
            (success, access_token, message)
        """
        user = self.repo.get_by_username(username)
        if not user:
            return False, None, "用户名或密码错误"

        if not self.verify_password(password, user.hashed_password):
            return False, None, "用户名或密码错误"

        token = self._create_token(user)
        logger.info(f"用户登录: {username}")
        return True, token, "登录成功"

    # ---- 修改密码 ----
    def change_password(self, user_id: int, old_password: str, new_password: str) -> tuple[bool, str]:
        """修改用户密码。

        Returns:
            (success, message)
        """
        user = self.repo.get_by_id(user_id)
        if not user:
            return False, "用户不存在"

        if not self.verify_password(old_password, user.hashed_password):
            return False, "原密码错误"

        new_hashed = self.hash_password(new_password)
        self.repo.update_password(user_id, new_hashed)
        logger.info(f"用户 {user.username} 修改了密码")
        return True, "密码修改成功"

    # ---- JWT ----
    def _create_token(self, user: User) -> str:
        """为用户签发 JWT access token。"""
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "role": user.role.value,
            "exp": expire,
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        """解码并验证 JWT token，失败返回 None。"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except JWTError:
            return None

    def get_current_user(self, token: str) -> Optional[User]:
        """根据 JWT token 获取当前用户。"""
        payload = self.decode_token(token)
        if payload is None:
            return None
        user_id = payload.get("sub")
        if user_id is None:
            return None
        return self.repo.get_by_id(int(user_id))
