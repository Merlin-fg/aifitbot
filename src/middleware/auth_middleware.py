"""JWT 认证中间件——同时支持 Cookie 和 Authorization Header 两种方式。"""

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlmodel import Session

from src.config import JWT_SECRET, JWT_ALGORITHM
from src.database import get_session
from src.models.user import User, UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def _extract_token(request: Request, header_token: Optional[str]) -> Optional[str]:
    """优先从 Authorization Header 取 Token，其次从 Cookie 取。"""
    if header_token:
        return header_token
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        return cookie_token
    return None


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    """从 Cookie 或 Authorization Header 中解析当前登录用户。

    未登录则抛出 401。
    """
    actual_token = _extract_token(request, token)
    if actual_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录",
        )

    try:
        payload = jwt.decode(actual_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Token 无效")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")

    user = session.get(User, int(user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")

    return user


async def get_optional_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> Optional[User]:
    """可选认证——已登录返回 User，未登录返回 None（不报错）。"""
    actual_token = _extract_token(request, token)
    if actual_token is None:
        return None
    try:
        payload = jwt.decode(actual_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id:
            return session.get(User, int(user_id))
    except JWTError:
        pass
    return None


def require_admin(user: User = Depends(get_current_user)) -> User:
    """检查是否为管理员，非管理员返回 403。"""
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅管理员可访问此页面",
        )
    return user
