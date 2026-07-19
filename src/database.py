"""数据库引擎与会话管理。"""

from sqlmodel import SQLModel, Session, create_engine
from src.config import DATABASE_URL

# SQLite 需要 check_same_thread=False 以支持多线程访问
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)


def init_db():
    """创建所有表（开发阶段使用，生产环境应使用 Alembic 迁移）。"""
    SQLModel.metadata.create_all(engine)


def get_session():
    """获取数据库会话的上下文管理器。"""
    with Session(engine) as session:
        yield session
