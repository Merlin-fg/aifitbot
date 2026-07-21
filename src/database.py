"""数据库引擎与会话管理。"""

from sqlmodel import SQLModel, Session, create_engine
from src.config import DATABASE_URL

# SQLite 需要 check_same_thread=False 以支持多线程访问
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)


def init_db():
    """创建所有表 + 启用 WAL 模式以支持并发读写。"""
    SQLModel.metadata.create_all(engine)
    # WAL 模式：读写不互斥，解决并发写入阻塞问题
    if "sqlite" in DATABASE_URL:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()


def get_session():
    """获取数据库会话的上下文管理器。"""
    with Session(engine) as session:
        yield session
