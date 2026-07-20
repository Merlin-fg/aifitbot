"""AIFitBot 测试公共 fixtures。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlmodel import SQLModel, Session, create_engine
from unittest.mock import MagicMock, patch


@pytest.fixture
def db_session():
    """SQLite 内存数据库 session。"""
    engine = create_engine("sqlite:///:memory:", echo=False,
                           connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine)


@pytest.fixture
def mock_llm():
    """Mock LLM — 返回固定回答。"""
    from langchain_core.language_models import BaseChatModel
    llm = MagicMock(spec=BaseChatModel)
    llm.invoke.return_value.content = "Mock 回答：这是测试回复"
    # astream 用于流式
    async def mock_astream(*args, **kwargs):
        for token in ["Mock", " 回答"]:
            chunk = MagicMock()
            chunk.content = token
            yield chunk
    llm.astream = mock_astream
    return llm


@pytest.fixture
def mock_embeddings():
    """Mock Embeddings。"""
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1] * 384 for _ in range(5)]
    emb.embed_query.return_value = [0.1] * 384
    return emb


@pytest.fixture
def mock_chroma():
    """Mock ChromaDB 向量库。"""
    from langchain_core.documents import Document
    store = MagicMock()
    store.as_retriever.return_value = MagicMock()
    store._collection = MagicMock()
    store._collection.get.return_value = {"ids": [], "metadatas": []}
    # similarity_search_with_score 返回默认文档
    doc = Document(page_content="测试知识：深蹲要腰背挺直", metadata={"source": "test.md"})
    store.similarity_search_with_score.return_value = [(doc, 0.85)]
    return store


@pytest.fixture
def auth_service(db_session):
    """AuthService 实例。"""
    from src.services.auth_service import AuthService
    return AuthService(db_session)


@pytest.fixture
def registered_user(auth_service):
    """创建一个已注册的测试用户。"""
    auth_service.register("testuser", "123456")
    return "testuser", "123456"
