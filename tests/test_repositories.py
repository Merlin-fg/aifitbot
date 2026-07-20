"""repositories 层测试——数据库 CRUD 操作。"""

import pytest
from datetime import datetime
from src.repositories.user_repo import UserRepository
from src.repositories.session_repo import SessionRepository
from src.repositories.message_repo import MessageRepository
from src.repositories.document_repo import DocumentRepository


class TestUserRepository:
    def test_create_user(self, db_session):
        repo = UserRepository(db_session)
        user = repo.create("alice", "hashed_pw")
        assert user.username == "alice"
        assert user.id is not None

    def test_get_by_username_found(self, db_session):
        repo = UserRepository(db_session)
        repo.create("bob", "pw")
        user = repo.get_by_username("bob")
        assert user is not None
        assert user.username == "bob"

    def test_get_by_username_not_found(self, db_session):
        repo = UserRepository(db_session)
        assert repo.get_by_username("nobody") is None

    def test_update_password(self, db_session):
        repo = UserRepository(db_session)
        user = repo.create("eve", "old_pw")
        ok = repo.update_password(user.id, "new_pw")
        assert ok is True
        # 验证密码已更新
        updated = repo.get_by_id(user.id)
        assert updated.hashed_password == "new_pw"

    def test_update_password_invalid_id(self, db_session):
        repo = UserRepository(db_session)
        assert repo.update_password(9999, "pw") is False

    def test_user_count(self, db_session):
        repo = UserRepository(db_session)
        assert repo.user_count() == 0
        repo.create("u1", "pw")
        repo.create("u2", "pw")
        assert repo.user_count() == 2

    def test_get_recent(self, db_session):
        repo = UserRepository(db_session)
        repo.create("old", "pw")
        repo.create("new", "pw")
        recent = repo.get_recent(1)
        assert len(recent) == 1
        assert recent[0].username == "new"


class TestSessionRepository:
    def test_create_session(self, db_session):
        repo = SessionRepository(db_session)
        sess = repo.create(user_id=1, title="测试会话")
        assert sess.title == "测试会话"
        assert sess.user_id == 1

    def test_get_by_user(self, db_session):
        repo = SessionRepository(db_session)
        repo.create(user_id=1, title="A")
        repo.create(user_id=1, title="B")
        repo.create(user_id=2, title="C")
        sessions = repo.get_by_user(1)
        assert len(sessions) == 2

    def test_update_title(self, db_session):
        repo = SessionRepository(db_session)
        sess = repo.create(user_id=1, title="旧标题")
        ok = repo.update_title(sess.id, "新标题")
        assert ok
        updated = repo.get_by_id(sess.id)
        assert updated.title == "新标题"

    def test_delete_session(self, db_session):
        repo = SessionRepository(db_session)
        sess = repo.create(user_id=1)
        ok = repo.delete(sess.id)
        assert ok
        assert repo.get_by_id(sess.id) is None


class TestMessageRepository:
    def test_create_and_retrieve(self, db_session):
        repo = MessageRepository(db_session)
        msg = repo.create(session_id=1, role="user", content="你好世界")
        assert msg.id is not None
        messages = repo.get_by_session(1)
        assert len(messages) == 1
        assert messages[0].content == "你好世界"

    def test_delete_by_session(self, db_session):
        repo = MessageRepository(db_session)
        repo.create(session_id=1, role="user", content="msg1")
        repo.create(session_id=1, role="assistant", content="msg2")
        repo.delete_by_session(1)
        assert len(repo.get_by_session(1)) == 0


class TestDocumentRepository:
    def test_create_and_list(self, db_session):
        repo = DocumentRepository(db_session)
        repo.create("test.md", "uuid.md", "md", 1024, status="ready")
        repo.create("doc.pdf", "uuid2.pdf", "pdf", 2048)
        docs = repo.list_all()
        assert len(docs) == 2

    def test_update_status(self, db_session):
        repo = DocumentRepository(db_session)
        doc = repo.create("file.md", "u.md", "md", 100)
        repo.update_status(doc.id, "ready", chunk_count=3)
        updated = repo.get_by_id(doc.id)
        assert updated.status == "ready"
        assert updated.chunk_count == 3

    def test_delete(self, db_session):
        repo = DocumentRepository(db_session)
        doc = repo.create("del.md", "u.md", "md", 100)
        repo.delete(doc.id)
        assert repo.get_by_id(doc.id) is None
