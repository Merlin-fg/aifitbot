"""models 层测试——纯数据模型验证。"""

import pytest
from src.models.user import User, UserRole
from src.models.session import Session
from src.models.message import Message
from src.models.document import Document


class TestUserRole:
    def test_enum_values(self):
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.USER.value == "user"

    def test_enum_from_string(self):
        assert UserRole("admin") == UserRole.ADMIN
        assert UserRole("user") == UserRole.USER


class TestUser:
    def test_default_role(self, db_session):
        """新用户默认角色为 user。"""
        from src.services.auth_service import AuthService
        auth = AuthService(db_session)
        user = auth.repo.create("newuser", "hashed_pw_for_test")
        assert user.role == UserRole.USER

    def test_unique_username(self, db_session):
        """重名用户应失败。"""
        from src.services.auth_service import AuthService
        auth = AuthService(db_session)
        auth.repo.create("dup", "pw1")
        with pytest.raises(Exception):
            auth.repo.create("dup", "pw2")

    def test_created_at_auto_set(self, db_session):
        from src.services.auth_service import AuthService
        auth = AuthService(db_session)
        user = auth.repo.create("timeuser", "pw")
        assert user.created_at is not None


class TestSession:
    def test_default_title(self, db_session):
        sess = Session(user_id=1)
        assert sess.title == "新对话"

    def test_custom_title(self, db_session):
        sess = Session(user_id=1, title="增肌训练")
        assert sess.title == "增肌训练"


class TestMessage:
    def test_create_message(self):
        msg = Message(session_id=1, role="user", content="你好")
        assert msg.role == "user"
        assert msg.content == "你好"
        assert msg.references is None

    def test_message_with_references(self):
        import json
        refs = [{"source": "test.md", "content": "测试内容", "score": 0.9}]
        msg = Message(session_id=1, role="assistant", content="回答",
                       references=json.dumps(refs))
        assert msg.references is not None


class TestDocument:
    def test_default_status(self):
        doc = Document(filename="test.md", stored_name="uuid.md", file_type="md", size_bytes=100)
        assert doc.status == "processing"

    def test_unique_stored_name(self):
        doc = Document(filename="a.md", stored_name="unique123.md", file_type="md", size_bytes=100)
        assert doc.stored_name == "unique123.md"
