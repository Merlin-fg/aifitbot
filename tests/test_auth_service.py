"""auth_service 测试——纯逻辑，无需 Mock。"""

import pytest
from src.services.auth_service import AuthService
from src.models.user import UserRole


class TestPasswordHashing:
    def test_hash_and_verify(self, auth_service):
        hashed = auth_service.hash_password("mypassword")
        assert hashed != "mypassword"
        assert auth_service.verify_password("mypassword", hashed)

    def test_wrong_password(self, auth_service):
        hashed = auth_service.hash_password("correct")
        assert not auth_service.verify_password("wrong", hashed)

    def test_long_password_handled(self, auth_service):
        """超长密码通过 SHA-256 预哈希，不截断。"""
        long_pw = "中文字符测试" * 20  # > 72 bytes
        hashed = auth_service.hash_password(long_pw)
        assert auth_service.verify_password(long_pw, hashed)

    def test_empty_password(self, auth_service):
        hashed = auth_service.hash_password("")
        assert auth_service.verify_password("", hashed)


class TestRegistration:
    def test_register_success(self, auth_service):
        ok, msg = auth_service.register("newuser", "123456")
        assert ok is True
        assert "成功" in msg

    def test_register_duplicate(self, auth_service):
        auth_service.register("dup", "123456")
        ok, msg = auth_service.register("dup", "654321")
        assert ok is False
        assert "已存在" in msg

    def test_user_has_default_role(self, auth_service, db_session):
        auth_service.register("roleuser", "123456")
        user = auth_service.repo.get_by_username("roleuser")
        assert user.role == UserRole.USER


class TestLogin:
    def test_login_success(self, auth_service):
        auth_service.register("loginuser", "secret123")
        ok, token, msg = auth_service.login("loginuser", "secret123")
        assert ok is True
        assert token is not None
        assert "成功" in msg

    def test_login_wrong_password(self, auth_service):
        auth_service.register("loginuser", "secret123")
        ok, token, msg = auth_service.login("loginuser", "wrong")
        assert ok is False
        assert token is None

    def test_login_nonexistent(self, auth_service):
        ok, token, msg = auth_service.login("ghost", "pw")
        assert ok is False

    def test_token_decodable(self, auth_service):
        auth_service.register("tokenuser", "pw123456")
        _, token, _ = auth_service.login("tokenuser", "pw123456")
        payload = auth_service.decode_token(token)
        assert payload is not None
        assert payload["username"] == "tokenuser"
        assert payload["role"] == "user"


class TestChangePassword:
    def test_change_success(self, auth_service):
        auth_service.register("chuser", "oldpass")
        user = auth_service.repo.get_by_username("chuser")
        ok, msg = auth_service.change_password(user.id, "oldpass", "newpass")
        assert ok is True
        # 新密码可登录
        ok2, _, _ = auth_service.login("chuser", "newpass")
        assert ok2 is True

    def test_change_wrong_old(self, auth_service):
        auth_service.register("chuser2", "oldpass")
        user = auth_service.repo.get_by_username("chuser2")
        ok, msg = auth_service.change_password(user.id, "wrongold", "newpass")
        assert ok is False


class TestJWT:
    def test_decode_invalid_token(self, auth_service):
        assert auth_service.decode_token("invalid.token.here") is None

    def test_token_contains_user_info(self, auth_service):
        auth_service.register("jwtuser", "password")
        user = auth_service.repo.get_by_username("jwtuser")
        current = auth_service.get_current_user(
            auth_service._create_token(user))
        assert current is not None
        assert current.username == "jwtuser"
