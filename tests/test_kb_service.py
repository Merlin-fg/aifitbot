"""kb_service 测试——Mock 向量库和文件操作。"""

import pytest
from unittest.mock import MagicMock, patch, call
from src.services.kb_service import KBService, ALLOWED_EXTENSIONS
from src.repositories.vector_repo import VectorRepository


class TestKBService:
    @pytest.fixture
    def mock_vector_repo(self):
        repo = MagicMock(spec=VectorRepository)
        repo.data_dir = "/tmp/test_data"
        repo.add_document.return_value = 3  # 返回 3 个 chunks
        repo.remove_document.return_value = True
        repo.delete_file.return_value = None
        return repo

    @pytest.fixture
    def kb_service(self, db_session, mock_vector_repo):
        return KBService(db_session, mock_vector_repo)

    def test_upload_md_success(self, kb_service, mock_vector_repo):
        content = "# 测试文档\n\n这是测试内容".encode("utf-8")
        ok, msg = kb_service.upload_document(content, "test.md")
        assert ok is True
        assert "3" in msg  # 3 个 chunks
        mock_vector_repo.add_document.assert_called_once()

    def test_upload_reject_invalid_extension(self, kb_service):
        content = b"test"
        ok, msg = kb_service.upload_document(content, "test.exe")
        assert ok is False
        assert "不支持" in msg

    def test_upload_reject_large_file(self, kb_service):
        content = b"x" * (21 * 1024 * 1024)  # 21MB
        ok, msg = kb_service.upload_document(content, "big.md")
        assert ok is False
        assert "过大" in msg

    def test_upload_vectorization_failure(self, db_session, mock_vector_repo):
        mock_vector_repo.add_document.side_effect = RuntimeError("API 错误")
        kb = KBService(db_session, mock_vector_repo)
        ok, msg = kb.upload_document(b"content", "fail.md")
        assert ok is False
        assert "失败" in msg
        # 应调用了 cleanup
        mock_vector_repo.delete_file.assert_called_once()

    def test_list_documents(self, kb_service):
        kb_service.repo.create("a.md", "a.md", "md", 100, status="ready")
        kb_service.repo.create("b.pdf", "b.pdf", "pdf", 200, status="ready")
        docs = kb_service.list_documents()
        assert len(docs) == 2

    def test_delete_document(self, kb_service, mock_vector_repo):
        doc = kb_service.repo.create("del.md", "stored.md", "md", 100, status="ready")
        ok, msg = kb_service.delete_document(doc.id)
        assert ok is True
        mock_vector_repo.remove_document.assert_called_once_with("stored.md")
        mock_vector_repo.delete_file.assert_called_once_with("stored.md")

    def test_delete_nonexistent(self, kb_service):
        ok, msg = kb_service.delete_document(9999)
        assert ok is False


class TestAllowedExtensions:
    def test_all_md_pdf_txt_allowed(self):
        assert ".md" in ALLOWED_EXTENSIONS
        assert ".pdf" in ALLOWED_EXTENSIONS
        assert ".txt" in ALLOWED_EXTENSIONS
        assert ".exe" not in ALLOWED_EXTENSIONS
        assert ".jpg" not in ALLOWED_EXTENSIONS
