"""知识库管理服务——文档上传、删除、向量化的编排逻辑。"""

import os
import uuid

from sqlmodel import Session

from src.repositories.document_repo import DocumentRepository
from src.repositories.vector_repo import VectorRepository
from src.models.document import Document
from src.utils.logger import logger

ALLOWED_EXTENSIONS = {".md", ".pdf", ".txt"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


class KBService:
    """知识库管理服务——协调文档存储和向量化两个子系统。"""

    def __init__(self, session: Session, vector_repo: VectorRepository):
        """初始化知识库服务。

        Args:
            session: 数据库会话，用于文档元数据 CRUD。
            vector_repo: 向量库仓库，负责文档切分、向量生成和检索。
        """
        self.repo = DocumentRepository(session)
        self.vector_repo = vector_repo

    def upload_document(self, file_content: bytes, original_filename: str) -> tuple[bool, str]:
        """处理文档上传：校验 → 存文件 → 记录元数据 → 向量化。

        Returns:
            (success, message)
        """
        # 校验扩展名
        ext = os.path.splitext(original_filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"不支持的文件类型: {ext}，支持: {', '.join(ALLOWED_EXTENSIONS)}"

        # 校验大小
        if len(file_content) > MAX_FILE_SIZE:
            return False, f"文件过大（最大 20MB），当前: {len(file_content) / 1024 / 1024:.1f}MB"

        # 生成唯一存储名
        stored_name = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(self.vector_repo.data_dir, stored_name)

        # 确保目录存在
        os.makedirs(self.vector_repo.data_dir, exist_ok=True)

        # 写入文件
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 记录元数据
        doc = self.repo.create(
            filename=original_filename,
            stored_name=stored_name,
            file_type=ext.lstrip("."),
            size_bytes=len(file_content),
            status="processing",
        )

        # 向量化
        try:
            chunk_count = self.vector_repo.add_document(stored_name, display_name=original_filename)
            self.repo.update_status(doc.id, "ready", chunk_count=chunk_count)
            logger.info(f"文档就绪: {original_filename} → {stored_name}, {chunk_count} chunks")
            return True, f"上传成功，已切分为 {chunk_count} 个知识块"
        except Exception as e:
            self.repo.update_status(doc.id, "error", error_msg=str(e))
            try:
                self.vector_repo.delete_file(stored_name)
            except Exception as del_err:
                logger.warning(f"清理文件失败: {stored_name}, {del_err}")
            logger.error(f"向量化失败: {original_filename}, {e}")
            return False, f"文档处理失败: {e}"

    def list_documents(self) -> list[Document]:
        """获取所有文档列表。"""
        return self.repo.list_all()

    def delete_document(self, doc_id: int) -> tuple[bool, str]:
        """删除文档：从向量库移除 → 删文件 → 删元数据记录。"""
        doc = self.repo.get_by_id(doc_id)
        if not doc:
            return False, "文档不存在"

        # 从向量库删除
        self.vector_repo.remove_document(doc.stored_name)
        # 删物理文件
        self.vector_repo.delete_file(doc.stored_name)
        # 删元数据
        self.repo.delete(doc_id)
        logger.info(f"文档已删除: {doc.filename}")
        return True, "文档已删除"

    def get_document(self, doc_id: int) -> Document | None:
        """获取单个文档。"""
        return self.repo.get_by_id(doc_id)
