"""文档元数据 CRUD。"""

from typing import Optional

from sqlmodel import Session, select
from src.models.document import Document


class DocumentRepository:
    """文档元数据表操作。"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, filename: str, stored_name: str, file_type: str,
               size_bytes: int, chunk_count: int = 0, status: str = "processing") -> Document:
        """记录新文档。"""
        doc = Document(
            filename=filename,
            stored_name=stored_name,
            file_type=file_type,
            size_bytes=size_bytes,
            chunk_count=chunk_count,
            status=status,
        )
        self.session.add(doc)
        self.session.commit()
        self.session.refresh(doc)
        return doc

    def list_all(self) -> list[Document]:
        """列出所有文档（按时间倒序）。"""
        statement = select(Document).order_by(Document.created_at.desc())
        return list(self.session.exec(statement).all())

    def get_by_id(self, doc_id: int) -> Optional[Document]:
        """按 ID 获取文档。"""
        return self.session.get(Document, doc_id)

    def get_by_stored_name(self, stored_name: str) -> Optional[Document]:
        """按存储文件名查找。"""
        statement = select(Document).where(Document.stored_name == stored_name)
        return self.session.exec(statement).first()

    def update_status(self, doc_id: int, status: str, chunk_count: int = 0,
                       error_msg: str | None = None) -> bool:
        """更新文档状态。"""
        doc = self.get_by_id(doc_id)
        if not doc:
            return False
        doc.status = status
        if chunk_count:
            doc.chunk_count = chunk_count
        if error_msg:
            doc.error_msg = error_msg
        self.session.add(doc)
        self.session.commit()
        return True

    def delete(self, doc_id: int) -> Optional[Document]:
        """删除文档记录，返回被删的文档（用于清理文件）。"""
        doc = self.get_by_id(doc_id)
        if not doc:
            return None
        self.session.delete(doc)
        self.session.commit()
        return doc
