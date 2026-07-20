"""向量数据库仓库——封装 ChromaDB 的读写操作。"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

from src.utils.logger import logger


class VectorRepository:
    """ChromaDB 向量库管理。

    负责：文档加载 → 文本切分 → 向量嵌入 → 检索 → 按来源删除。
    """

    def __init__(self, persist_dir: str = "chroma_db",
                 data_dir: str = "data",
                 embedding=None):
        """初始化向量库仓库。

        Args:
            persist_dir: ChromaDB 数据持久化目录，向量库文件存于此。
            data_dir: 原始文档存储目录，上传的文件放这里。
            embedding: 嵌入函数实例（如 DashScopeEmbeddings），用于把文本转为向量。
                       传入 None 会导致检索时报错。
        """
        self.persist_dir = persist_dir
        self.data_dir = data_dir
        self.embedding = embedding
        self._store: Optional[Chroma] = None

    def _get_store(self) -> Chroma:
        """懒加载 ChromaDB 实例（首次访问时才连接磁盘，避免启动阻塞）。"""
        if self._store is None:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding,
            )
        return self._store

    # ---- 文档处理 ----
    def _load_file(self, file_path: str) -> List[LCDocument]:
        """根据扩展名选择合适的加载器。"""
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    def _split_documents(self, docs: List[LCDocument]) -> List[LCDocument]:
        """把长文档切分成小块（每块 500 字符，重叠 50 字符）。

        chunk_overlap=50 确保块之间信息不丢失（一个句子可能跨两个块）。
        separators 列表包含中文标点，使切分尽量在句子边界处断开，而非
        生硬地切断句子。"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", " "],
        )
        return splitter.split_documents(docs)

    # ---- 公开接口 ----
    def add_document(self, stored_name: str, display_name: str = "") -> int:
        """将 data/ 目录下的文件向量化并存入 Chroma。

        Args:
            stored_name: 实际存储的文件名（如 uuid.pdf）
            display_name: 显示用的原始文件名

        Returns:
            切分后的文档块数量。
        """
        file_path = os.path.join(self.data_dir, stored_name)
        raw_docs = self._load_file(file_path)

        # 写入来源文件名到元数据（优先用显示名）
        source_name = display_name if display_name else stored_name
        for doc in raw_docs:
            doc.metadata["source"] = source_name

        chunks = self._split_documents(raw_docs)
        store = self._get_store()
        store.add_documents(chunks)
        logger.info(f"向量库新增文档: {stored_name}, {len(chunks)} 个块")
        return len(chunks)

    def remove_document(self, stored_name: str) -> bool:
        """从向量库中删除指定文档的所有向量块。

        注意：直接访问 ChromaDB 内部 ._collection 属性（非公共 API），
        因为 Chroma 的公共接口不支持按 metadata.source 过滤删除。

        Args:
            stored_name: 存储文件名，与 metadata 的 source 字段匹配。
        """
        store = self._get_store()
        try:
            collection = store._collection  # ChromaDB 底层集合对象，非公共 API
            # 按 metadata.source 过滤，找到属于该文档的所有块 ID
            results = collection.get(where={"source": stored_name})
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                logger.info(f"向量库删除文档: {stored_name}, {len(ids)} 个块")
            return True
        except Exception as e:
            logger.error(f"删除向量失败: {stored_name}, {e}")
            return False

    def delete_file(self, stored_name: str):
        """删除 data/ 目录下的原始文件。"""
        file_path = os.path.join(self.data_dir, stored_name)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError as e:
            logger.warning(f"文件删除失败: {file_path}, {e}")

    def as_retriever(self, k: int = 3):
        """返回 LangChain 检索器，用于 RAG 管道中的文档检索。

        Args:
            k: 返回的最相似文档块数量（默认 3）。
        """
        return self._get_store().as_retriever(search_kwargs={"k": k})

    def get_document_sources(self) -> List[str]:
        """获取向量库中所有的文档来源。"""
        store = self._get_store()
        try:
            collection = store._collection
            results = collection.get()
            sources = set()
            for meta in results.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return list(sources)
        except Exception:
            return []

    def reload(self):
        """强制重新加载向量库。"""
        self._store = None
