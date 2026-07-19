import os
from typing import List

import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from src.config import API_KEY, BASE_URL, EMBEDDING_MODEL


# ============================================================
# 阿里云百炼 Embedding 适配器
# ============================================================
class DashScopeEmbeddings(BaseModel, Embeddings):
    """阿里云百炼 DashScope Embedding API 封装，兼容 LangChain Embeddings 接口。"""

    api_key: str = Field(default=API_KEY)
    base_url: str = Field(default=BASE_URL)
    model: str = Field(default=EMBEDDING_MODEL)

    @property
    def _api_url(self) -> str:
        """DashScope embedding 端点，替换 compatible-mode 为 api/v1。"""
        # https://....com/compatible-mode/v1 → https://....com/api/v1/...
        base = self.base_url.rstrip("/").replace("/compatible-mode/v1", "")
        return f"{base}/api/v1/services/embeddings/text-embedding/text-embedding"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本列表。"""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        # 阿里云 embedding API 单次限制 25 条
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = requests.post(
                self._api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": {"texts": batch}},
                timeout=60,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"DashScope Embedding 失败 (HTTP {response.status_code}): {response.text}"
                )
            result = response.json()
            embeddings = result.get("output", {}).get("embeddings", [])
            all_embeddings.extend([e["embedding"] for e in embeddings])

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询文本。"""
        return self.embed_documents([text])[0]


# ============================================================
# 文档加载与切分
# ============================================================
def load_and_split_documents(data_dir: str = "data"):
    """加载 data/ 目录下所有 .md 文件并切分为文本块。"""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " "],
    )
    return text_splitter.split_documents(documents)


# ============================================================
# 向量存储
# ============================================================
def get_vectorstore():
    """加载或创建 ChromaDB 向量库。"""
    persist_dir = "chroma_db"
    embeddings = DashScopeEmbeddings()

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    else:
        docs = load_and_split_documents()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
    return vectorstore


# ============================================================
# RAG 链组装
# ============================================================
def create_rag_chain(llm):
    """手工组装 RAG 检索链：检索 → 格式化文档 → 填充 prompt → LLM → 提取文本。"""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "你是一位专业的私人健身教练与营养师。根据以下已知的健身知识片段回答用户问题。"
        "如果知识片段不足以回答，可以结合你自身的专业知识补充，但请明确指出哪些来自知识库、哪些来自你的补充。"
        "\n\n参考知识片段：{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
