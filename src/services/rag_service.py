"""RAG 问答服务——检索 + 增强生成 + 引用溯源。"""

import json
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.repositories.vector_repo import VectorRepository
from src.utils.logger import logger


SYSTEM_PROMPT = (
    "你是一位专业的私人健身教练与营养师。根据以下已知的健身知识片段回答用户问题。"
    "如果知识片段不足以回答，可以结合你自身的专业知识补充，"
    "但请务必在回答中明确指出哪些内容来自知识库、哪些来自你的补充知识。"
    "\n\n参考知识片段：\n{context}"
)


class RAGService:
    """RAG 检索增强生成服务。"""

    def __init__(self, llm, vector_repo: VectorRepository):
        self.llm = llm
        self.vector_repo = vector_repo
        self._chain = None

    def _get_chain(self):
        """懒加载 RAG 链（首次使用才构建）。"""
        if self._chain is None:
            retriever = self.vector_repo.as_retriever(k=3)

            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ])

            def format_docs(docs):
                return "\n\n---\n\n".join(
                    f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
                    for d in docs
                )

            self._chain = (
                {
                    "context": retriever | format_docs,
                    "input": RunnablePassthrough(),
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        return self._chain

    def query(self, user_input: str) -> dict:
        """执行 RAG 查询。

        Returns:
            {
                "answer": str,           # 生成的回答
                "references": [          # 引用的知识片段
                    {"source": "xxx.md", "content": "片段原文...", "score": 0.92},
                    ...
                ]
            }
        """
        # 先检索（获取带分数的文档）
        retriever = self.vector_repo.as_retriever(k=3)
        try:
            # 用 similarity_search_with_score 获取分数
            docs_with_scores = self.vector_repo._get_store().similarity_search_with_score(
                user_input, k=3
            )
        except Exception:
            docs_with_scores = []

        references = []
        for doc, score in docs_with_scores:
            references.append({
                "source": doc.metadata.get("source", "未知"),
                "content": doc.page_content[:300],  # 截断显示
                "score": round(float(score), 4),
            })

        # 生成回答
        chain = self._get_chain()
        answer = chain.invoke(user_input)

        logger.info(f"RAG 查询完成，检索到 {len(references)} 条参考，回答 {len(answer)} 字")
        return {"answer": answer, "references": references}
