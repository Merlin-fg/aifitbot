"""RAG 问答服务——从知识库检索相关文档，再由 LLM 基于文档生成带引用的回答。"""

import json
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.repositories.vector_repo import VectorRepository
from src.utils.logger import logger

# 系统提示词模板，{context} 会被替换为检索到的知识库片段
SYSTEM_PROMPT = (
    "你是一位专业的私人健身教练与营养师。根据以下已知的健身知识片段回答用户问题。"
    "如果知识片段不足以回答，可以结合你自身的专业知识补充，"
    "但请务必在回答中明确指出哪些内容来自知识库、哪些来自你的补充知识。"
    "\n\n参考知识片段：\n{context}"
)


class RAGService:
    """RAG（检索增强生成）服务——先从向量库检索相关文档，再让 LLM 基于文档生成回答。

    工作流程：
    1. 用户提问 → 2. 向量库检索相似文档 → 3. 文档+问题拼成提示词 → 4. LLM 生成回答 + 引用标注
    """

    def __init__(self, llm, vector_repo: VectorRepository):
        """初始化 RAG 服务。

        Args:
            llm: LangChain ChatModel 实例（如 ChatOpenAI），用于生成回答。
            vector_repo: 向量库仓库，提供文档检索能力。
        """
        self.llm = llm
        self.vector_repo = vector_repo
        self._chain = None  # 懒加载：首次查询时才构建 LangChain 管道

    def _get_chain(self):
        """构建 LangChain LCEL 处理管道（仅首次调用，之后复用）。

        管道数据流：
        用户输入 → {
            context: 向量库检索 → format_docs(文档列表转为文本),
            input:  原样透传用户问题
        } → 填入提示词模板 → LLM 生成 → 提取纯文本字符串
        """
        if self._chain is None:
            # 从向量库获取检索器（k=3 表示取最相似的 3 个文档块）
            retriever = self.vector_repo.as_retriever(k=3)

            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ])

            def format_docs(docs):
                """将检索到的 Document 列表格式化为带来源标注的文本块。"""
                return "\n\n---\n\n".join(
                    f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
                    for d in docs
                )

            # 用 | 管道符串联各步骤：检索→拼提示词→LLM→提取文本
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
        """执行 RAG 查询：检索知识库 → LLM 生成回答 → 返回答案+引用。

        注意：这里分两次访问向量库：
        - _get_chain() 内的 retriever 用于把文档拼入提示词（不需要分数）
        - 下面的 similarity_search_with_score 用于获取相似度分数展示给用户
          这是因为 LangChain 的 retriever 标准接口不返回分数，
          所以需要直接调用 ChromaDB 底层方法获取带分数的结果。

        Returns:
            dict 包含:
                answer: str      — 生成的回答
                references: list — 引用的知识片段，每项含 source/content/score
        """
        # 第一步：检索文档并获取相似度分数（给用户看引用）
        retriever = self.vector_repo.as_retriever(k=3)
        try:
            # 绕过 LangChain retriever，直接用 ChromaDB 底层 API 获取分数
            docs_with_scores = self.vector_repo._get_store().similarity_search_with_score(
                user_input, k=3
            )
        except Exception:
            docs_with_scores = []

        references = []
        for doc, score in docs_with_scores:
            references.append({
                "source": doc.metadata.get("source", "未知"),
                "content": doc.page_content[:300],  # 截取前300字预览，避免引用过长
                "score": round(float(score), 4),
            })

        # 第二步：通过 LangChain 管道生成回答（管道内会再做一次检索）
        chain = self._get_chain()
        answer = chain.invoke(user_input)

        logger.info(f"RAG 查询完成，检索到 {len(references)} 条参考，回答 {len(answer)} 字")
        return {"answer": answer, "references": references}
