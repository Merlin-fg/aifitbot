"""RAG 问答服务——查询改写 + 混合检索 + 重排序 + 对话摘要 + 高频缓存。"""

import hashlib
import json
import time
from typing import List, Optional, Dict, Tuple

import requests
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.repositories.vector_repo import VectorRepository
from src.config import RAG_K, RAG_MAX_CHUNK, API_KEY, BASE_URL, RAG_HISTORY_ROUNDS
from src.utils.logger import logger

SYSTEM_PROMPT = (
    "你是一位专业的私人健身教练与营养师。根据以下已知的健身知识片段回答用户问题。"
    "如果知识片段不足以回答，可以结合你自身的专业知识补充，"
    "但请务必在回答中明确指出哪些内容来自知识库、哪些来自你的补充知识。"
    "\n\n## 训练计划隐藏格式（极其重要，必须遵守）"
    "\n当用户询问训练动作或计划时，在回答末尾插入训练计划 JSON，"
    "\n用 STARTJSON 和 ENDJSON 标记包裹，必须各占单独一行："
    "\nSTARTJSON"
    "\n{{\"plan_name\":\"计划名\",\"exercises\":[{{\"name\":\"动作\",\"sets\":4,\"reps\":\"8-12\",\"rest_sec\":90,\"notes\":\"要点\"}}]}}"
    "\nENDJSON"
    "\n标记单词必须完整拼写、独占一行、前后不要加任何其他字符。"
    "\n\n## 打卡引导"
    "\n如果本次回答包含了训练计划，在回答最末尾追加一行："
    "\n💡 需要我把这个计划加入训练打卡吗？回复\"需要\"即可。"
    "{profile}"
    "\n\n参考知识片段：\n{context}"
)

QUERY_REWRITE_PROMPT = (
    "将用户的健身口语化问题改写为用于检索知识库的关键词。"
    "把口语表达替换为标准健身术语，提取核心概念。只输出改写后的关键词，不要解释。\n"
    "示例：'怎么把胳膊练粗' → '肱二头肌 肱三头肌 增肌训练 手臂动作'\n"
    "示例：'肚子太大了想减掉' → '减脂 腹部 热量缺口 有氧运动 腹肌训练'\n"
    "示例：'我想胸肌变大' → '胸大肌 增肌训练 卧推 胸部动作'\n"
    "现在改写：{question}"
)

SUMMARIZE_PROMPT = (
    "将以下对话历史压缩为一段简短摘要（不超过150字），"
    "只保留关键的健身问题和教练建议的核心要点。\n\n"
    "对话历史：\n{history}\n\n摘要："
)


class RAGService:
    """RAG 服务 v2——查询改写 + 混合检索 + 重排序 + 对话摘要 + 高频缓存。

    v2 新增：
    - BM25 + 向量 RRF 混合检索（在 vector_repo 中实现）
    - 阿里云 DashScope gte-rerank 重排序
    - 超 6 轮对话自动摘要压缩
    """

    # 高频缓存
    _cache: Dict[str, Tuple[dict, float]] = {}
    _CACHE_TTL = 3600
    _CACHE_MAX_SIZE = 100

    # 重排序 API 端点
    _RERANK_URL = (
        (BASE_URL or "").rstrip("/").replace("/compatible-mode/v1", "")
        + "/api/v1/services/rerank/text-rerank/text-rerank"
    ) if BASE_URL else ""

    def __init__(self, llm, vector_repo: VectorRepository):
        self.llm = llm
        self.vector_repo = vector_repo
        self._chain = None
        self._rewrite_chain = None

    @classmethod
    def _cache_key(cls, question: str) -> str:
        """生成缓存键（问题 hash）。"""
        return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _cache_get(cls, question: str) -> Optional[dict]:
        """查缓存，过期返回 None。"""
        key = cls._cache_key(question)
        entry = cls._cache.get(key)
        if entry:
            result, ts = entry
            if time.time() - ts < cls._CACHE_TTL:
                return result
            del cls._cache[key]
        return None

    @classmethod
    def _cache_set(cls, question: str, result: dict):
        """写入缓存，超容量时淘汰最旧的。"""
        if len(cls._cache) >= cls._CACHE_MAX_SIZE:
            oldest = min(cls._cache.items(), key=lambda x: x[1][1])
            del cls._cache[oldest[0]]
        cls._cache[cls._cache_key(question)] = (result, time.time())

    def _rerank(self, query: str, docs: List,
                top_k: int = 3) -> List:
        """阿里云 DashScope gte-rerank 重排序。

        对检索结果做二次精排，提升 Top-k 准确率。
        """
        if not docs or not self._RERANK_URL:
            return docs

        documents = [d.page_content[:500] for d in docs]
        try:
            resp = requests.post(
                self._RERANK_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gte-rerank",
                    "input": {"query": query, "documents": documents},
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("output", {}).get("results", [])
                if results:
                    # 按 relevance_score 降序重排
                    ranked = sorted(results, key=lambda x: x.get("relevance_score", 0),
                                    reverse=True)
                    reranked = [docs[r["index"]] for r in ranked[:top_k]
                               if r["index"] < len(docs)]
                    logger.info(
                        f"重排序: {len(docs)} → {len(reranked)} (gte-rerank)"
                    )
                    return reranked
        except Exception as e:
            logger.warning(f"重排序失败（降级使用原始结果）: {e}")
        return docs[:top_k]

    def _summarize_history(self, chat_history: List) -> List:
        """对话摘要压缩：超 6 轮时，将旧消息压缩为摘要。

        保留最近 6 轮（12 条），更早的消息用 LLM 压缩成一段摘要。
        """
        max_msgs = RAG_HISTORY_ROUNDS * 2  # 6 轮 = 12 条
        if len(chat_history) <= max_msgs:
            return chat_history

        # 超出部分（旧消息）
        old_msgs = chat_history[:-max_msgs]
        recent_msgs = chat_history[-max_msgs:]

        # 构建历史文本
        history_text = ""
        for m in old_msgs:
            role = "用户" if isinstance(m, HumanMessage) else "教练"
            content = m.content if hasattr(m, 'content') else str(m)
            history_text += f"{role}：{content[:200]}\n"

        # 调用 LLM 生成摘要
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            prompt = ChatPromptTemplate.from_messages([
                ("human", SUMMARIZE_PROMPT),
            ])
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({"history": history_text}).strip()
            if summary:
                logger.info(f"对话摘要压缩: {len(old_msgs)} 条 → {len(summary)} 字摘要")
                return [SystemMessage(
                    content=f"[更早的对话摘要] {summary}"
                )] + recent_msgs
        except Exception as e:
            logger.warning(f"摘要生成失败: {e}")

        return recent_msgs  # 降级：直接截断

    def _get_rewrite_chain(self):
        """查询改写链（轻量，只做关键词提取）。"""
        if self._rewrite_chain is None:
            prompt = ChatPromptTemplate.from_messages([
                ("human", QUERY_REWRITE_PROMPT),
            ])
            self._rewrite_chain = prompt | self.llm | StrOutputParser()
        return self._rewrite_chain

    def _rewrite_query(self, user_input: str) -> str:
        """将用户口语化问题改写为标准检索词。"""
        try:
            chain = self._get_rewrite_chain()
            rewritten = chain.invoke({"question": user_input}).strip()
            # 排除明显无效的改写：空返回、返回原文、返回过长文本
            if (rewritten and rewritten != user_input
                    and len(rewritten) < len(user_input) * 5):
                logger.info(f"查询改写: '{user_input[:40]}...' → '{rewritten[:60]}...'")
                return rewritten
        except Exception as e:
            logger.warning(f"查询改写失败，使用原始问题: {e}")
        return user_input

    def _get_chain(self):
        """构建 LangChain RAG 管道（带对话历史支持）。"""
        if self._chain is None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            def format_docs(docs):
                return "\n\n---\n\n".join(
                    f"[来源: {d.metadata.get('source', '未知')}]\n"
                    f"[标题: {d.metadata.get('title', '')}]\n"
                    f"{d.page_content}"
                    for d in docs
                )

            self._chain = (
                {
                    "context": itemgetter("context"),
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                    "profile": itemgetter("profile"),
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        return self._chain

    def query(self, user_input: str,
              chat_history: Optional[List] = None,
              profile_text: str = "") -> dict:
        """执行 RAG 查询。

        Args:
            user_input: 用户问题
            chat_history: 对话历史，格式为 [HumanMessage, AIMessage, ...]
            profile_text: 用户档案文本（注入 system prompt）

        Returns:
            {"answer": str, "references": list}
        """
        # 高频缓存：无对话历史的简单问题优先查缓存
        if not chat_history:
            cached = self._cache_get(user_input)
            if cached:
                logger.info(f"RAG 缓存命中: '{user_input[:30]}...'")
                return cached

        # 第一步：查询改写
        search_query = self._rewrite_query(user_input)

        # 第二步：混合检索（BM25 + 向量 → RRF 融合，取 5 条候选）
        docs, scores, category = self.vector_repo.search_hybrid(search_query, k=5)

        # 第三步：重排序（gte-rerank 精排 → top 3）
        if len(docs) > RAG_K:
            docs = self._rerank(search_query, docs, top_k=RAG_K)

        # 构建引用列表
        references = []
        context_docs = []
        for doc, score in zip(docs, scores[:len(docs)]):
            references.append({
                "source": doc.metadata.get("source", "未知"),
                "title": doc.metadata.get("title", ""),
                "content": doc.page_content[:RAG_MAX_CHUNK],
                "score": round(float(score), 4),
                "tags": doc.metadata.get("tags", ""),
            })
            context_docs.append(doc)

        # 第四步：对话摘要压缩（超 6 轮自动压缩旧消息）
        if chat_history is None:
            chat_history = []
        else:
            chat_history = self._summarize_history(chat_history)

        # 构建档案文本
        profile_block = f"\n\n## 用户档案\n{profile_text}" if profile_text else ""

        # 第五步：构建上下文并生成回答
        chain = self._get_chain()
        formatted_context = "\n\n---\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n"
            f"[标题: {d.metadata.get('title', '')}]\n"
            f"{d.page_content}"
            for d in context_docs
        )

        try:
            answer = chain.invoke({
                "context": formatted_context,
                "input": user_input,
                "chat_history": chat_history,
                "profile": profile_block,
            })
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            # 降级：不带 chat_history 试一次
            answer = chain.invoke({
                "context": formatted_context,
                "input": user_input,
                "chat_history": [],
                "profile": profile_block,
            })

        result = {"answer": answer, "references": references}

        # 缓存无历史的查询结果
        if not chat_history:
            self._cache_set(user_input, result)

        logger.info(
            f"RAG v2 完成 [{category}]: "
            f"'{user_input[:30]}...' → {len(references)} refs, {len(answer)} chars"
        )
        return result
