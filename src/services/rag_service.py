"""RAG 问答服务 v4——向量 + BM25 混合检索 + 质量门控 + 对话摘要 + 高频缓存。

v3→v4 移除项（评估数据驱动）：
- HyDE 假设文档：Hit@3 +3.4pp 但不值得一次额外 LLM 调用
- 多 Query 扩展：Hit@3 -6.6pp，负优化
- gte-rerank-v2 重排序：Hit@3 -23pp，领域过拟合
"""

import hashlib
import time
from typing import List, Optional, Dict, Tuple

from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.repositories.vector_repo import VectorRepository
from src.config import RAG_K, RAG_MAX_CHUNK, RAG_HISTORY_ROUNDS
from src.utils.logger import logger

SYSTEM_PROMPT = (
    "你是专业健身教练。只能根据参考知识片段回答，不得编造。"
    "若片段中无相关信息，回复'抱歉，当前知识库中暂无相关信息'。"
    "以教练口吻直接回答，不要提及'知识库'。"
    "请详细回答，充分说明训练原理、动作要点和注意事项，至少300字。"
    "\n\n## 回答格式（极其重要）"
    "\n你必须按以下顺序组织回答："
    "\n1. 先用自然语言给出详细的训练建议，说明训练原理、每个动作的组数次数、动作要领和注意事项"
    "\n2. 然后在末尾附上结构化训练计划 JSON"
    "\n3. JSON 之后追加打卡引导语"
    "\n\n结构化 JSON 格式（用 STARTJSON 和 ENDJSON 各占一行包裹）："
    "\nSTARTJSON"
    "\n{{\"plan_name\":\"计划名\",\"exercises\":[{{\"name\":\"动作\",\"sets\":4,\"reps\":\"8-12\",\"rest_sec\":90,\"notes\":\"要点\"}}]}}"
    "\nENDJSON"
    "\n💡 需要我把这个计划加入训练打卡吗？回复\"需要\"即可。"
    "\n{profile}"
    "\n\n参考知识片段：\n{context}"
    "\n\n请根据用户档案中的身体数据和目标来个性化你的建议。在回答中显式提及用户的身高、体重、目标和可用器械，让用户感受到建议是为他们量身定制的。"
)

SUMMARIZE_PROMPT = (
    "将以下对话历史压缩为一段简短摘要（不超过150字），"
    "只保留关键的健身问题和教练建议的核心要点。\n\n"
    "对话历史：\n{history}\n\n摘要："
)

class RAGService:
    """RAG 服务 v4——向量 + BM25 混合检索 + 质量门控 + 对话摘要 + 高频缓存。

    极简检索管道：快速预检 → 混合检索 → 质量门控 → 上下文构建。
    零额外 LLM 调用，端到端 ~3s。
    """

    # 高频缓存
    _cache: Dict[str, Tuple[dict, float]] = {}
    _CACHE_TTL = 3600
    _CACHE_MAX_SIZE = 100

    def __init__(self, llm, vector_repo: VectorRepository):
        self.llm = llm
        self.vector_repo = vector_repo
        self._chain = None

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

    def retrieve(self, user_input: str) -> dict:
        """检索管道 v4——向量 + BM25 混合检索 + 质量门控（极简快速版）。

        移除 HyDE（省一次 LLM 调用，Hit@3 从 96.7% → 93.3%，响应时间减半）
        移除多 Query（评估 Hit@3 86.7% 低于纯向量的 93.3%）
        移除 rerank（评估 Hit@3 70% 比混合检索的 93.3% 差 23pp）

        Returns:
            {"docs": [...], "scores": [...], "category": str,
             "references": [...], "formatted_context": str,
             "search_query": str, "quality": str, "top_score": float}
        """
        _t0 = time.perf_counter()

        # 0. 快速预检——一次向量检索判断是否无关
        from src.config import RAG_FAST_REJECT
        quick_docs, quick_scores, quick_cat, quick_sim = self.vector_repo.search_vector_only(
            user_input, k=3
        )
        _t1 = time.perf_counter()
        if quick_sim < RAG_FAST_REJECT:
            logger.info(f"[TIMING] 快速预检: {_t1-_t0:.2f}s → 拒答 (sim={quick_sim:.3f})")
            return {
                "docs": [], "scores": [], "category": quick_cat,
                "references": [], "formatted_context": "",
                "search_query": user_input,
                "quality": "reject", "top_score": quick_sim,
            }

        # 1. 混合检索（向量 + BM25 → 内部 RRF，一次调用，零 LLM）
        docs, scores, cat, vec_sim = self.vector_repo.search_hybrid(user_input, k=RAG_K)
        best_vec_sim = vec_sim
        _t2 = time.perf_counter()

        # 2. 质量门控——直接使用向量相似度
        from src.config import RAG_REJECT_THRESHOLD, RAG_WEAK_THRESHOLD
        gate_score = best_vec_sim

        if gate_score < RAG_REJECT_THRESHOLD:
            quality = "reject"
            logger.info(f"质量门控 REJECT: score={gate_score:.3f} < {RAG_REJECT_THRESHOLD}")
        elif gate_score < RAG_WEAK_THRESHOLD:
            quality = "weak"
            logger.info(f"质量门控 WEAK: score={gate_score:.3f} < {RAG_WEAK_THRESHOLD}")
        else:
            quality = "ok"

        # 3. 构建 references
        references = []
        for doc, score in zip(docs, scores[:len(docs)]):
            references.append({
                "source": doc.metadata.get("source", "未知"),
                "title": doc.metadata.get("title", ""),
                "content": doc.page_content[:RAG_MAX_CHUNK],
                "score": round(float(score), 4),
                "tags": doc.metadata.get("tags", ""),
            })

        formatted_context = "\n\n---\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n"
            f"[标题: {d.metadata.get('title', '')}]\n"
            f"{d.page_content}"
            for d in docs
        )

        _t3 = time.perf_counter()
        logger.info(
            f"[TIMING] retrieve() 分步耗时: "
            f"预检={_t1-_t0:.2f}s | 混合检索={_t2-_t1:.2f}s | 构建结果={_t3-_t2:.2f}s | "
            f"总计={_t3-_t0:.2f}s"
        )

        return {
            "docs": docs, "scores": scores, "category": cat,
            "references": references, "formatted_context": formatted_context,
            "search_query": user_input,
            "quality": quality, "top_score": gate_score,
        }

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
                # 缓存结果也要检查质量——拒答结果不缓存，每次重新评估
                if cached.get("answer", "").startswith("抱歉"):
                    pass  # 拒答不缓存，走正常流程
                else:
                    logger.info(f"RAG 缓存命中: '{user_input[:30]}...'")
                    return cached

        # 检索（复用 retrieve 方法）
        ret = self.retrieve(user_input)
        references = ret["references"]
        formatted_context = ret["formatted_context"]
        search_query = ret["search_query"]
        category = ret["category"]
        quality = ret.get("quality", "ok")
        top_score = ret.get("top_score", 0.0)

        # 门控：拒答
        if quality == "reject":
            reject_answer = (
                "抱歉，当前知识库中暂无相关信息，建议查阅专业健身书籍或咨询持证教练。"
                "\n\n💡 提示：AI 健身教练目前的知识库主要涵盖训练动作、营养饮食、拉伸恢复和训练原理四大领域，"
                "你可以尝试换个方式提问。"
            )
            result = {"answer": reject_answer, "references": []}
            return result

        # 第四步：对话摘要压缩（超 6 轮自动压缩旧消息）
        if chat_history is None:
            chat_history = []
        else:
            chat_history = self._summarize_history(chat_history)

        # 构建档案文本
        profile_block = f"\n\n## 用户档案\n{profile_text}" if profile_text else ""

        # 第五步：生成回答
        chain = self._get_chain()

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

        # 弱关联时追加免责声明
        if quality == "weak":
            result["answer"] = (
                answer
                + "\n\n---\n⚠️ 知识库中与此问题的相关度较低（rerank score: "
                + f"{top_score:.2f}），以上回答结合了有限知识片段，仅供参考。"
            )

        # 缓存无历史的查询结果（拒答不缓存）
        if not chat_history and quality == "ok":
            self._cache_set(user_input, result)

        logger.info(
            f"RAG v2 完成 [{category}]: "
            f"'{user_input[:30]}...' → {len(references)} refs, {len(answer)} chars"
        )
        return result
