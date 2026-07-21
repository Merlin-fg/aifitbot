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

HYDE_PROMPT = (
    "你是专业的健身教练。下面的用户问题可能表达不完整或过于简短。"
    "请你把它扩展成一段100-200字的「理想健身教材段落」，"
    "就像你在编写一本健身百科全书中对应的章节那样去写。"
    "不要写成对话形式，要写成教材风格的专业段落。\n"
    "用户问题：{question}\n"
    "教材段落："
)

class RAGService:
    """RAG 服务 v4——向量 + BM25 混合检索 + 质量门控 + 对话摘要 + 高频缓存（极简快速版）。

    优化原则：每个组件必须有评估数据支撑。速度优先，零额外 LLM 调用。
    - 保留 BM25 + 向量混合检索（Hit@3 93.3%）✓
    - 移除 HyDE（Hit@3 96.7%→93.3%，损失 3.4pp，省一次 LLM 调用）✗
    - 移除多 Query（Hit@3 86.7%，负优化）✗
    - 移除 rerank（Hit@3 70%，过拟合）✗
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
        self._hyde_chain = None

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
                top_k: int = 3) -> Tuple[List, List[float]]:
        """阿里云 DashScope gte-rerank 重排序。

        对检索结果做二次精排，提升 Top-k 准确率。

        Returns:
            (reranked_docs, relevance_scores) — 排序后的文档和对齐的关联分数
        """
        if not docs or not self._RERANK_URL:
            return docs, [0.0] * len(docs[:top_k])

        documents = [d.page_content[:500] for d in docs]
        try:
            resp = requests.post(
                self._RERANK_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gte-rerank-v2",
                    "input": {"query": query, "documents": documents},
                    "top_n": top_k,
                    "return_documents": False,
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
                    reranked_scores = [r.get("relevance_score", 0)
                                      for r in ranked[:top_k]
                                      if r["index"] < len(docs)]
                    logger.info(
                        f"重排序: {len(docs)} → {len(reranked)} (gte-rerank), "
                        f"top score: {reranked_scores[0] if reranked_scores else 0:.3f}"
                    )
                    return reranked, reranked_scores
        except Exception as e:
            logger.warning(f"重排序失败（降级使用原始结果）: {e}")
        return docs[:top_k], [0.0] * len(docs[:top_k])

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

    def _get_hyde_chain(self):
        """HyDE 链——生成假设性文档，用于消弭短查询与知识库之间的语义鸿沟。"""
        if self._hyde_chain is None:
            prompt = ChatPromptTemplate.from_messages([
                ("human", HYDE_PROMPT),
            ])
            self._hyde_chain = prompt | self.llm | StrOutputParser()
        return self._hyde_chain

    def _hyde_generate(self, user_input: str) -> str:
        """生成假设性文档（Hypothetical Document）。

        让 LLM 先"虚构"一段理想的知识库内容，用这段内容的 embedding
        去做向量检索，匹配度远高于直接用口语化短 query。

        失败时返回空字符串，调用方降级使用原始 query。
        """
        try:
            chain = self._get_hyde_chain()
            hyde_doc = chain.invoke({"question": user_input}).strip()
            if hyde_doc and len(hyde_doc) >= 30:
                logger.info(f"HyDE 生成: {len(hyde_doc)} chars → '{hyde_doc[:60]}...'")
                return hyde_doc
        except Exception as e:
            logger.warning(f"HyDE 生成失败: {e}")
        return ""

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

    def _rrf_merge(self, result_groups: List[Tuple[List, List]],
                   top_k: int = 5) -> Tuple[List, List]:
        """RRF 融合——合并多路检索结果（HyDE + 原始 query 等）。

        Args:
            result_groups: [(docs, scores), ...]，每路一个元素
            top_k: 返回文档数
        Returns:
            (merged_docs, merged_scores)
        """
        RRF_K = 60
        doc_scores: Dict[str, Tuple[any, float]] = {}

        for docs, _scores in result_groups:
            for rank, doc in enumerate(docs, 1):
                key = doc.page_content[:80]  # 前80字做去重键
                rrf = 1.0 / (RRF_K + rank)
                if key in doc_scores:
                    doc_scores[key] = (doc, doc_scores[key][1] + rrf)
                else:
                    doc_scores[key] = (doc, rrf)

        ranked = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        merged_docs = [r[0] for r in ranked[:top_k]]
        merged_scores = [r[1] for r in ranked[:top_k]]
        return merged_docs, merged_scores

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
