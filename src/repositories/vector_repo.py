"""向量数据库仓库——多 collection + BM25 混合召回 + RRF 融合 + 父子切割。"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from rank_bm25 import BM25Okapi

from src.utils.logger import logger
from src.config import CHROMA_DIR

# === 类别 → Collection 名称映射 ===
COLLECTION_MAP = {
    "actions":     "aifitbot_actions",
    "principles":  "aifitbot_principles",
    "nutrition":   "aifitbot_nutrition",
    "stretching":  "aifitbot_stretching",
}

CATEGORY_KEYWORDS = {
    "actions": [
        "动作", "怎么做", "要领", "卧推", "深蹲", "硬拉", "引体向上", "弯举", "划船",
        "俯卧撑", "侧平举", "推举", "卷腹", "平板支撑", "弓箭步", "腿举",
        "后束", "反向飞鸟", "面拉", "飞鸟", "臂屈伸", "弯举",
        "胸", "背", "肩", "手臂", "腿", "臀", "腹", "核心", "肱二", "肱三",
        "练胸", "练背", "练肩", "练腿", "练手臂", "练腹",
        "胸部", "背部", "肩部", "腿部", "腹部", "臀部", "二头肌", "三头肌",
    ],
    "nutrition": [
        "吃", "喝", "营养", "饮食", "蛋白", "碳水", "脂肪", "热量", "卡路里",
        "增肌餐", "减脂餐", "补剂", "肌酸", "蛋白粉", "维生素", "矿物质",
        "食谱", "餐", "空腹", "训练前吃", "训练后吃", "鸡胸", "鸡蛋", "牛奶",
        "减脂饮食", "增肌饮食", "食物", "营养素", "Omega", "水分", "喝水",
    ],
    "principles": [
        "计划", "频率", "周期", "原理", "误区", "恢复", "休息", "睡眠",
        "热身", "伤病", "受伤", "预防", "安全", "过度训练",
        "减脂原理", "增肌原理", "力量训练原理", "策略", "方法",
        "新手", "初级", "入门", "平台期", "进阶", "训练量", "容量",
        "HIIT", "减载", "超级组", "递减组", "周期化", "MRV", "超量恢复",
    ],
    "stretching": [
        "拉伸", "泡沫轴", "筋膜", "放松", "活动度", "柔韧性", "按摩",
        "恢复", "DOMS", "酸痛", "睡眠", "冷热", "冰浴", "筋膜枪",
        "脊柱", "胸椎", "髋关节", "踝关节", "腕关节", "足底", "颈部",
        "减压", "悬挂",
    ],
}

# RRF 融合常数
RRF_K = 60


def _tokenize_cn(text: str) -> List[str]:
    """中英文混合分词：中文按字切分+二元组，英文保留单词。"""
    tokens = []
    buf = ""
    for ch in text:
        if ch.isascii() and (ch.isalpha() or ch.isdigit()):
            buf += ch.lower()
        else:
            if buf:
                tokens.append(buf)
                buf = ""
            if ch.strip():
                tokens.append(ch)
    if buf:
        tokens.append(buf)
    # 生成二元组增强匹配
    bigrams = [tokens[i] + tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams


class VectorRepository:
    """多 Collection 向量库 + BM25 混合检索 + RRF 融合。

    检索策略：
    1. 向量检索（语义匹配）— ChromaDB similarity_search
    2. BM25 检索（关键词匹配）— rank_bm25
    3. RRF 融合两路结果
    4. 支持重排序（外部调用 rag_service 的 reranker）
    """

    def __init__(self, persist_dir: str = CHROMA_DIR,
                 data_dir: str = "data", embedding=None):
        self.persist_dir = persist_dir
        self.data_dir = data_dir
        self.embedding = embedding
        self._stores: dict[str, Chroma] = {}
        # BM25 索引：collection_name → (BM25Okapi, List[LCDocument])
        self._bm25: Dict[str, Tuple[BM25Okapi, List[LCDocument]]] = {}
        # 文档内容缓存：collection_name → List[str]
        self._doc_texts: Dict[str, List[str]] = {}
        # 父子切割：parent_id → {"content": str, "metadata": dict}
        self._parents_file = os.path.join(self.persist_dir, "parents.json")
        self._parents: Dict[str, dict] = {}
        self._load_parents()

    def _get_store(self, collection_name: str = "aifitbot_general") -> Chroma:
        key = collection_name
        if key not in self._stores:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._stores[key] = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding,
                collection_name=collection_name,
            )
        return self._stores[key]

    def _load_parents(self):
        """从磁盘加载父块映射。"""
        if os.path.exists(self._parents_file):
            try:
                with open(self._parents_file, "r", encoding="utf-8") as f:
                    self._parents = json.load(f)
            except Exception as e:
                logger.warning(f"加载 parents.json 失败: {e}")
                self._parents = {}

    def _save_parents(self):
        """持久化父块映射到磁盘。"""
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            with open(self._parents_file, "w", encoding="utf-8") as f:
                json.dump(self._parents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存 parents.json 失败: {e}")

    # ================================================================
    # 文档加载与语义分片
    # ================================================================

    def _load_file(self, file_path: str) -> List[LCDocument]:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(file_path).load()
        return TextLoader(file_path, encoding="utf-8").load()

    def _semantic_split(self, docs: List[LCDocument]) -> Tuple[List[LCDocument], Dict[str, dict]]:
        """父子切割——子块 300 字做检索匹配，父块保留完整章节喂 LLM。

        每个 ## 章节 = 一个父块。父块 ≤300 字的直接复用为子块，
        超过 300 字的用 RecursiveCharacterTextSplitter 切分为多个子块。
        所有子块带 parent_id 元数据指向父块。

        Returns:
            (children, parents): children 存入 ChromaDB 做索引，parents 持久化到磁盘
        """
        children = []
        parents = {}
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=30,
            separators=["\n\n", "\n", "。", "！", "？", "；", " "],
        )
        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get("source", "未知")
            sections = re.split(r'\n(?=## )', content)
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                tags = self._extract_tags(section)
                title_match = re.match(r'^## (.+)', section)
                section_title = title_match.group(1).strip() if title_match else ""
                # 用内容前 100 字 + 标题的 hash 做 parent_id
                pid_key = (section_title + section[:100]).encode("utf-8")
                parent_id = hashlib.md5(pid_key).hexdigest()[:12]

                # 父块：完整章节
                parents[parent_id] = {
                    "content": section,
                    "metadata": {"source": source, "title": section_title, "tags": tags},
                }

                # 子块：检索用
                if len(section) <= 300:
                    children.append(LCDocument(
                        page_content=section,
                        metadata={
                            "source": source, "title": section_title, "tags": tags,
                            "parent_id": parent_id,
                        }
                    ))
                else:
                    sub_texts = child_splitter.split_text(section)
                    for sd in sub_texts:
                        children.append(LCDocument(
                            page_content=sd,
                            metadata={
                                "source": source, "title": section_title, "tags": tags,
                                "parent_id": parent_id,
                            }
                        ))
        logger.info(
            f"父子切割: {len(docs)} 源文档 → "
            f"{len(children)} 子块(检索用) + {len(parents)} 父块(喂LLM用)"
        )
        return children, parents

    def _extract_tags(self, text: str) -> str:
        match = re.search(r'标签:\s*(.+)', text)
        return match.group(1).strip() if match else ""

    # ================================================================
    # 类别判断
    # ================================================================

    def _classify_file(self, filename: str) -> str:
        name = filename.lower()
        if "action" in name or "动作" in name:
            return "actions"
        elif "nutrition" in name or "营养" in name or "饮食" in name:
            return "nutrition"
        elif "principle" in name or "原理" in name:
            return "principles"
        elif "stretch" in name or "拉伸" in name or "恢复" in name:
            return "stretching"
        return "general"

    def _classify_query(self, query: str) -> str:
        scores = {}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query)
            if score > 0:
                scores[cat] = score
        if not scores:
            return "general"
        return max(scores, key=scores.get)

    # ================================================================
    # BM25 索引管理
    # ================================================================

    def _build_bm25(self, collection_name: str):
        """为指定 collection 构建 BM25 索引。从 ChromaDB 读取所有文档。"""
        try:
            store = self._get_store(collection_name)
            collection = store._collection
            results = collection.get(include=["documents", "metadatas"])
            docs = []
            texts = []
            for content, meta in zip(results.get("documents", []),
                                     results.get("metadatas", [])):
                if content:
                    docs.append(LCDocument(page_content=content, metadata=meta or {}))
                    texts.append(content)
            if texts:
                tokenized = [_tokenize_cn(t) for t in texts]
                self._bm25[collection_name] = (BM25Okapi(tokenized), docs)
                self._doc_texts[collection_name] = texts
                logger.info(f"BM25 索引已构建 [{collection_name}]: {len(texts)} 文档")
        except Exception as e:
            logger.warning(f"BM25 索引构建失败 [{collection_name}]: {e}")

    def _get_bm25(self, collection_name: str) -> Tuple[BM25Okapi, List[LCDocument]]:
        """获取 BM25 索引（懒加载）。"""
        if collection_name not in self._bm25:
            self._build_bm25(collection_name)
        return self._bm25.get(collection_name, (None, []))

    def _resolve_parents(self, docs: List[LCDocument]) -> List[LCDocument]:
        """子块 → 父块替换，去重。检索用子块，喂 LLM 用父块完整章节。"""
        if not self._parents:
            return docs
        seen: set = set()
        resolved = []
        for doc in docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid in self._parents:
                if pid not in seen:
                    seen.add(pid)
                    parent = self._parents[pid]
                    resolved.append(LCDocument(
                        page_content=parent["content"],
                        metadata={**doc.metadata, **parent["metadata"]},
                    ))
            else:
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    resolved.append(doc)
        return resolved

    # ================================================================
    # 混合检索 + RRF 融合
    # ================================================================

    def _rrf_fusion(self, vector_results: List[Tuple[LCDocument, float]],
                    bm25_results: List[Tuple[LCDocument, float]],
                    k: int = 3) -> List[Tuple[LCDocument, float]]:
        """RRF (Reciprocal Rank Fusion) 融合两路检索结果。

        公式: RRFscore(d) = sum(1 / (K + rank_i))
        其中 K=60, rank_i 是文档在第 i 路检索中的排名。
        """
        scores: Dict[str, Tuple[LCDocument, float]] = {}

        for rank, (doc, _) in enumerate(vector_results, 1):
            key = doc.page_content[:80]  # 用前80字作为去重键
            rrf = 1.0 / (RRF_K + rank)
            scores[key] = (doc, scores.get(key, (doc, 0))[1] + rrf)

        for rank, (doc, _) in enumerate(bm25_results, 1):
            key = doc.page_content[:80]
            rrf = 1.0 / (RRF_K + rank)
            scores[key] = (doc, scores.get(key, (doc, 0))[1] + rrf)

        # 按 RRF 分数降序排列
        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def search_hybrid(self, query: str, k: int = 5) -> Tuple[List[LCDocument], List[float], str, float]:
        """混合检索：向量 + BM25 → RRF 融合。

        先检索 k*2 条候选，融合后取 top-k 返回。

        Returns:
            (documents, scores, category_used, max_vector_similarity)
        """
        category = self._classify_query(query)
        coll_name = COLLECTION_MAP.get(category, "aifitbot_general")

        # 1. 向量检索（取 2k 候选）
        store = self._get_store(coll_name)
        vec_raw = []
        max_vec_sim = 0.0
        try:
            vec_raw = store.similarity_search_with_score(query, k=k * 2)
            # ChromaDB L2 distance: smaller = more similar
            # For normalized embeddings: L2² = 2 - 2·cos_sim → cos_sim = 1 - L2²/2
            if vec_raw:
                min_dist = min(s for _, s in vec_raw)
                max_vec_sim = max(0.0, 1.0 - min_dist / 2.0)
        except Exception:
            vec_raw = []

        # 2. BM25 检索（取 2k 候选）
        bm25_idx, bm25_docs = self._get_bm25(coll_name)
        bm25_raw = []
        if bm25_idx and bm25_docs:
            tokenized = _tokenize_cn(query)
            bm25_scores = bm25_idx.get_scores(tokenized)
            # 按分数排序取 top 2k
            indexed = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
            for idx, score in indexed[:k * 2]:
                if score > 0:
                    bm25_raw.append((bm25_docs[idx], float(score)))

        # 3. RRF 融合
        if vec_raw and bm25_raw:
            merged = self._rrf_fusion(vec_raw, bm25_raw, k=k)
            logger.info(
                f"混合检索 [{category}] RRF: "
                f"向量 {len(vec_raw)} + BM25 {len(bm25_raw)} → {len(merged)}"
            )
        elif vec_raw:
            merged = [(d, s) for d, s in vec_raw[:k]]
        else:
            merged = [(d, s) for d, s in bm25_raw[:k]]

        docs = [m[0] for m in merged]
        scores = [m[1] for m in merged]
        # 子块 → 父块替换
        docs = self._resolve_parents(docs)
        return docs, scores, category, max_vec_sim

    def search_vector_only(self, query: str, k: int = 3) -> Tuple[List[LCDocument], List[float], str, float]:
        """纯向量检索（不含 BM25）。供 HyDE 使用——长文本不适用 BM25。"""
        category = self._classify_query(query)
        coll_name = COLLECTION_MAP.get(category, "aifitbot_general")
        store = self._get_store(coll_name)
        max_vec_sim = 0.0
        try:
            vec_raw = store.similarity_search_with_score(query, k=k)
            if vec_raw:
                min_dist = min(s for _, s in vec_raw)
                max_vec_sim = max(0.0, 1.0 - min_dist / 2.0)
                docs = self._resolve_parents([d for d, _ in vec_raw])
                scores = [s for _, s in vec_raw[:len(docs)]]
                return docs, scores, category, max_vec_sim
        except Exception as e:
            logger.warning(f"纯向量检索失败: {e}")
        return [], [], category, 0.0

    # ================================================================
    # 公开接口
    # ================================================================

    def add_document(self, stored_name: str, display_name: str = "") -> int:
        file_path = os.path.join(self.data_dir, stored_name)
        raw_docs = self._load_file(file_path)
        source_name = display_name or stored_name
        for doc in raw_docs:
            doc.metadata["source"] = source_name

        children, parents = self._semantic_split(raw_docs)
        self._parents.update(parents)
        self._save_parents()
        category = self._classify_file(display_name or stored_name)

        coll_name = COLLECTION_MAP.get(category, "aifitbot_general")
        store = self._get_store(coll_name)
        store.add_documents(children)

        # 同时存入 general
        if category in COLLECTION_MAP:
            self._get_store("aifitbot_general").add_documents(children)

        # 清除对应 BM25 缓存，下次查询时重建
        self._bm25.pop(coll_name, None)
        if category in COLLECTION_MAP:
            self._bm25.pop("aifitbot_general", None)

        logger.info(f"文档 {stored_name} → {category} 库, {len(children)} 子块")
        return len(children)

    def remove_document(self, stored_name: str) -> bool:
        source = stored_name
        for coll_name in list(COLLECTION_MAP.values()) + ["aifitbot_general"]:
            try:
                store = self._get_store(coll_name)
                collection = store._collection
                results = collection.get(where={"source": source})
                ids = results.get("ids", [])
                if ids:
                    collection.delete(ids=ids)
            except Exception as e:
                logger.warning(f"删除 {coll_name} 中 {source} 失败: {e}")
        # 清除 parents 中匹配 source 的条目
        before = len(self._parents)
        self._parents = {
            k: v for k, v in self._parents.items()
            if v.get("metadata", {}).get("source") != source
        }
        if len(self._parents) < before:
            self._save_parents()
        # 清除所有 BM25 缓存
        self._bm25.clear()
        return True

    def delete_file(self, stored_name: str):
        file_path = os.path.join(self.data_dir, stored_name)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError as e:
            logger.warning(f"文件删除失败: {file_path}, {e}")

    def search(self, query: str, k: int = 3) -> tuple[List[LCDocument], List[float], str, float]:
        """纯向量检索（兼容旧接口）。"""
        return self.search_hybrid(query, k=k)

    def as_retriever(self, k: int = 3):
        return self._get_store("aifitbot_general").as_retriever(search_kwargs={"k": k})

    def rebuild_all(self) -> int:
        """重建知识库——清空所有 collection，重新从 data/ 加载并切割。"""
        import glob
        # 清空所有 collection
        for coll_name in list(COLLECTION_MAP.values()) + ["aifitbot_general"]:
            try:
                store = self._get_store(coll_name)
                ids = store._collection.get().get("ids", [])
                if ids:
                    store._collection.delete(ids=ids)
            except Exception as e:
                logger.warning(f"清空 collection {coll_name} 失败: {e}")
        # 重置状态
        self._stores.clear()
        self._bm25.clear()
        self._doc_texts.clear()
        self._parents.clear()
        # 重新加载所有文档
        total = 0
        patterns = ["**/*.md", "**/*.txt", "**/*.pdf"]
        for pattern in patterns:
            for file_path in glob.glob(os.path.join(self.data_dir, pattern), recursive=True):
                name = os.path.basename(file_path)
                count = self.add_document(name, name)
                total += count
        logger.info(f"知识库重建完成: {total} 子块, {len(self._parents)} 父块")
        return total
