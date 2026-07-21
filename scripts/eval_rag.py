"""RAG 检索层评估——Hit@3 + MRR + Recall@3 + 消融实验。

用法:
    python scripts/eval_rag.py
"""

import json
import sys
import io
import time
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.bot import get_llm
from src.rag import DashScopeEmbeddings
from src.repositories.vector_repo import VectorRepository
from src.services.rag_service import RAGService


# ============================================================
# 指标计算
# ============================================================

def calc_hit_at_k(retrieved_titles: List[str], expected_titles: List[str], k: int = 3) -> bool:
    """前 K 个结果中是否至少包含一个期待标题。"""
    return any(et in retrieved_titles[:k] for et in expected_titles)


def calc_mrr(retrieved_titles: List[str], expected_titles: List[str]) -> float:
    """Mean Reciprocal Rank——第一个相关结果的排名的倒数。"""
    for rank, title in enumerate(retrieved_titles, 1):
        if title in expected_titles:
            return 1.0 / rank
    return 0.0


def calc_recall_at_k(retrieved_titles: List[str], expected_titles: List[str], k: int = 3) -> float:
    """前 K 个结果中覆盖了多少期待标题的比例。"""
    if not expected_titles:
        return 1.0
    found = sum(1 for et in expected_titles if et in retrieved_titles[:k])
    return found / len(expected_titles)


# ============================================================
# 消融实验：不同的检索配置
# ============================================================

def run_ablation(rag: RAGService, vr: VectorRepository, queries: List[dict]):
    """跑消融实验，逐个关掉组件测指标。"""
    results = {}

    # --- 模式 1: 纯向量检索 ---
    print("\n--- 模式 1: 纯向量 ---")
    results["纯向量"] = evaluate_mode(queries, vr, rag, mode="vector")

    # --- 模式 2: 向量 + BM25 ---
    print("\n--- 模式 2: 向量+BM25 ---")
    results["向量+BM25"] = evaluate_mode(queries, vr, rag, mode="hybrid")

    # --- 模式 3: 向量 + BM25 + 多Query ---
    print("\n--- 模式 3: 向量+BM25+多Query ---")
    results["+多Query"] = evaluate_mode(queries, vr, rag, mode="multi_query")

    # --- 模式 4: 向量 + BM25 + HyDE ---
    print("\n--- 模式 4: 向量+BM25+HyDE ---")
    results["+HyDE"] = evaluate_mode(queries, vr, rag, mode="hyde")

    # --- 模式 5: 完整管道 ---
    print("\n--- 模式 5: 完整管道 ---")
    results["完整管道"] = evaluate_mode(queries, vr, rag, mode="full")

    return results


def evaluate_mode(queries, vr, rag, mode: str) -> dict:
    """对给定模式跑全部 query，计算指标。"""
    hits = 0
    total_mrr = 0.0
    total_recall = 0.0
    total_time = 0.0
    valid = 0  # 有结果的 query 数

    for qi, q in enumerate(queries, 1):
        question = q["question"]
        expected = q["expected_titles"]

        t0 = time.time()
        titles = retrieve_by_mode(question, vr, rag, mode)
        elapsed = time.time() - t0

        if titles:
            total_time += elapsed
            valid += 1
        else:
            # 没检索到结果，按全错算
            pass

        hit = calc_hit_at_k(titles, expected, k=3)
        mrr = calc_mrr(titles, expected)
        recall = calc_recall_at_k(titles, expected, k=3)

        if hit:
            hits += 1
        total_mrr += mrr
        total_recall += recall

        status = "✓" if hit else "✗"
        if qi <= 5 or qi % 10 == 0:
            print(f"  [{status}] Q{qi}: {question[:30]}... → {titles[:2]}")

    n = len(queries)
    return {
        "Hit@3": f"{hits}/{n} ({hits/n*100:.1f}%)",
        "MRR": f"{total_mrr/n:.3f}",
        "Recall@3": f"{total_recall/n:.3f}",
        "Avg Time": f"{total_time/valid:.1f}s" if valid else "N/A",
        "hits_val": hits,
        "mrr_val": total_mrr / n,
        "recall_val": total_recall / n,
    }


def retrieve_by_mode(question: str, vr: VectorRepository, rag: RAGService, mode: str) -> List[str]:
    """按指定模式检索，返回前 3 个文档的标题列表。"""
    titles = []

    if mode == "vector":
        docs, scores, cat, vs = vr.search_vector_only(question, k=3)
        titles = [d.metadata.get("title", "") for d in docs]

    elif mode == "hybrid":
        docs, scores, cat, vs = vr.search_hybrid(question, k=3)
        titles = [d.metadata.get("title", "") for d in docs]

    elif mode == "multi_query":
        variants = rag._multi_query_expand(question)
        all_results = []
        for variant in variants:
            docs, scores, cat, vs = vr.search_hybrid(variant, k=3)
            if docs:
                all_results.append((docs, scores))
        if len(all_results) >= 2:
            merged_docs, merged_scores = rag._multi_query_rrf(all_results, top_k=3)
            titles = [d.metadata.get("title", "") for d in merged_docs]
        elif all_results:
            titles = [d.metadata.get("title", "") for d in all_results[0][0]]

    elif mode == "hyde":
        # HyDE + 原始向量检索的混合
        hyde_doc = rag._hyde_generate(question)
        orig_docs, _, _, _ = vr.search_hybrid(question, k=3)
        all_results = [(orig_docs, [0.5] * len(orig_docs))]
        if hyde_doc:
            hyde_docs, hyde_scores, _, _ = vr.search_vector_only(hyde_doc, k=3)
            if hyde_docs:
                all_results.append((hyde_docs, hyde_scores))
        if len(all_results) >= 2:
            merged_docs, _ = rag._multi_query_rrf(all_results, top_k=3)
            titles = [d.metadata.get("title", "") for d in merged_docs]
        else:
            titles = [d.metadata.get("title", "") for d in orig_docs]

    elif mode == "full":
        ret = rag.retrieve(question)
        titles = [r["title"] for r in ret.get("references", [])]

    return [t for t in titles if t]  # 过滤空标题


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("AIFitBot RAG 检索质量评估 + 消融实验")
    print("=" * 60)

    # 加载标注数据
    with open("data/eval_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"\n加载 {len(queries)} 条评估 query")

    # 初始化
    print("初始化 RAG 服务...")
    llm = get_llm()
    embedding = DashScopeEmbeddings()
    vr = VectorRepository(embedding=embedding)
    rag = RAGService(llm, vr)

    # 跑消融实验
    results = run_ablation(rag, vr, queries)

    # 输出表格
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(f"{'配置':<25} {'Hit@3':<20} {'MRR':<10} {'Recall@3':<12} {'平均耗时':<10}")
    print("-" * 77)
    for mode, r in results.items():
        print(f"{mode:<25} {r['Hit@3']:<20} {r['MRR']:<10} {r['Recall@3']:<12} {r['Avg Time']:<10}")

    # 消融增益
    print("\n消融增益（相对于纯向量）:")
    baseline = results["纯向量"]
    for mode in ["向量+BM25", "+多Query", "+HyDE", "完整管道"]:
        if mode in results:
            delta_hit = results[mode]["hits_val"] - baseline["hits_val"]
            delta_mrr = results[mode]["mrr_val"] - baseline["mrr_val"]
            delta_recall = results[mode]["recall_val"] - baseline["recall_val"]
            print(f"  {mode}: Hit@3 +{delta_hit}, MRR +{delta_mrr:.3f}, Recall@3 +{delta_recall:.3f}")

    # 保存结果
    out_path = Path("scripts/eval_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if not kk.endswith("_val")}
                   for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
