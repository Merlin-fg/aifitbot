"""RAG 生成层评估——使用 RAGAS 评估 Faithfulness + Answer Relevancy + Context Precision + Context Recall。

用法:
    python scripts/eval_ragas.py
"""

import json
import sys
import io
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.bot import get_llm
from src.rag import DashScopeEmbeddings
from src.repositories.vector_repo import VectorRepository
from src.services.rag_service import RAGService


def prepare_ragas_dataset(rag: RAGService, queries: List[dict]) -> dict:
    """为 RAGAS 准备评估数据集。

    RAGAS 需要的字段:
    - question: 用户问题
    - answer: RAG 生成的回答
    - contexts: 检索到的文档内容列表
    - ground_truth: 参考答案（可选，Context Recall 需要）
    """
    dataset = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print(f"生成 {len(queries)} 条答案...")
    for i, q in enumerate(queries, 1):
        question = q["question"]
        print(f"  [{i}/{len(queries)}] {question[:40]}...", end=" ", flush=True)

        try:
            result = rag.query(question)
            answer = result.get("answer", "")
            # 用 retrieve 拿到检索到的文档内容
            ret = rag.retrieve(question)
            contexts = [ref["content"] for ref in ret.get("references", [])]

            dataset["question"].append(question)
            dataset["answer"].append(answer)
            dataset["contexts"].append(contexts)
            dataset["ground_truth"].append(q.get("ground_truth", ""))

            print(f"OK ({len(answer)} chars, {len(contexts)} ctx)")
        except Exception as e:
            print(f"FAIL: {e}")

    return dataset


def run_ragas_eval(dataset: dict) -> dict:
    """运行 RAGAS 评估。"""
    from datasets import Dataset as HFDataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    # 构建 HuggingFace Dataset
    hf_dataset = HFDataset.from_dict(dataset)

    # 只对有 ground_truth 的样本评估 context_recall
    metrics = [faithfulness, answer_relevancy, context_precision]
    if any(dataset["ground_truth"]):
        metrics.append(context_recall)

    print(f"\n运行 RAGAS 评估（{len(metrics)} 个指标）...")
    result = evaluate(
        hf_dataset,
        metrics=metrics,
        llm=get_ragas_llm(),
    )

    return result


def get_ragas_llm():
    """获取 RAGAS 需要的 LLM（用于裁判评分）。"""
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from src.config import BASE_URL, API_KEY, MODEL_NAME

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,  # 裁判需要确定性
    )
    return LangchainLLMWrapper(llm)


def main():
    print("=" * 60)
    print("AIFitBot RAG 生成质量评估 (RAGAS)")
    print("=" * 60)

    # 加载标注数据
    with open("data/eval_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"\n加载 {len(queries)} 条评估 query")

    # 初始化 RAG
    print("初始化 RAG 服务...")
    llm = get_llm()
    embedding = DashScopeEmbeddings()
    vr = VectorRepository(embedding=embedding)
    rag = RAGService(llm, vr)

    # 准备数据集
    dataset = prepare_ragas_dataset(rag, queries)

    # 跑 RAGAS 评估
    try:
        result = run_ragas_eval(dataset)

        print("\n" + "=" * 60)
        print("RAGAS 评估结果")
        print("=" * 60)
        for metric_name, score in result.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {metric_name:<25} {score:.4f}  {bar}")

        # 保存
        scores = {k: float(v) for k, v in result.items()}
        with open("scripts/eval_ragas_results.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 scripts/eval_ragas_results.json")

    except Exception as e:
        print(f"\nRAGAS 评估失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n降级方案：跳过 RAGAS，仅输出检索层评估结果。")


if __name__ == "__main__":
    main()
