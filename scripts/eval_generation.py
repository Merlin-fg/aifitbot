"""RAG 生成层评估——LLM-as-Judge 评估忠实度、答案相关性、上下文精确率、上下文召回率。

用法:
    python scripts/eval_generation.py
"""

import json
import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.bot import get_llm
from src.rag import DashScopeEmbeddings
from src.repositories.vector_repo import VectorRepository
from src.services.rag_service import RAGService
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# LLM-as-Judge Prompts（模仿 RAGAS 四个指标）
# ============================================================

FAITHFULNESS_PROMPT = (
    "你的任务是判断一个回答是否忠实于给定的参考文档。\n"
    "给出一个 0 到 1 之间的分数：\n"
    "- 1.0: 回答完全基于参考文档，没有编造任何信息\n"
    "- 0.7-0.9: 大部分基于文档，有少量无关但无害的补充\n"
    "- 0.4-0.6: 有部分编造或与文档矛盾\n"
    "- 0.0-0.3: 大面积编造，与文档无关\n"
    "\n用户问题: {question}\n"
    "\n参考文档:\n{context}\n"
    "\n回答:\n{answer}\n"
    "\n请只输出一个 JSON: {{\"score\": <0到1的分数>, \"reason\": \"<一句话理由>\"}}"
)

ANSWER_RELEVANCY_PROMPT = (
    "你的任务是判断一个回答是否切中用户的问题。\n"
    "给出一个 0 到 1 之间的分数：\n"
    "- 1.0: 回答完全切题，直接回应用户问题\n"
    "- 0.7-0.9: 大部分切题，有少量无关内容\n"
    "- 0.4-0.6: 部分切题，但绕了很多弯\n"
    "- 0.0-0.3: 答非所问\n"
    "\n用户问题: {question}\n"
    "\n回答:\n{answer}\n"
    "\n请只输出一个 JSON: {{\"score\": <0到1的分数>, \"reason\": \"<一句话理由>\"}}"
)

CONTEXT_PRECISION_PROMPT = (
    "你的任务是判断检索到的文档片段是否与用户问题相关。\n"
    "给出一个 0 到 1 之间的分数：\n"
    "- 1.0: 所有片段都与问题直接相关\n"
    "- 0.7-0.9: 大部分相关，有1个不太相关\n"
    "- 0.4-0.6: 一半相关一半无关\n"
    "- 0.0-0.3: 只有少数或没有相关\n"
    "\n用户问题: {question}\n"
    "\n检索到的片段（共{count}个）:\n{context}\n"
    "\n请只输出一个 JSON: {{\"score\": <0到1的分数>, \"reason\": \"<一句话理由>\"}}"
)

CONTEXT_RECALL_PROMPT = (
    "你的任务是判断检索到的文档是否覆盖了参考答案中的所有关键信息。\n"
    "给出一个 0 到 1 之间的分数：\n"
    "- 1.0: 检索片段完全覆盖了参考答案的所有要点\n"
    "- 0.7-0.9: 覆盖了大部分要点，缺1-2个\n"
    "- 0.4-0.6: 覆盖了部分要点\n"
    "- 0.0-0.3: 几乎没有覆盖参考答案的要点\n"
    "\n用户问题: {question}\n"
    "\n检索到的片段（共{count}个）:\n{context}\n"
    "\n参考答案:\n{ground_truth}\n"
    "\n请只输出一个 JSON: {{\"score\": <0到1的分数>, \"reason\": \"<一句话理由>\"}}"
)


# ============================================================
# LLM Judge
# ============================================================

class LLMJudge:
    """用 LLM 做裁判，对 RAG 输出打分。"""

    def __init__(self, llm):
        self.llm = llm

    def judge(self, prompt_text: str) -> float:
        """发送裁判 prompt，解析返回的 JSON 分数。"""
        from langchain_core.messages import HumanMessage
        result = None
        try:
            result = self.llm.invoke([HumanMessage(content=prompt_text)])
            result = result.content if hasattr(result, 'content') else str(result)
            result = result.strip()
            # 解析 JSON（有时 LLM 会多输出）
            result = result.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(result)
            return float(data.get("score", 0.5))
        except Exception as e:
            if result:
                import re
                nums = re.findall(r"0\.\d+|[01]\.\d*", str(result))
                return float(nums[0]) if nums else 0.5
            return 0.5  # API 调用失败时的默认值


# ============================================================
# 评估主流程
# ============================================================

def main():
    print("=" * 60)
    print("AIFitBot RAG 生成质量评估 (LLM-as-Judge)")
    print("=" * 60)

    # 加载数据
    with open("data/eval_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"\n加载 {len(queries)} 条评估 query")

    # 初始化
    print("初始化 RAG 服务...")
    llm = get_llm()
    embedding = DashScopeEmbeddings()
    vr = VectorRepository(embedding=embedding)
    rag = RAGService(llm, vr)

    judge = LLMJudge(llm)

    # 收集所有评估指标
    scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }
    per_query = []

    print(f"\n评估 {len(queries)} 条 query（每条评估 4 个指标）...")
    for i, q in enumerate(queries, 1):
        question = q["question"]
        ground_truth = q.get("ground_truth", "")
        print(f"\n[{i}/{len(queries)}] {question[:50]}...")

        # 获取 RAG 回答和检索上下文（只调一次 retrieve 避免重复 LLM 调用）
        ret = rag.retrieve(question)
        answer = ""  # 从 ret 里的 context 生成答案
        contexts = [ref["content"] for ref in ret.get("references", [])]
        context_text = "\n---\n".join(contexts[:3])

        if ret["quality"] == "reject":
            answer = "抱歉，知识库中暂无相关信息。"
        else:
            # 直接调 LLM 生成，不重复 retrieve
            from langchain_core.messages import HumanMessage, SystemMessage
            from src.services.rag_service import SYSTEM_PROMPT
            msgs = [
                SystemMessage(content=SYSTEM_PROMPT.format(
                    profile="", context=context_text
                )),
                HumanMessage(content=question),
            ]
            gen_result = rag.llm.invoke(msgs)
            answer = gen_result.content if hasattr(gen_result, 'content') else str(gen_result)

        # 四个维度评估
        # 1. Faithfulness
        faith_score = judge.judge(FAITHFULNESS_PROMPT.format(
            question=question, context=context_text, answer=answer[:2000]
        ))
        print(f"  Faithfulness: {faith_score:.2f}")

        # 2. Answer Relevancy
        relevancy_score = judge.judge(ANSWER_RELEVANCY_PROMPT.format(
            question=question, answer=answer[:2000]
        ))
        print(f"  Answer Relevancy: {relevancy_score:.2f}")

        # 3. Context Precision
        precision_score = judge.judge(CONTEXT_PRECISION_PROMPT.format(
            question=question, context=context_text,
            count=len(contexts)
        ))
        print(f"  Context Precision: {precision_score:.2f}")

        # 4. Context Recall（需要 ground_truth）
        recall_score = None
        if ground_truth:
            recall_score = judge.judge(CONTEXT_RECALL_PROMPT.format(
                question=question, context=context_text,
                ground_truth=ground_truth, count=len(contexts)
            ))
            print(f"  Context Recall: {recall_score:.2f}")

        scores["faithfulness"].append(faith_score)
        scores["answer_relevancy"].append(relevancy_score)
        scores["context_precision"].append(precision_score)
        if recall_score is not None:
            scores["context_recall"].append(recall_score)

        per_query.append({
            "question": question[:50],
            "faithfulness": faith_score,
            "answer_relevancy": relevancy_score,
            "context_precision": precision_score,
            "context_recall": recall_score,
        })

    # 汇总
    print("\n" + "=" * 60)
    print("RAG 生成质量评估结果")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        vals = scores[metric]
        if vals:
            avg = sum(vals) / len(vals)
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            print(f"  {metric:<25} {avg:.3f}  {bar}")

    # 保存
    summary = {k: sum(v)/len(v) if v else 0 for k, v in scores.items()}
    out = {"summary": summary, "per_query": per_query}
    with open("scripts/eval_generation_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到 scripts/eval_generation_results.json")


if __name__ == "__main__":
    main()
