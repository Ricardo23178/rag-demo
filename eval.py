"""评测：检索命中率 / MRR / 延迟；可选跑完整回答并用关键词判断答案覆盖。

用法：
    python eval.py            # 只评检索（不需要 API Key）
    python eval.py --llm      # 同时评回答（需要 DEEPSEEK_API_KEY）
"""
import argparse
import statistics
import time

from rag import RagEngine
from config import TOP_K

# 问题基于课件实际文字内容设计，避开纯图片页
EVAL_SET = [
    ("数据库设计分为哪几个阶段？", ["需求分析", "概念结构", "逻辑结构"]),
    ("数据库建设的基本规律是什么？", ["三分技术", "七分管理"]),
    ("数据库设计有哪些典型方法？", ["新奥尔良", "规范设计"]),
    ("需求分析阶段调查的重点是什么？", ["数据", "处理"]),
    ("调查用户需求常用哪些方法？", ["跟班作业", "开调查会"]),
    ("数据字典包括哪些内容？", ["数据项", "数据结构", "数据流"]),
    ("数据项是什么？", ["最小", "不可再分"]),
    ("概念模型有哪些特点？", ["易于理解", "易于更改", "反映现实"]),
    ("实体之间的联系有哪几种？", ["一对一", "一对多", "多对多"]),
    ("实体与属性划分的原则是什么？", ["属性", "不可分"]),
    ("合并E-R图时会出现哪几类冲突？", ["属性冲突", "命名冲突", "结构冲突"]),
    ("描述概念模型用什么工具？", ["E-R", "E-R模型"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="同时评测生成回答")
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    engine = RagEngine()
    rows = []
    hits_at_k = 0
    reciprocal_ranks = []
    latencies = []

    for question, keywords in EVAL_SET:
        start = time.time()
        hits = engine.retrieve(question, args.k)
        latency = time.time() - start
        latencies.append(latency)

        rank = None
        for index, hit in enumerate(hits, start=1):
            if any(keyword in hit.text for keyword in keywords):
                rank = index
                break
        if rank:
            hits_at_k += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

        answer_ok = ""
        if args.llm:
            try:
                answer = engine.answer_text(question, hits)
                answer_ok = "是" if any(k in answer for k in keywords) else "否"
            except Exception as exc:
                answer_ok = f"错误:{type(exc).__name__}"
        top_label = hits[0].label if hits else "-"
        rows.append((question, rank, top_label, f"{latency:.2f}", answer_ok))

    total = len(EVAL_SET)
    hit_rate = hits_at_k / total
    mrr = sum(reciprocal_ranks) / total
    p50 = statistics.median(latencies)

    lines = [
        "# 检索评测报告",
        "",
        f"- 问题数：{total}",
        f"- top_k：{args.k}",
        f"- 命中率 hit@{args.k}：**{hit_rate:.1%}**（{hits_at_k}/{total}）",
        f"- MRR：**{mrr:.3f}**",
        f"- 检索延迟：均值 {statistics.mean(latencies) * 1000:.0f} ms，中位数 {p50 * 1000:.0f} ms",
        "",
    ]
    if args.llm:
        answered = sum(1 for r in rows if r[4] == "是")
        lines.append(f"- 回答关键词覆盖率：{answered}/{total} = {answered / total:.1%}")
        lines.append("")
    header = "| # | 问题 | 首个命中排名 | Top1 来源 | 检索耗时(s) | 回答覆盖 |"
    sep = "|---|------|--------------|-----------|--------------|----------|"
    lines.extend([header, sep])
    for i, (question, rank, label, latency, answer_ok) in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {question} | {rank if rank else '未命中'} | {label} | {latency} | {answer_ok} |"
        )

    report = "\n".join(lines)
    print(report)
    with open("eval_report.md", "w", encoding="utf-8") as handle:
        handle.write(report + "\n")


if __name__ == "__main__":
    main()
