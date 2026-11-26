"""分层评测：原文 / 改写 / 组合 / 负样本，分别看命中、覆盖与拒答。

设计意图：
- L1 原文型：照课件措辞提问，检验最基本的检索链路；
- L2 改写型：口语化、同义替换，检验语义检索是否只认字面；
- L3 组合型：答案要跨小节拼，检验证据覆盖率（不要求单块命中）；
- L4 负样本：课件里根本没有的主题，必须拒答，检验阈值是否可靠。

指标：
- hit@k / MRR：仅统计 L1、L2（单块可答）；
- 证据覆盖率：L3 的关键词在 top-k 中出现的比例；
- 拒答准确率：L4 中 top1 低于 MIN_SCORE 的比例；
- 阈值余量：正样本最低分与负样本最高分之间的 gap，为负说明当前阈值不安全。

用法：
    python eval.py                 # 只评检索（不需要 API Key）
    python eval.py --llm           # 额外评生成回答
    python eval.py --k 6           # 改 top-k
"""
import argparse
import statistics
import time

from config import MIN_SCORE, TOP_K
from rag import RagEngine

# 层级, 问题, 期望关键词
POSITIVE = [
    # ---------- L1 原文型：直接照课件措辞 ----------
    ("L1", "数据库设计分为哪几个阶段？", ["需求分析", "概念结构", "逻辑结构"]),
    ("L1", "数据库建设的基本规律是什么？", ["三分技术", "七分管理"]),
    ("L1", "数据库设计有哪些典型方法？", ["新奥尔良", "规范设计"]),
    ("L1", "需求分析阶段调查的重点是什么？", ["数据", "处理"]),
    ("L1", "调查用户需求常用哪些方法？", ["跟班作业", "开调查会"]),
    ("L1", "数据字典包括哪些内容？", ["数据项", "数据结构", "数据流"]),
    ("L1", "概念模型有哪些特点？", ["易于理解", "易于更改"]),
    ("L1", "实体之间的联系有哪几种？", ["一对一", "一对多", "多对多"]),
    ("L1", "合并E-R图时会出现哪几类冲突？", ["属性冲突", "命名冲突", "结构冲突"]),
    ("L1", "描述概念模型用什么工具？", ["E-R"]),
    # ---------- L2 改写型：口语化，考验语义而不只是字面 ----------
    ("L2", "建一个数据库一般要经过几步？", ["需求分析", "概念结构", "逻辑结构"]),
    ("L2", "做数据库项目，技术和管理哪个更关键？", ["三分技术", "七分管理"]),
    ("L2", "需求调研的时候主要盯什么？", ["数据", "处理"]),
    ("L2", "数据字典里都记了些啥？", ["数据项", "数据结构"]),
    ("L2", "E-R图合到一起会打架吗？", ["冲突"]),
    ("L2", "概念模型好在哪？", ["易于理解", "易于更改"]),
    # ---------- L3 组合型：答案分散在多个小节 ----------
    ("L3", "从需求分析到概念设计，中间要产出什么？", ["数据字典", "E-R"]),
    ("L3", "各级模式分别在哪个阶段形成？", ["需求分析", "概念设计", "逻辑设计"]),
    ("L3", "E-R图里什么东西该当属性、什么该当实体？", ["属性", "不可分"]),
    ("L3", "局部E-R图合并前要做哪些准备？", ["冲突", "合并"]),
]

# 问题, 说明（为什么课件里没有）
NEGATIVE = [
    ("事务的隔离级别有哪几种？", "事务与并发控制，不在第 7 章"),
    ("B+树索引和哈希索引有什么区别？", "索引结构，不在第 7 章"),
    ("数据库的锁有哪几种粒度？", "封锁机制，不在第 7 章"),
    ("存储过程和触发器有什么区别？", "数据库编程，不在第 7 章"),
    ("SQL 里 LEFT JOIN 和 INNER JOIN 有什么区别？", "SQL 语法，不在第 7 章"),
    ("数据库备份有哪几种方式？", "运维备份，不在第 7 章"),
    ("什么是 MVCC？", "多版本并发控制，不在第 7 章"),
    ("死锁是怎么产生的，怎么预防？", "并发控制，不在第 7 章"),
]


def _first_rank(hits, keywords):
    """返回第一个包含全部关键词的命中排名；未命中返回 None。"""
    for index, hit in enumerate(hits, start=1):
        if all(keyword in hit.text for keyword in keywords):
            return index
    return None


def _coverage(hits, keywords):
    """关键词在 top-k 任意块中出现的比例，用于 L3。"""
    if not keywords:
        return 1.0
    joined = "\n".join(hit.text for hit in hits)
    return sum(1 for keyword in keywords if keyword in joined) / len(keywords)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="同时评测生成回答")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--collection", default=None, help="指定集合名")
    args = parser.parse_args()

    engine = RagEngine(args.collection)

    rows = []
    latencies = []
    ranks = []          # L1/L2 的 rank，用于 hit@k 与 MRR
    top1_scores = []    # 正样本 top1 分数
    coverage_rows = []  # L3 覆盖率
    level_stats = {"L1": [], "L2": [], "L3": []}

    for level, question, keywords in POSITIVE:
        start = time.time()
        hits = engine.retrieve(question, args.k)
        latencies.append(time.time() - start)

        top1 = hits[0].score if hits else 0.0
        top1_scores.append(top1)
        top1_label = hits[0].label if hits else "-"

        if level == "L3":
            cover = _coverage(hits, keywords)
            coverage_rows.append(cover)
            level_stats["L3"].append(cover)
            rows.append((level, question, "覆盖", f"{cover:.0%}", top1_label, f"{top1:.3f}", ""))
            continue

        rank = _first_rank(hits, keywords)
        ranks.append(1 / rank if rank else 0.0)
        level_stats[level].append(1.0 if rank else 0.0)

        answer_ok = ""
        if args.llm:
            try:
                answer = engine.answer_text(question, hits)
                answer_ok = "是" if any(k in answer for k in keywords) else "否"
            except Exception as exc:
                answer_ok = f"错误:{type(exc).__name__}"
        rows.append(
            (
                level,
                question,
                rank if rank else "未命中",
                f"{1 / rank:.2f}" if rank else "-",
                top1_label,
                f"{top1:.3f}",
                answer_ok,
            )
        )

    # ---------- 负样本 ----------
    neg_rows = []
    neg_scores = []
    refused = 0
    for question, note in NEGATIVE:
        start = time.time()
        hits = engine.retrieve(question, args.k)
        latencies.append(time.time() - start)

        top1 = hits[0].score if hits else 0.0
        neg_scores.append(top1)
        ok = engine.is_low_confidence(hits)
        if ok:
            refused += 1
        neg_rows.append(
            (
                question,
                note,
                hits[0].label if hits else "-",
                f"{top1:.3f}",
                "是" if ok else "否",
            )
        )

    # ---------- 汇总 ----------
    total_pos = len(ranks)
    hit_at_k = sum(1 for r in ranks if r > 0)
    hit_rate = hit_at_k / total_pos if total_pos else 0.0
    mrr = sum(ranks) / total_pos if total_pos else 0.0
    avg_cover = statistics.mean(coverage_rows) if coverage_rows else 0.0

    level_line = " / ".join(
        f"{lv} {sum(v) / len(v):.0%}" for lv, v in sorted(level_stats.items()) if v
    )

    refuse_rate = refused / len(NEGATIVE)
    min_pos = min(top1_scores) if top1_scores else 0.0
    max_neg = max(neg_scores) if neg_scores else 0.0
    gap = min_pos - max_neg
    suggested = (min_pos + max_neg) / 2

    p50 = statistics.median(latencies)
    lines = [
        "# 分层检索评测报告",
        "",
        "## 总览",
        "",
        f"- 正样本：{total_pos + len(coverage_rows)} 条"
        f"（L1 原文 {len(level_stats['L1'])} / L2 改写 {len(level_stats['L2'])} / "
        f"L3 组合 {len(level_stats['L3'])}）｜负样本：{len(NEGATIVE)} 条",
        f"- top_k：{args.k}　当前阈值 MIN_SCORE：{MIN_SCORE:.2f}",
        f"- **hit@{args.k}：{hit_rate:.1%}**（{hit_at_k}/{total_pos}，仅 L1/L2）",
        f"- **MRR：{mrr:.3f}**（仅 L1/L2）",
        f"- 分层：{level_line}",
        f"- L3 证据覆盖率：**{avg_cover:.1%}**",
        f"- **负样本拒答准确率：{refuse_rate:.1%}**（{refused}/{len(NEGATIVE)}）",
        f"- 阈值余量：正样本最低 top1 **{min_pos:.3f}** ↔ 负样本最高 top1 **{max_neg:.3f}**，"
        f"gap **{gap:+.3f}**（建议阈值 ≈ {suggested:.2f}）",
        f"- 检索延迟：均值 {statistics.mean(latencies) * 1000:.0f} ms，P50 {p50 * 1000:.0f} ms",
        "",
    ]
    if args.llm:
        answered = sum(1 for r in rows if r[6] == "是")
        lines.append(f"- 回答关键词覆盖率：{answered}/{total_pos} = {answered / total_pos:.1%}")
        lines.append("")

    lines += [
        "## 正样本明细",
        "",
        "| 层 | 问题 | 命中排名 | 倒数排名 | Top1 来源 | Top1 分数 | 回答覆盖 |",
        "|---|---|---|---|---|---|---|",
    ]
    for level, question, rank, rr, label, score, answer_ok in rows:
        lines.append(f"| {level} | {question} | {rank} | {rr} | {label} | {score} | {answer_ok} |")

    lines += [
        "",
        "## 负样本明细（应拒答）",
        "",
        "| 问题 | 为什么算负样本 | Top1 来源 | Top1 分数 | 正确拒答 |",
        "|---|---|---|---|---|",
    ]
    for question, note, label, score, ok in neg_rows:
        lines.append(f"| {question} | {note} | {label} | {score} | {ok} |")

    # ---------- 诊断 ----------
    missed = [r[1] for r in rows if r[2] == "未命中"]
    leaked = [r[0] for r in neg_rows if r[4] == "否"]
    lines += ["", "## 诊断", ""]
    if missed:
        lines.append(f"- 未命中（{len(missed)} 条）：{'；'.join(missed)}")
        lines.append("  - 先看切块是否把答案切开，再看是否需要混合检索（BM25）补专有名词匹配。")
    else:
        lines.append("- 正样本全部命中：若 L1 也是 100%，说明题目仍偏简单，可再加干扰项或更口语的改写。")
    if leaked:
        lines.append(f"- 负样本误答（{len(leaked)} 条，top1 不低于阈值）：{'；'.join(leaked)}")
        lines.append(
            f"  - 这些题 top1 高于 {MIN_SCORE:.2f}，说明阈值偏低或语料里有近似表述，需要调阈值或补充硬负样本。"
        )
    else:
        lines.append("- 负样本全部正确拒答，当前阈值在这批题上是安全的。")
    if gap <= 0:
        lines.append("- **阈值不可分**：正负样本分数区间重叠，当前阈值必然误伤一侧，需要重标定或改进切块。")

    report = "\n".join(lines)
    print(report)
    with open("eval_report.md", "w", encoding="utf-8") as handle:
        handle.write(report + "\n")


if __name__ == "__main__":
    main()
