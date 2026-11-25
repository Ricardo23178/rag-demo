"""调试：打印每条问题的 top1 相似度，用于选拒答阈值。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval import EVAL_SET
from rag import RagEngine

IN_DOC = [q for q, _ in EVAL_SET]
OUT_DOC = [
    "索引优化有哪些策略？",
    "期末考试重点是什么？",
    "MySQL 的 MVCC 是怎么实现的？",
    "怎么写一份优秀的简历？",
]

engine = RagEngine()
print("== 文档内问题（应高） ==")
in_scores = []
for q in IN_DOC:
    top = engine.retrieve(q, 1)[0]
    in_scores.append(top.score)
    print(f"{top.score:.3f}  {q}")

print("\n== 文档外问题（应低） ==")
out_scores = []
for q in OUT_DOC:
    top = engine.retrieve(q, 1)[0]
    out_scores.append(top.score)
    print(f"{top.score:.3f}  {q}")

print(f"\n文档内 min={min(in_scores):.3f} max={max(in_scores):.3f}")
print(f"文档外 min={min(out_scores):.3f} max={max(out_scores):.3f}")
