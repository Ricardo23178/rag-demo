"""冒烟测试：给定问题，打印命中片段与模型回答（含耗时）。"""
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from rag import RagEngine

questions = sys.argv[1:] or ["数据库设计分为哪几个阶段？"]

engine = RagEngine()
for question in questions:
    t0 = time.time()
    hits = engine.retrieve(question)
    t1 = time.time()
    print(f"\n=== 问：{question}")
    for i, hit in enumerate(hits, start=1):
        print(f"  [{i}] {hit.label}  sim={hit.score}")
    print(f"  检索耗时 {t1 - t0:.2f}s")
    try:
        answer = engine.answer_text(question, hits)
        print(f"  回答（{time.time() - t1:.2f}s）：{answer}")
    except Exception as exc:
        print(f"  生成失败：{type(exc).__name__}: {exc}")
