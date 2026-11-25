"""命令行入口：python -m agent.main

对照观察点：
- 问「数据库设计分几个阶段」→ search_courseware（走 RAG，带出处）
- 问「我今天要吃什么药」→ get_today_reminders（真实 HTTP 接口）
- 问「我最近吃得怎么样」→ get_recent_records
- 说「20 点那个我吃了」→ 先反问确认，再 confirm_reminder
"""
from __future__ import annotations

from agent.loop import run_turn


def main() -> None:
    history: list[dict] = []
    print("=" * 52)
    print("  个人助理 Agent  |  学习 + 用药")
    print("-" * 52)
    print("  试试这几句：")
    print("    我今天要吃什么药？")
    print("    我最近吃得怎么样？")
    print("    数据库设计分为哪几个阶段？")
    print("    20 点那个我吃了")
    print("-" * 52)
    print("  直接打字，空行回车退出")
    print("=" * 52)
    print()

    while True:
        try:
            text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        try:
            answer, history = run_turn(text, history, verbose=True)
        except Exception as exc:  # 模型或网络异常，不让整个会话崩掉
            print(f"\n[出错] {type(exc).__name__}: {exc}\n")
            continue
        print(f"\n助手：{answer}\n")


if __name__ == "__main__":
    main()
