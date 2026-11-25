"""命令行入口：python -m agent.main

对照观察点：
- 问「数据库设计分几个阶段」→ 应调用 search_courseware
- 问「这门课有哪些小节」→ 应调用 list_sections（不检索）
- 说「把刚才的回答存成笔记」→ 应先问你确认，再调 save_note
"""
from __future__ import annotations

from agent.loop import run_turn


def main() -> None:
    history: list[dict] = []
    print("课件助手 Agent（agent 分支）已就绪，空行退出")
    print("试试：这门课有哪些小节 / 数据库设计分几个阶段 / 把上面内容存成笔记\n")
    while True:
        try:
            text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        answer, history = run_turn(text, history, verbose=True)
        print(f"\n助手：{answer}\n")


if __name__ == "__main__":
    main()
