"""Agent 主循环：判断意图 -> 调工具 -> 看结果 -> 再决定，直到给出回答。

四个护栏都在这里：
1. 写操作（save_note）必须经用户确认；
2. 检索无证据时工具自己拒答，循环不替它编；
3. 工具抛错转成结构化错误回填，不让对话崩掉；
4. 最大轮数上限，防止模型反复调工具死循环。
"""
from __future__ import annotations

import json
from typing import Callable

from openai import OpenAI

from agent.tools import TOOLS, execute
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

MAX_STEPS = 5
MAX_HISTORY = 6
WRITE_TOOLS = {"save_note"}
REQUEST_TIMEOUT = 20

SYSTEM_PROMPT = """你是一个课件学习助手，整门课的资料只有一份 PDF 课件。你可以调用三个工具：

- search_courseware：查课件里的具体知识（概念、原理、SQL 写法、术语解释）
- list_sections：查课件目录与章节页码范围
- save_note：把整理好的内容存成本地笔记（写操作）

工作规则：
1. 先判断用户想干什么，再选工具。问知识点才检索，问目录直接列章节，不要一律检索。
2. 只依据工具返回的内容回答；工具返回 found=false 时，直说课件里没有，不要用常识补。
3. 要存笔记时，先把标题和内容给用户看并确认，用户同意后再调用 save_note。
4. 回答用简体中文，分点、简洁；引用课件内容时保留出处（小节 + 页码）。
5. 如果用户的请求与课件无关，直接说明你只能回答这门课的内容。"""


def _client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY，请检查 .env")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _default_confirm(prompt: str) -> bool:
    return input(prompt).strip().lower() in {"y", "yes", "是", "确认", "好"}


def run_turn(
    user_input: str,
    history: list[dict] | None = None,
    confirm: Callable[[str], bool] | None = None,
    verbose: bool = False,
) -> tuple[str, list[dict]]:
    """跑一轮对话，返回 (回答, 新的历史)。

    confirm 可注入：测试时传 lambda _: True 即可跳过交互。
    """
    history = list(history or [])
    confirm = confirm or _default_confirm
    client = _client()

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-MAX_HISTORY:]
    messages.append({"role": "user", "content": user_input})

    for step in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.2,
            timeout=REQUEST_TIMEOUT,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            return answer, history + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ]

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if verbose:
                print(f"  [step {step + 1}] 调用 {name}({args})")

            if name in WRITE_TOOLS and not confirm(
                f"\n即将执行写操作 {name}：{json.dumps(args, ensure_ascii=False)}\n确认吗？(y/N) "
            ):
                result: dict = {"saved": False, "reason": "用户未确认，已取消"}
            else:
                try:
                    result = execute(name, args)
                except Exception as exc:  # 工具异常降级
                    result = {"error": f"工具执行失败：{type(exc).__name__}: {exc}"}

            if verbose:
                brief = json.dumps(result, ensure_ascii=False)
                print(f"  [step {step + 1}] 返回 {brief[:160]}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return "这个请求我绕了太多步还没处理完，换个问法或拆小一点再试。", history
