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
WRITE_TOOLS = {"confirm_reminder"}
REQUEST_TIMEOUT = 20

SYSTEM_PROMPT = """你是用户的个人助理，能查两类东西：

【学习】search_courseware —— 在《数据库系统概论》课件里查知识点，回答必须带出处。
【用药】get_today_reminders / get_recent_records —— 查用户自己的服药提醒与记录。
        confirm_reminder —— 把某条提醒标记为已服/稍后/跳过（写操作）。

工作规则：
1. 先判断用户问的是哪一类，再选工具。问知识点才走检索，问自己的药直接查接口。
2. 只依据工具返回的内容回答，工具说没找到就直说没找到，不要用常识补。
3. 写操作（confirm_reminder）必须先向用户确认是哪一条、什么动作，得到同意后再调用。
4. 用药合规红线：只陈述记录里的事实（几点、什么药、是否确认）。
   绝不给出用药建议、不判断药物相互作用、不推荐剂量或替代药。
   遇到这类问题，明确说明「我不能提供用药建议，请咨询医生或药师」。
5. 回答用简体中文，简洁分点；引用课件时保留出处（小节 + 页码）。"""


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
                result: dict = {"ok": False, "error": "用户未确认，已取消，未执行任何写操作"}
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
