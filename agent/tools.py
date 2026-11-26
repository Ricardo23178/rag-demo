"""Agent 可调用的四个工具：查课件知识（RAG）+ 查/改用药数据（真实接口）。

两类数据源并列，正好说明「同一个 Agent 能接不同后端」：
- 课件 → Chroma 向量检索（非结构化）；
- 用药 → Spring Boot REST 接口（结构化 + JWT）。
"""
from __future__ import annotations

from agent import backend
from rag import RagEngine

_engine: RagEngine | None = None


def _get_engine() -> RagEngine:
    """延迟初始化：避免 import 阶段就加载 embedding 模型。"""
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_courseware",
            "description": (
                "在《数据库系统概论》课件中检索并回答。用户问课程知识点、"
                "概念、原理、E-R 模型、数据字典等学习问题时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题或关键词"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_today_reminders",
            "description": (
                "查询当前用户今天的服药提醒与完成情况。用户问「我今天要吃什么药」"
                "「有没有漏吃」「几点吃」时调用。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_records",
            "description": (
                "查询当前用户最近的服药记录（已服 / 稍后 / 跳过 / 超时未确认）。"
                "用户问「我最近吃得怎么样」「昨天有没有按时吃」时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "返回条数，默认 10，最多 50",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_reminder",
            "description": (
                "把某条提醒标记为已服 / 稍后 / 跳过。属于写操作，"
                "调用前必须先向用户确认是哪一条、什么动作。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {
                        "type": "integer",
                        "description": "提醒任务 ID，从 get_today_reminders 的结果里取",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["TAKEN", "DELAYED", "SKIPPED"],
                        "description": "TAKEN=已服，DELAYED=稍后，SKIPPED=跳过",
                    },
                },
                "required": ["reminder_id"],
            },
        },
    },
]


def search_courseware(query: str) -> dict:
    """查知识：复用原有 RAG 链路，低置信度直接拒答。"""
    engine = _get_engine()
    hits = engine.retrieve(query)
    if engine.is_low_confidence(hits):
        top = hits[0].score if hits else 0.0
        return {
            "ok": False,
            "answer": f"课件中未找到相关内容（最高相似度 {top:.2f}），请勿凭常识作答，直接告知没找到。",
            "citations": [],
        }
    answer = engine.answer_text(query, hits)
    return {
        "ok": True,
        "answer": answer,
        "citations": [
            {"section": h.section, "pages": h.pages, "score": h.score} for h in hits
        ],
    }


def get_today_reminders() -> dict:
    return backend.get_today_reminders()


def get_recent_records(limit: int = 10) -> dict:
    return backend.get_recent_records(limit)


def confirm_reminder(reminder_id: int, action: str = "TAKEN") -> dict:
    return backend.confirm_reminder(reminder_id, action)


def execute(name: str, args: dict) -> dict:
    """统一分发。后端错误在这里转成结构化结果，不抛给主循环。"""
    try:
        if name == "search_courseware":
            return search_courseware(**args)
        if name == "get_today_reminders":
            return get_today_reminders()
        if name == "get_recent_records":
            return get_recent_records(**args)
        if name == "confirm_reminder":
            return confirm_reminder(**args)
        return {"ok": False, "error": f"未知工具：{name}"}
    except backend.BackendError as exc:
        return {"ok": False, "error": str(exc)}
