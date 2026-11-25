"""Agent 可调用的三个工具：查课件内容、列章节结构、存笔记。

设计原则：只包一层，不重写。检索与生成仍然走 rag.RagEngine，
这里只负责把已有能力包装成模型能调用的函数。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from config import BASE_DIR
from rag import RagEngine

NOTES_DIR = BASE_DIR / "notes"

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
                "在课件内容中检索并回答。用户问知识点、概念、原理、SQL 写法、"
                "某个术语是什么意思时调用。返回答案与出处（小节 + 页码）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索问题或关键词，尽量保留用户原话中的专业术语",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sections",
            "description": (
                "列出课件包含的所有小节及页码范围。用户问「课件有哪些内容」"
                "「目录是什么」「第几节讲什么」时调用，不需要检索。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": (
                "把整理好的内容保存成本地 Markdown 笔记。属于写操作，"
                "调用前必须先向用户确认标题与内容。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "笔记标题，用作文件名"},
                    "content": {"type": "string", "description": "笔记正文，Markdown 格式"},
                },
                "required": ["title", "content"],
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
            "found": False,
            "answer": f"课件中未找到相关内容（最高相似度 {top:.2f}），请不要凭常识回答，改为告知用户没找到。",
            "citations": [],
        }
    answer = engine.answer_text(query, hits)
    return {
        "found": True,
        "answer": answer,
        "citations": [
            {"section": h.section, "pages": h.pages, "score": h.score} for h in hits
        ],
    }


def list_sections() -> dict:
    """查结构：直接从 Chroma 的元数据聚合，不调用模型。"""
    engine = _get_engine()
    data = engine.collection.get(include=["metadatas"])
    buckets: dict[str, list[int]] = {}
    for meta in data.get("metadatas") or []:
        section = str(meta.get("section", "")).strip()
        if not section:
            continue
        raw = str(meta.get("pages", ""))
        nums = [
            int(part)
            for part in raw.replace("—", "-").replace("~", "-").split("-")
            if part.strip().isdigit()
        ]
        buckets.setdefault(section, []).extend(nums)

    sections = [
        {
            "section": name,
            "pages": f"{min(pages)}-{max(pages)}" if pages else "",
        }
        for name, pages in sorted(buckets.items())
    ]
    return {"count": len(sections), "sections": sections}


def save_note(title: str, content: str) -> dict:
    """写文件：真实副作用，所以外层必须先确认。"""
    NOTES_DIR.mkdir(exist_ok=True)
    safe_title = "".join(c for c in title if c not in '\\/:*?"<>|').strip() or "note"
    path = NOTES_DIR / f"{safe_title}.md"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    path.write_text(f"# {title}\n\n> 生成于 {stamp}\n\n{content}\n", encoding="utf-8")
    return {"saved": True, "path": str(path), "title": title}


def execute(name: str, args: dict) -> dict:
    """统一的工具分发入口。"""
    if name == "search_courseware":
        return search_courseware(**args)
    if name == "list_sections":
        return list_sections()
    if name == "save_note":
        return save_note(**args)
    return {"error": f"未知工具：{name}"}
