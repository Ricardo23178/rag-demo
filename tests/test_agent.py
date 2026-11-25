"""Agent 层测试：工具 schema、分发、四个护栏。

全部用假模型替身，不消耗 API、不依赖网络，可以在 CI 里跑。
运行：python -m pytest tests -q
"""
import json
import types

import pytest

import agent.loop as loop
from agent import tools


def _tc(name, args, cid="call_1"):
    """构造一个 OpenAI 风格的 tool_call。"""
    payload = {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }
    tc = types.SimpleNamespace()
    tc.id = cid
    tc.function = types.SimpleNamespace(
        name=name, arguments=json.dumps(args, ensure_ascii=False)
    )
    tc.model_dump = lambda: payload
    return tc


def _resp(content=None, tool_calls=None):
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class FakeClient:
    """按顺序吐出预设响应，并记录每次请求的 messages。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("模型被调用的次数超出预期")
        return self._responses.pop(0)


def _patch_client(monkeypatch, responses):
    client = FakeClient(responses)
    monkeypatch.setattr(loop, "_client", lambda: client)
    return client


# ---------------- 工具层 ----------------


def test_tools_schema():
    assert len(tools.TOOLS) == 3
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert names == ["search_courseware", "list_sections", "save_note"]
    for tool in tools.TOOLS:
        assert tool["type"] == "function"
        assert tool["function"]["description"].strip()


def test_list_sections_reads_chroma():
    result = tools.list_sections()
    assert result["count"] > 0
    assert set(result["sections"][0]) == {"section", "pages"}


def test_save_note_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "NOTES_DIR", tmp_path)
    result = tools.save_note("测试/非法:名", "- 第一条内容")
    assert result["saved"] is True
    path = tmp_path / "测试非法名.md"
    assert path.exists()
    assert "第一条内容" in path.read_text(encoding="utf-8")


def test_execute_unknown_tool():
    assert "error" in tools.execute("not_exists", {})


# ---------------- 主循环 ----------------


def test_direct_answer_without_tool(monkeypatch):
    client = _patch_client(monkeypatch, [_resp(content="直接回答")])
    answer, history = loop.run_turn("你好", [], confirm=lambda _: True)
    assert answer == "直接回答"
    assert history[-2:] == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "直接回答"},
    ]
    assert len(client.calls) == 1


def test_tool_call_then_answer(monkeypatch):
    client = _patch_client(
        monkeypatch,
        [
            _resp(tool_calls=[_tc("list_sections", {})]),
            _resp(content="课件共 11 个小节"),
        ],
    )
    answer, _ = loop.run_turn("有哪些小节", [], confirm=lambda _: True)
    assert "11" in answer
    tool_msgs = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and "count" in tool_msgs[0]["content"]


def test_write_tool_needs_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "NOTES_DIR", tmp_path)
    client = _patch_client(
        monkeypatch,
        [
            _resp(tool_calls=[_tc("save_note", {"title": "笔记", "content": "正文"})]),
            _resp(content="已取消，没有写入"),
        ],
    )
    asked = []
    loop.run_turn("存成笔记", [], confirm=lambda prompt: asked.append(prompt) or False)

    assert asked, "写操作前必须询问确认"
    assert not list(tmp_path.glob("*.md")), "用户未确认时不应落盘"
    tool_msgs = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["saved"] is False


def test_write_tool_executes_after_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "NOTES_DIR", tmp_path)
    _patch_client(
        monkeypatch,
        [
            _resp(tool_calls=[_tc("save_note", {"title": "笔记", "content": "正文"})]),
            _resp(content="已保存"),
        ],
    )
    loop.run_turn("存成笔记", [], confirm=lambda _: True)
    assert (tmp_path / "笔记.md").exists()


def test_tool_exception_degrades(monkeypatch):
    def boom(name, args):
        raise RuntimeError("chroma 挂了")

    monkeypatch.setattr(loop, "execute", boom)
    client = _patch_client(
        monkeypatch,
        [
            _resp(tool_calls=[_tc("search_courseware", {"query": "任意"})]),
            _resp(content="兜底回答"),
        ],
    )
    answer, _ = loop.run_turn("问点什么", [], confirm=lambda _: True)

    assert answer == "兜底回答"
    tool_msgs = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
    payload = json.loads(tool_msgs[0]["content"])
    assert "工具执行失败" in payload["error"]


def test_max_steps_guard(monkeypatch):
    responses = [
        _resp(tool_calls=[_tc("list_sections", {})]) for _ in range(loop.MAX_STEPS)
    ]
    client = _patch_client(monkeypatch, responses)
    answer, _ = loop.run_turn("死循环", [], confirm=lambda _: True)

    assert "换个问法" in answer
    assert len(client.calls) == loop.MAX_STEPS


def test_history_is_trimmed(monkeypatch):
    client = _patch_client(monkeypatch, [_resp(content="ok")])
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
        for i in range(20)
    ]
    loop.run_turn("新问题", history, confirm=lambda _: True)

    sent = client.calls[0]["messages"]
    assert len(sent) == 1 + loop.MAX_HISTORY + 1
    assert sent[1]["content"] == "msg14"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
