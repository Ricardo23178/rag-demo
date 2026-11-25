"""Agent 层测试：工具归一化、分发、四个护栏。

全部用假模型和假接口，不消耗 API、不依赖后端启动。
运行：python -m pytest tests -q
"""
import json
import types

import pytest

import agent.backend as backend
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
    assert len(tools.TOOLS) == 4
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert names == [
        "search_courseware",
        "get_today_reminders",
        "get_recent_records",
        "confirm_reminder",
    ]
    for tool in tools.TOOLS:
        assert tool["type"] == "function"
        assert tool["function"]["description"].strip()


def test_today_reminders_normalized(monkeypatch):
    fake = {
        "code": 200,
        "data": [
            {
                "id": 2394,
                "medicineName": "阿莫西林胶囊",
                "dosagePerTime": 1.0,
                "dosageUnit": "粒",
                "mealRelation": 2,
                "scheduledAt": "2026-11-25T20:00:00",
                "status": "TRIGGERED",
                "confirmed": False,
            }
        ],
    }
    monkeypatch.setattr(backend, "_request", lambda *a, **k: fake)

    result = tools.get_today_reminders()
    assert result["ok"] is True and result["count"] == 1
    item = result["reminders"][0]
    assert item["餐时"] == "餐后"
    assert item["状态"] == "已触发"
    assert item["计划时间"] == "20:00"
    assert item["剂量"] == "1粒"


def test_records_respect_limit(monkeypatch):
    rows = [
        {
            "medicineName": "药A",
            "action": "TAKEN",
            "actionTime": "2026-11-25T08:00:00",
            "delayMinutes": None,
            "skipReason": None,
        }
        for _ in range(20)
    ]
    monkeypatch.setattr(backend, "_request", lambda *a, **k: {"code": 200, "data": rows})

    assert tools.get_recent_records(limit=5)["count"] == 5
    assert tools.get_recent_records(limit=999)["count"] == 20  # 上限夹到 50，不报错


def test_confirm_reminder_calls_endpoint(monkeypatch):
    captured = {}

    def fake_request(method, path, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["body"] = kwargs.get("json")
        return {"code": 200, "data": {"status": "CONFIRMED"}}

    monkeypatch.setattr(backend, "_request", fake_request)

    result = tools.confirm_reminder(42, "TAKEN")
    assert result["ok"] is True
    assert result["最新状态"] == "已确认"
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/reminders/42/confirm"
    assert captured["body"]["action"] == "TAKEN"


def test_confirm_reminder_rejects_bad_action():
    result = tools.confirm_reminder(1, "EAT")
    assert result["ok"] is False
    assert "不支持" in result["error"]


def test_execute_converts_backend_error(monkeypatch):
    def boom():
        raise backend.BackendError("登录失败：账号或密码错误")

    monkeypatch.setattr(tools, "get_today_reminders", boom)
    result = tools.execute("get_today_reminders", {})
    assert result["ok"] is False
    assert "登录" in result["error"]


def test_execute_unknown_tool():
    assert tools.execute("not_exists", {})["ok"] is False


# ---------------- 主循环 ----------------


def test_direct_answer_without_tool(monkeypatch):
    _patch_client(monkeypatch, [_resp(content="直接回答")])
    answer, history = loop.run_turn("你好", [], confirm=lambda _: True)

    assert answer == "直接回答"
    assert history[-2:] == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "直接回答"},
    ]


def test_tool_call_then_answer(monkeypatch):
    monkeypatch.setattr(
        tools,
        "get_today_reminders",
        lambda: {"ok": True, "count": 2, "reminders": [{"药品": "阿莫西林"}]},
    )
    client = _patch_client(
        monkeypatch,
        [
            _resp(tool_calls=[_tc("get_today_reminders", {})]),
            _resp(content="今天还有 2 次没确认"),
        ],
    )
    answer, _ = loop.run_turn("我今天要吃什么药", [], confirm=lambda _: True)

    assert "2" in answer
    tool_msgs = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and "reminders" in tool_msgs[0]["content"]


def test_write_tool_needs_confirmation(monkeypatch):
    called = []
    monkeypatch.setattr(
        tools, "confirm_reminder", lambda *a, **k: called.append(a) or {"ok": True}
    )
    client = _patch_client(
        monkeypatch,
        [
            _resp(
                tool_calls=[
                    _tc("confirm_reminder", {"reminder_id": 2394, "action": "TAKEN"})
                ]
            ),
            _resp(content="已取消"),
        ],
    )
    asked = []
    loop.run_turn("我吃了", [], confirm=lambda prompt: asked.append(prompt) or False)

    assert asked, "写操作前必须询问确认"
    assert not called, "未确认时不应真正调用接口"
    tool_msgs = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["ok"] is False


def test_write_tool_executes_after_confirmation(monkeypatch):
    called = []
    monkeypatch.setattr(
        tools,
        "confirm_reminder",
        lambda reminder_id, action="TAKEN": called.append((reminder_id, action))
        or {"ok": True, "最新状态": "已确认"},
    )
    _patch_client(
        monkeypatch,
        [
            _resp(
                tool_calls=[
                    _tc("confirm_reminder", {"reminder_id": 2394, "action": "TAKEN"})
                ]
            ),
            _resp(content="已标记为已服"),
        ],
    )
    loop.run_turn("确认吃了", [], confirm=lambda _: True)

    assert called == [(2394, "TAKEN")]


def test_tool_exception_degrades(monkeypatch):
    def boom(name, args):
        raise RuntimeError("接口 500 了")

    monkeypatch.setattr(loop, "execute", boom)
    client = _patch_client(
        monkeypatch,
        [
            _resp(tool_calls=[_tc("get_today_reminders", {})]),
            _resp(content="兜底回答"),
        ],
    )
    answer, _ = loop.run_turn("问点什么", [], confirm=lambda _: True)

    assert answer == "兜底回答"
    tool_msgs = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
    payload = json.loads(tool_msgs[0]["content"])
    assert "工具执行失败" in payload["error"]


def test_max_steps_guard(monkeypatch):
    monkeypatch.setattr(
        tools, "get_today_reminders", lambda: {"ok": True, "count": 0, "reminders": []}
    )
    responses = [
        _resp(tool_calls=[_tc("get_today_reminders", {})]) for _ in range(loop.MAX_STEPS)
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
