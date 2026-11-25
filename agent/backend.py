"""用药后端的接口封装：登录缓存 token、查今日待办、查服药记录、确认服药。

这里是"工具调用真实 HTTP 接口"的落点，和课件 RAG 的区别在于：
- 需要身份（JWT），所以要登录并缓存 token；
- 返回是结构化字段，要归一化成模型好读的扁平 dict；
- 错误码要显式判断，不能让工具静默返回空。
"""
from __future__ import annotations

import os
import threading

import requests

# 必须先导入 config：它负责把 .env 注入 os.environ，否则下面读到的密码是空
import config  # noqa: F401  (import 顺序有意为之)

BASE_URL = os.getenv("MED_BACKEND_URL", "http://localhost:8080").rstrip("/")
ACCOUNT = os.getenv("MED_ACCOUNT", "admin")
PASSWORD = os.getenv("MED_PASSWORD", "")
TIMEOUT = 10

MEAL = {0: "无关", 1: "餐前", 2: "餐后", 3: "晨起空腹", 4: "睡前"}
STATUS = {
    "WAITING": "等待",
    "TRIGGERED": "已触发",
    "CONFIRMED": "已确认",
    "EXPIRED": "已超时",
    "CANCELLED": "已取消",
}
ACTION = {
    "TAKEN": "已服",
    "DELAYED": "稍后",
    "SKIPPED": "跳过",
    "UNCONFIRMED": "超时未确认",
}

_lock = threading.Lock()
_token: str | None = None


class BackendError(RuntimeError):
    """接口或登录失败，统一抛出，由工具层转成 {"ok": false}。"""


def _login() -> str:
    if not PASSWORD:
        raise BackendError("未配置 MED_PASSWORD，无法登录用药后端")
    resp = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"account": ACCOUNT, "password": PASSWORD},
        timeout=TIMEOUT,
    )
    body = resp.json()
    if body.get("code") != 200:
        raise BackendError(f"登录失败：{body.get('message')}")
    return body["data"]["token"]


def _auth() -> dict:
    """首次调用登录，之后复用；线程安全，避免并发登录。"""
    global _token
    if _token is None:
        with _lock:
            if _token is None:
                _token = _login()
    return {"Authorization": f"Bearer {_token}"}


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{BASE_URL}{path}"
    resp = requests.request(method, url, headers=_auth(), timeout=TIMEOUT, **kwargs)
    try:
        body = resp.json()
    except ValueError:
        raise BackendError(f"接口返回非 JSON（HTTP {resp.status_code}）")
    if body.get("code") != 200:
        raise BackendError(f"接口返回 {body.get('code')}：{body.get('message')}")
    return body


def get_today_reminders() -> dict:
    """查询今日待办提醒（只读）。"""
    body = _request("GET", "/api/reminders/today")
    tasks = body.get("data") or []
    items = []
    for task in tasks:
        items.append(
            {
                "任务ID": task["id"],
                "药品": task["medicineName"],
                "剂量": f'{float(task["dosagePerTime"]):g}{task["dosageUnit"]}',
                "计划时间": task["scheduledAt"][11:16],
                "餐时": MEAL.get(task.get("mealRelation"), "未知"),
                "状态": STATUS.get(task["status"], task["status"]),
                "已确认": bool(task.get("confirmed")),
            }
        )
    return {"ok": True, "count": len(items), "reminders": items}


def get_recent_records(limit: int = 10) -> dict:
    """查询最近服药记录（只读）。"""
    body = _request("GET", "/api/records")
    rows = body.get("data") or []
    items = []
    for row in rows[: max(1, min(limit, 50))]:
        action_time = (row.get("actionTime") or "").replace("T", " ")[:16]
        items.append(
            {
                "药品": row["medicineName"],
                "动作": ACTION.get(row["action"], row["action"]),
                "时间": action_time,
                "延迟分钟": row.get("delayMinutes"),
                "跳过原因": row.get("skipReason"),
            }
        )
    return {"ok": True, "count": len(items), "records": items}


def confirm_reminder(reminder_id: int, action: str = "TAKEN") -> dict:
    """确认服药（写操作）。action 只允许 TAKEN / DELAYED / SKIPPED。"""
    action = (action or "TAKEN").upper()
    if action not in {"TAKEN", "DELAYED", "SKIPPED"}:
        return {"ok": False, "error": f"不支持的动作：{action}，只能是 TAKEN / DELAYED / SKIPPED"}
    body = _request(
        "POST",
        f"/api/reminders/{int(reminder_id)}/confirm",
        json={"action": action, "clientRequestId": f"agent-{reminder_id}-{action}"},
    )
    task = body.get("data") or {}
    return {
        "ok": True,
        "任务ID": int(reminder_id),
        "动作": ACTION.get(action, action),
        "最新状态": STATUS.get(task.get("status", ""), task.get("status")),
    }
