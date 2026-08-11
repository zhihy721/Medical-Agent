import atexit
import os
import threading
import time
import uuid
from collections import OrderedDict

from flask import Flask, jsonify, render_template, request, session

from agent.factory import create_agent, get_agent_runtime
from config_manager import (
    DEFAULT_CONFIG,
    apply_config_to_environment,
    get_config_status,
    read_config,
    save_config,
)
from llm.llm import LLM
from llm.prompt import SYSTEM_PROMPT
from memory.memory import ConversationMemory
from memory.profile_store import InMemoryProfileStore, JsonFileProfileStore
from memory.session_store import InMemorySessionStore, JsonFileSessionStore
from observability.events import event_logger, setup_events
from observability.logger import setup_logging
from observability.metrics import summarize_events
from mcp_bridge.adapter import connect_mcp_servers, shutdown_mcp_servers
from tools.registry import default_registry

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "medical-agent-demo-secret")

apply_config_to_environment()
llm = LLM(system_prompt=SYSTEM_PROMPT)
agent_runtime = get_agent_runtime()

# MCP 接入：连接 enabled 的远端服务并注册其工具；任何失败只降级跳过，不影响应用启动
connect_mcp_servers()
atexit.register(shutdown_mcp_servers)


class SessionCache:
    """Web 会话缓存：TTL 过期驱逐 + 容量上限 LRU 淘汰 + 每会话锁。

    避免 agent_sessions 字典无界增长导致内存泄漏，
    并串行化同一会话的并发请求，防止 case_state 与 JSON 会话文件被并发覆盖。
    clock 可注入以便测试。
    """

    def __init__(self, ttl_seconds=1800, max_entries=128, clock=None):
        self._entries = OrderedDict()
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock or time.monotonic
        self._mu = threading.Lock()

    def get_or_create(self, session_id, factory):
        """返回会话条目 {agent, lock, last_access}，不存在则用 factory 创建。"""
        now = self._clock()
        with self._mu:
            self._evict_expired(now)
            entry = self._entries.get(session_id)
            if entry is None:
                while len(self._entries) >= self._max:
                    self._entries.popitem(last=False)
                entry = {"agent": factory(), "lock": threading.Lock(), "last_access": now}
                self._entries[session_id] = entry
            else:
                entry["last_access"] = now
                self._entries.move_to_end(session_id)
            return entry

    def remove(self, session_id):
        with self._mu:
            self._entries.pop(session_id, None)

    def clear(self):
        with self._mu:
            self._entries.clear()

    def _evict_expired(self, now):
        expired = [sid for sid, entry in self._entries.items() if now - entry["last_access"] > self._ttl]
        for sid in expired:
            self._entries.pop(sid, None)

    def __len__(self):
        with self._mu:
            return len(self._entries)


session_cache = SessionCache()

_config = read_config()
_data_dir = _config.get("DATA_DIR", "data")
profile_store = JsonFileProfileStore(data_dir=f"{_data_dir}/profiles")
session_store = JsonFileSessionStore(data_dir=f"{_data_dir}/sessions")

# 初始化统一日志与 JSONL 事件流
_log_dir = _config.get("LOG_DIR", "logs")
setup_logging(_log_dir)
setup_events(_log_dir)


def _reload_runtime():
    global llm, agent_runtime

    apply_config_to_environment()
    llm = LLM(system_prompt=SYSTEM_PROMPT)
    agent_runtime = get_agent_runtime()
    session_cache.clear()


def _config_from_payload(payload):
    payload = payload or {}
    updates = {}
    for key in DEFAULT_CONFIG:
        if key in payload:
            updates[key] = payload[key]
    return updates


def _merged_config_for_test(payload):
    config = read_config()
    updates = _config_from_payload(payload)
    for key, value in updates.items():
        value = str(value or "").strip()
        if key == "DEEPSEEK_API_KEY" and not value:
            continue
        config[key] = value
    return config


def _get_agent():
    user_id = session.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        session["user_id"] = user_id

    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id

    def _factory():
        memory = ConversationMemory(
            profile_store=profile_store,
            user_id=user_id,
            session_store=session_store,
            session_id=session_id,
        )
        return create_agent(
            llm=llm,
            memory=memory,
            runtime=agent_runtime,
            thread_id=session_id,
        )

    return session_cache.get_or_create(session_id, _factory)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/config/status", methods=["GET"])
def api_config_status():
    status_data = get_config_status()
    status_data["llm_status"] = llm.get_runtime_status()
    return jsonify(status_data)


@app.route("/api/config", methods=["POST"])
def api_save_config():
    try:
        payload = request.get_json(silent=True) or {}
        config = save_config(_config_from_payload(payload), preserve_blank_api_key=True)
        _reload_runtime()
        return jsonify(
            {
                "ok": True,
                "message": "Configuration saved.",
                "status": get_config_status(config),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.route("/api/config/test", methods=["POST"])
def api_test_config():
    try:
        config = _merged_config_for_test(request.get_json(silent=True) or {})
        provider = config.get("LLM_PROVIDER", "deepseek").strip().lower()
        if provider == "mock":
            return jsonify({"ok": True, "message": "Mock mode is available."})

        api_url = config.get("DEEPSEEK_API_URL", "").strip()
        api_key = config.get("DEEPSEEK_API_KEY", "").strip()
        if not api_url or not api_key:
            return jsonify({"ok": False, "message": "DeepSeek API URL and API key are required."}), 400

        import requests

        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.get("DEEPSEEK_MODEL", "deepseek-chat"),
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=20,
        )
        response.raise_for_status()
        return jsonify({"ok": True, "message": "Connection test passed."})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.route("/api/config/reload", methods=["POST"])
def api_reload_config():
    _reload_runtime()
    return jsonify({"ok": True, "status": get_config_status()})


@app.route("/status", methods=["GET"])
def status():
    entry = _get_agent()
    agent = entry["agent"]
    with entry["lock"]:
        snapshot = agent.get_case_snapshot()
    return jsonify(
        {
            "case_state": snapshot,
            "llm_status": snapshot.get("llm_status", {}),
            "llm_degraded": bool(snapshot.get("llm_status", {}).get("degraded")),
            "agent_runtime": getattr(agent, "runtime_name", agent_runtime),
            "config_status": get_config_status(),
            "metrics": llm.get_metrics(),
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "请输入有效的症状描述或问题。"})

    try:
        entry = _get_agent()
        agent = entry["agent"]
        # 同一会话串行化：避免并发请求互相覆盖 case_state 与会话文件
        with entry["lock"]:
            response = agent.run(user_input)
            snapshot = agent.get_case_snapshot()
        return jsonify(
            {
                "response": response,
                "case_state": snapshot,
                "llm_status": snapshot.get("llm_status", {}),
                "llm_degraded": bool(snapshot.get("llm_status", {}).get("degraded")),
                "agent_runtime": getattr(agent, "runtime_name", agent_runtime),
                "config_status": get_config_status(),
                "metrics": llm.get_metrics(),
            }
        )
    except Exception as exc:
        return jsonify({"response": f"系统处理失败: {exc}"}), 500


@app.route("/api/debug/trace", methods=["GET"])
def api_debug_trace():
    """调试接口：读取当前会话的执行轨迹事件流与统计摘要。"""
    try:
        limit = max(1, min(int(request.args.get("limit", 200)), 1000))
    except ValueError:
        limit = 200
    session_id = session.get("session_id", "")
    events = event_logger.read_events(session_id=session_id, limit=limit) if session_id else []
    return jsonify(
        {
            "session_id": session_id,
            "events": events,
            "summary": summarize_events(events),
            "tools": default_registry.list_tools(),
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    session_id = session.get("session_id")
    if session_id:
        session_cache.remove(session_id)
    session["session_id"] = str(uuid.uuid4())
    entry = _get_agent()
    agent = entry["agent"]
    snapshot = agent.get_case_snapshot()
    return jsonify(
        {
            "response": "当前会话已重置，你可以开始新的问诊演示。",
            "case_state": snapshot,
            "llm_status": snapshot.get("llm_status", {}),
            "agent_runtime": getattr(agent, "runtime_name", agent_runtime),
            "config_status": get_config_status(),
        }
    )


if __name__ == "__main__":
    host = os.getenv("MEDICAL_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("MEDICAL_AGENT_PORT", "5000"))
    # debug 默认关闭：werkzeug 调试器存在远程执行风险，仅本地调试时通过 APP_DEBUG=true 打开
    debug = os.getenv("APP_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(debug=debug, host=host, port=port)
