import os
import uuid

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
from memory.profile_store import InMemoryProfileStore
from memory.session_store import InMemorySessionStore

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "medical-agent-demo-secret")

apply_config_to_environment()
llm = LLM(system_prompt=SYSTEM_PROMPT)
agent_sessions = {}
agent_runtime = get_agent_runtime()
profile_store = InMemoryProfileStore()
session_store = InMemorySessionStore()


def _reload_runtime():
    global llm, agent_runtime

    apply_config_to_environment()
    llm = LLM(system_prompt=SYSTEM_PROMPT)
    agent_runtime = get_agent_runtime()
    agent_sessions.clear()


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

    if session_id not in agent_sessions:
        memory = ConversationMemory(
            profile_store=profile_store,
            user_id=user_id,
            session_store=session_store,
            session_id=session_id,
        )
        agent_sessions[session_id] = create_agent(
            llm=llm,
            memory=memory,
            runtime=agent_runtime,
            thread_id=session_id,
        )

    return agent_sessions[session_id]


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
    agent = _get_agent()
    snapshot = agent.get_case_snapshot()
    return jsonify(
        {
            "case_state": snapshot,
            "llm_status": snapshot.get("llm_status", {}),
            "agent_runtime": getattr(agent, "runtime_name", agent_runtime),
            "config_status": get_config_status(),
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "请输入有效的症状描述或问题。"})

    try:
        agent = _get_agent()
        response = agent.run(user_input)
        snapshot = agent.get_case_snapshot()
        return jsonify(
            {
                "response": response,
                "case_state": snapshot,
                "llm_status": snapshot.get("llm_status", {}),
                "agent_runtime": getattr(agent, "runtime_name", agent_runtime),
                "config_status": get_config_status(),
            }
        )
    except Exception as exc:
        return jsonify({"response": f"系统处理失败: {exc}"}), 500


@app.route("/reset", methods=["POST"])
def reset():
    session_id = session.get("session_id")
    if session_id in agent_sessions:
        del agent_sessions[session_id]
    session["session_id"] = str(uuid.uuid4())
    agent = _get_agent()
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
    app.run(debug=True, host=host, port=port)
