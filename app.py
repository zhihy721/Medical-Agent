import os
import uuid

from flask import Flask, jsonify, render_template, request, session

from agent.factory import create_agent, get_agent_runtime
from llm.llm import LLM
from llm.prompt import SYSTEM_PROMPT
from memory.memory import ConversationMemory
from memory.profile_store import InMemoryProfileStore
from memory.session_store import InMemorySessionStore

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "medical-agent-demo-secret")

llm = LLM(system_prompt=SYSTEM_PROMPT)
agent_sessions = {}
agent_runtime = get_agent_runtime()
profile_store = InMemoryProfileStore()
session_store = InMemorySessionStore()


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


@app.route("/status", methods=["GET"])
def status():
    agent = _get_agent()
    snapshot = agent.get_case_snapshot()
    return jsonify(
        {
            "case_state": snapshot,
            "llm_status": snapshot.get("llm_status", {}),
            "agent_runtime": getattr(agent, "runtime_name", agent_runtime),
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
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
