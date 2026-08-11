from agent.factory import create_agent, get_agent_runtime
from config_manager import read_config
from llm.llm import LLM
from llm.prompt import SYSTEM_PROMPT
from mcp_bridge.adapter import connect_mcp_servers, shutdown_mcp_servers
from memory.memory import ConversationMemory
from observability.events import setup_events
from observability.logger import setup_logging


def main():
    # 初始化日志与事件流，CLI 模式下同样可回溯执行轨迹
    _config = read_config()
    setup_logging(_config.get("LOG_DIR", "logs"))
    setup_events(_config.get("LOG_DIR", "logs"))

    # MCP 接入：与 Web 入口一致，失败只降级跳过不影响 CLI 启动
    connect_mcp_servers()

    llm = LLM(system_prompt=SYSTEM_PROMPT)
    memory = ConversationMemory()
    runtime = get_agent_runtime()
    agent = create_agent(llm=llm, memory=memory, runtime=runtime, thread_id=memory.session_id)

    print(f"智能问诊 Agent 演示已启动，当前 runtime: {agent.runtime_name}，输入 quit 退出。")
    try:
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                break
            if not user_input:
                print("Agent: 请输入有效内容。")
                continue

            response = agent.run(user_input)
            print("Agent:", response)
    finally:
        shutdown_mcp_servers()


if __name__ == "__main__":
    main()
