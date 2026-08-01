from agent.factory import create_agent, get_agent_runtime
from llm.llm import LLM
from llm.prompt import SYSTEM_PROMPT
from memory.memory import ConversationMemory


def main():
    llm = LLM(system_prompt=SYSTEM_PROMPT)
    memory = ConversationMemory()
    runtime = get_agent_runtime()
    agent = create_agent(llm=llm, memory=memory, runtime=runtime, thread_id=memory.session_id)

    print(f"智能问诊 Agent 演示已启动，当前 runtime: {agent.runtime_name}，输入 quit 退出。")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            print("Agent: 请输入有效内容。")
            continue

        response = agent.run(user_input)
        print("Agent:", response)


if __name__ == "__main__":
    main()
