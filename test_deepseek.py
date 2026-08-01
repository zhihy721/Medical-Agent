#!/usr/bin/env python3

from llm.llm import LLM
from llm.prompt import SYSTEM_PROMPT


def test_deepseek_connection():
    llm = LLM(system_prompt=SYSTEM_PROMPT)
    status_before = llm.get_runtime_status()

    print("DeepSeek connection test")
    print("=" * 24)
    print("Configured provider:", status_before["configured_provider"])
    print("DeepSeek configured:", status_before["deepseek_configured"])

    if not status_before["deepseek_configured"]:
        print("DeepSeek is not configured. Fill config.env first.")
        return False

    response = llm.call("请只回复四个字：连接测试成功")
    status_after = llm.get_runtime_status()

    print("Last provider used:", status_after["last_provider_used"])
    if status_after["last_error"]:
        print("Last error:", status_after["last_error"])
    print("Response:", response)

    return status_after["last_provider_used"] == "deepseek"


if __name__ == "__main__":
    test_deepseek_connection()
