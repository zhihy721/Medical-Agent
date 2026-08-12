# -*- coding: utf-8 -*-
"""真实 LLM 联调冒烟脚本（不进 CI 基线）。

用 config.env 配置的 provider（默认 auto → DeepSeek）走完整 LangGraph 链路，
回放风寒束表用例（evaluation/cases/04_wind_cold_full_path.json）。真实 LLM 路径的
抽取/改写与 mock 不同，4 轮剧本未必收敛，因此剧本跑完后自动续答直至
final_advice 或达到轮次上限，观测：
  - 每轮实际 provider 与是否降级
  - final 回复是否引用知识库语料（R1 注入行 / R2 prompt 注入效果）
  - 安全边界行是否保留、回复是否疑似被 max_tokens 截断
真实 LLM 输出非确定，本脚本只输出观测结果，不做硬性断言。

用法：
    python evaluation/smoke_real_llm.py
"""
import io
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.factory import create_agent  # noqa: E402
from llm.llm import LLM  # noqa: E402
from llm.prompt import SYSTEM_PROMPT  # noqa: E402
from memory.memory import ConversationMemory  # noqa: E402

CASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases", "04_wind_cold_full_path.json")

# 剧本后自动续答的轮次上限（真实路径收敛比 mock 慢，防止死循环烧 token）
MAX_EXTRA_TURNS = 4
# 按上一轮动作选择续答话术：脉诊请求视为跳过，其余给温和的阴性补充
AUTO_REPLIES = {
    "request_pulse_input": "暂时没有脉诊设备，先不提供脉诊信息了",
    "ask_followup_single": "这方面没有明显不适，其他还好",
    "ask_followup_bundle": "这几方面都没有明显异常，其他还好",
    "summarize_progress": "没有其他要补充的了",
    "clarify_conflict": "以刚才最新说的为准",
}
DEFAULT_AUTO_REPLY = "没有其他要补充的了"

# R1 注入的知识库参考段标题；final 回复含此段说明确定性注入链路生效
KNOWLEDGE_MARKER = "知识库参考"
# R2 prompt 注入的语料线索：真实 LLM 若消费 knowledge_context，大概率提及方剂名
CORPUS_HINTS = ["桂枝汤", "荆防败毒散", "风寒"]
SAFETY_LINE = "不能替代医生面诊"


def load_turns():
    with open(CASE_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle).get("turns", [])


def main():
    llm = LLM(system_prompt=SYSTEM_PROMPT)
    status = llm.get_runtime_status()
    print("真实 LLM 联调冒烟")
    print("=" * 60)
    print(f"configured_provider: {status['configured_provider']}")
    print(f"deepseek_configured: {status['deepseek_configured']}")
    print(f"DEEPSEEK_MAX_TOKENS: {os.getenv('DEEPSEEK_MAX_TOKENS', '512（默认）')}")

    if not status["deepseek_configured"]:
        print("DeepSeek 未配置（缺 API URL 或 KEY），请先填写 config.env")
        return 1

    memory = ConversationMemory(session_id="smoke-real-llm", user_id="smoke-user")
    agent = create_agent(llm, memory, runtime="langgraph")

    turns = load_turns()
    final_response = ""
    final_action = ""
    converged = False

    def play_turn(index, user_text, phase):
        nonlocal final_response, final_action, converged
        response = agent.run(user_text)
        values = agent.get_graph_state().values
        plan = values.get("plan") or {}
        action = plan.get("next_action", "")
        final_response = response
        final_action = action
        if action == "final_advice":
            converged = True
        print("-" * 60)
        print(f"轮 {index} [{action}] {phase} provider={llm.last_provider_used} degraded={llm.degraded}")
        print(f"用户: {user_text}")
        print(f"回复: {response}")
        return action

    for index, turn in enumerate(turns, start=1):
        play_turn(index, turn["user"], "剧本")

    # 真实路径未收敛时自动续答：模拟用户配合回答，推动走到 final_advice 以观测注入效果
    extra_index = len(turns)
    while not converged and extra_index < len(turns) + MAX_EXTRA_TURNS:
        extra_index += 1
        reply = AUTO_REPLIES.get(final_action, DEFAULT_AUTO_REPLY)
        play_turn(extra_index, reply, "续答")

    metrics = llm.get_metrics()
    print("=" * 60)
    print("观测结果（非硬性断言）")
    print("-" * 60)
    print(f"收敛情况: {'已收敛 final_advice（第 %d 轮）' % extra_index if converged else '未收敛（达到轮次上限 %d）' % extra_index}")
    print(f"最终轮动作: {final_action}")
    print(f"最终轮 provider: {llm.last_provider_used}" + ("（已降级 mock，R2 效果未实测）" if llm.last_provider_used != "deepseek" else ""))
    print(f"引用知识库参考段（R1）: {'是' if KNOWLEDGE_MARKER in final_response else '否'}")
    hints_hit = [hint for hint in CORPUS_HINTS if hint in final_response]
    print(f"语料线索命中（R2 观测）: {hints_hit if hints_hit else '无'}")
    print(f"安全边界行保留: {'是' if SAFETY_LINE in final_response else '否'}")
    # 截断粗判：final 回复过短或去掉 Markdown 收尾符（如斜体 *）后仍无结束标点，提示检查 DEEPSEEK_MAX_TOKENS
    tail = final_response.rstrip("*- \n")
    looks_truncated = len(final_response) < 60 or not tail or tail[-1] not in "。！？…」）"
    print(f"疑似截断: {'是（建议调大 DEEPSEEK_MAX_TOKENS，如 1024）' if looks_truncated else '否'}")
    print(f"回复长度: {len(final_response)} 字符")
    print("-" * 60)
    print(
        f"调用指标: calls={metrics['call_count']} errors={metrics['error_count']} "
        f"mock_fallback={metrics['mock_fallback_count']} avg_latency={metrics['avg_latency_ms']}ms "
        f"prompt_tokens={metrics['total_prompt_tokens']} completion_tokens={metrics['total_completion_tokens']}"
    )
    return 0


if __name__ == "__main__":
    # Windows 控制台输出统一 UTF-8，避免中文乱码
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf8"}:
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
