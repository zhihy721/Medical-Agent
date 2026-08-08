"""
Legacy classic controller.

This module is kept as a compatibility fallback and behavior reference while
the project migrates fully to LangGraph orchestration. New orchestration work
should prefer `agent.graph.LangGraphMedicalAgent`.
"""

from agent.planner import Planner
from agent.router import Router
from agent.runtime_utils import (
    extract_case_slots,
    handle_post_pulse_reply,
    render_response,
    sync_action_to_memory,
    sync_plan_to_memory,
    sync_review_to_memory,
)
from tools.guideline_tool import get_guideline_tool
from tools.protocol import unwrap_tool_result
from tools.risk_tool import fallback_risk_result, risk_assessment_tool


# 与 LangGraph 运行时保持一致：调用协议版工具，失败时降级
def _run_risk_assessment(case_state):
    return unwrap_tool_result(risk_assessment_tool(case_state), fallback_risk_result)


def _run_guideline(case_state, risk_result, plan=None):
    return unwrap_tool_result(
        get_guideline_tool(case_state, risk_result, plan),
        lambda: {"summary": "指南生成异常，建议谨慎参考并补充信息。", "advice": []},
    )


class MedicalAgent:
    """
    Legacy hand-written orchestration runtime.

    The project now defaults to the LangGraph runtime. This class remains
    useful for debugging, behavior comparison, and rollback safety.
    """

    MAX_INTERNAL_STEPS = 3

    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.router = Router()
        self.planner = Planner()

    def run(self, user_input):
        self.memory.add_user(user_input)
        handle_post_pulse_reply(self.memory, user_input)

        extracted_slots = extract_case_slots(self.llm, user_input)
        self.memory.update_case(extracted_slots)
        case_state, risk_result, guideline_result, plan, action_result = self._run_agent_loop()

        response = render_response(
            self.llm, self.memory, action_result, case_state, risk_result, guideline_result, plan
        )

        sync_action_to_memory(
            self.memory,
            action_result,
            plan,
            step=self.memory.get_case_state().get("internal_steps", 1),
        )
        self.memory.add_assistant(response)
        return response

    def ingest_pulse_data(self, pulse_data):
        self.memory.update_pulse_data(pulse_data)
        case_state = self.memory.get_case_state()
        risk_result = _run_risk_assessment(case_state)
        plan = self.planner.create_plan(case_state)
        self.memory.update_triage(risk_result=risk_result)
        sync_plan_to_memory(self.memory, plan)
        return self.get_case_snapshot()

    def get_case_snapshot(self):
        snapshot = self.memory.get_case_state()
        snapshot["long_term_profile"] = self.memory.get_long_term_profile()
        snapshot["llm_status"] = self.llm.get_runtime_status()
        snapshot["session_id"] = getattr(self.memory, "session_id", "")
        snapshot["user_id"] = getattr(self.memory, "user_id", "")
        return snapshot

    def _run_agent_loop(self):
        override_action = ""
        final_bundle = None

        for step in range(1, self.MAX_INTERNAL_STEPS + 1):
            case_state = self.memory.get_case_state()
            risk_result = _run_risk_assessment(case_state)
            self.memory.update_triage(risk_result=risk_result)
            case_state = self.memory.get_case_state()

            plan = self.planner.create_plan(case_state)
            if override_action:
                plan["next_action"] = override_action
                plan["action_reason"] = f"{plan.get('action_reason', '')} 自检改判：{override_action}。".strip()
                override_action = ""

            sync_plan_to_memory(self.memory, plan, internal_step=step)
            case_state = self.memory.get_case_state()
            guideline_result = _run_guideline(case_state, risk_result, plan)
            action_result = self.router.route(case_state, plan, risk_result, guideline_result)

            review = self.planner.review_action(case_state, plan, risk_result, action_result)
            sync_review_to_memory(self.memory, review, fallback_stop_reason=plan.get("stop_condition", ""))
            final_bundle = (case_state, risk_result, guideline_result, plan, action_result)

            if review.get("needs_replan") and step < self.MAX_INTERNAL_STEPS:
                override_action = review["suggested_action"]
                continue
            break

        return final_bundle
