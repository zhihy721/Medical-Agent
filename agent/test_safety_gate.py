"""Safety invariants for Stage B planner proposals."""
import unittest
from unittest.mock import Mock, patch

from agent.safety_gate import apply_proposal, assert_response_action


def base_plan(**changes):
    plan = {
        "next_action": "ask_followup_bundle",
        "next_focus": "duration",
        "question_targets": ["duration", "severity"],
        "missing_slots": ["duration", "severity", "location"],
        "should_request_pulse": False,
        "action_reason": "规则追问",
        "contradictions": [],
    }
    plan.update(changes)
    return plan


class SafetyGateTests(unittest.TestCase):
    def test_high_disposition_cannot_be_rewritten_by_responder(self):
        from agent.runtime_utils import render_response
        from agent.router import ActionResult
        llm, memory = Mock(), Mock()
        llm.last_provider_used = 'deepseek'
        llm.call.return_value = '无需就医，休息就好。'
        action = ActionResult(name='risk_escalation', response='建议立即前往急诊。',
                              status='EMERGENCY_ESCALATION', is_final=True, missing_slots=[], render_mode='final')
        result = render_response(llm, memory, action, {}, {'risk': 'HIGH'}, {}, {})
        self.assertEqual(result, action.response)
        llm.call.assert_not_called()
        memory.get_prompt_context_text.assert_not_called()
        action.name = 'final_advice'
        with self.assertRaisesRegex(ValueError, 'HIGH'):
            render_response(llm, memory, action, {}, {'risk': 'HIGH'}, {}, {})

    def test_real_graph_high_response_keeps_deterministic_disposition(self):
        from agent.graph import LangGraphMedicalAgent
        from memory.memory import ConversationMemory
        llm = Mock()
        llm.last_provider_used = 'deepseek'
        llm.extract_json.return_value = {}
        llm.call.return_value = '无需就医，休息就好。'
        memory = ConversationMemory()
        # The graph/extractor/risk/router/respond path is real. External guideline
        # retrieval is fixed to keep this safety fixture offline and reproducible.
        with patch('agent.graph._run_guideline', return_value={'advice': []}), \
             patch('agent.runtime_utils._nearby_hospital_lines', return_value=[]):
            response = LangGraphMedicalAgent(llm, memory).run('我无法呼吸。')
        self.assertIn('就医', response)
        self.assertNotIn('无需就医', response)
        self.assertEqual(memory.get_case_state()['risk_level'], 'HIGH')
        llm.call.assert_not_called()

    def test_high_risk_always_escalates(self):
        plan, audit = apply_proposal(
            base_plan(), {"next_action": "final_advice"},
            {}, {"risk": "HIGH"},
        )
        self.assertEqual(plan["next_action"], "risk_escalation")
        self.assertEqual(audit["rule_override_reason"], "high_risk_requires_escalation")

    def test_invalid_action_returns_rule_plan(self):
        original = base_plan()
        plan, audit = apply_proposal(original, {"next_action": "run_tool"}, {}, {"risk": "LOW"})
        self.assertEqual(plan, original)
        self.assertEqual(audit["status"], "fallback")

    def test_valid_followup_targets_reorder_missing_slots(self):
        plan, audit = apply_proposal(
            base_plan(), {"next_action": "ask_followup_single", "next_focus": "severity",
                          "question_targets": ["severity"], "action_reason": "优先问严重程度"},
            {}, {"risk": "LOW"},
        )
        self.assertEqual(audit["status"], "accepted")
        self.assertEqual(plan["missing_slots"][0], "severity")
        self.assertEqual(plan["question_targets"], ["severity"])

    def test_unknown_target_and_pulse_are_rejected(self):
        plan, audit = apply_proposal(
            base_plan(), {"next_action": "ask_followup_single", "question_targets": ["diagnosis"]},
            {}, {"risk": "LOW"},
        )
        self.assertEqual(audit["rule_override_reason"], "invalid_question_targets")
        self.assertEqual(plan["next_action"], "ask_followup_bundle")
        _, audit = apply_proposal(base_plan(), {"next_action": "request_pulse_input"}, {}, {"risk": "LOW"})
        self.assertEqual(audit["rule_override_reason"], "pulse_gate_closed")

    def test_unresolved_conflict_blocks_final_advice(self):
        rule = base_plan(next_action="final_advice", missing_slots=[])
        plan, audit = apply_proposal(rule, {"next_action": "final_advice"},
                                     {"contradictions": ["年龄前后不一致"]}, {"risk": "LOW"})
        self.assertEqual(plan["next_action"], "clarify_conflict")
        self.assertEqual(audit["status"], "forced")
        plan, audit = apply_proposal(rule, None, {"contradictions": ["年龄前后不一致"]}, {"risk": "LOW"})
        self.assertEqual((plan["next_action"], audit["status"]), ("clarify_conflict", "forced"))

    def test_response_invariant_rejects_bypass(self):
        with self.assertRaisesRegex(ValueError, "HIGH"):
            assert_response_action("final_advice", {"risk": "HIGH"}, {})
        with self.assertRaisesRegex(ValueError, "conflict"):
            assert_response_action("final_advice", {"risk": "LOW"}, {"contradictions": ["年龄"]})
        assert_response_action("risk_escalation", {"risk": "HIGH"}, {})


if __name__ == "__main__":
    unittest.main()
