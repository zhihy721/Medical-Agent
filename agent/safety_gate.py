"""Deterministic action gate for proposed planner and reviewer decisions."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
from llm.deferred_scope import pending_confirmation, confirmation_plan


ACTIONS = frozenset({
    "risk_escalation", "clarify_conflict", "ask_followup_single",
    "ask_followup_bundle", "summarize_progress", "request_pulse_input",
    "final_advice",
})
FOLLOWUP_ACTIONS = frozenset({"ask_followup_single", "ask_followup_bundle"})


def _result(plan: dict, proposed: str, status: str, reason: str) -> tuple[dict, dict]:
    audit = {
        "proposed_action": proposed,
        "effective_action": plan.get("next_action", ""),
        "status": status,
        "rule_override_reason": reason,
    }
    return plan, audit


def apply_proposal(
    rule_plan: dict,
    proposal: dict | None,
    case_state: dict,
    risk_result: dict,
) -> tuple[dict, dict]:
    """Return a full executable rule plan with at most four model fields replaced."""
    plan = deepcopy(rule_plan)
    raw_action: Any = proposal.get("next_action") if isinstance(proposal, dict) else ""
    proposed = raw_action if isinstance(raw_action, str) else ""
    risk = (risk_result or {}).get("risk") or case_state.get("risk_level", "UNKNOWN")
    if risk == "HIGH":
        plan.update(next_action="risk_escalation", next_focus="emergency_advice",
                    question_targets=[], action_reason="高风险信号优先安全分诊。")
        status = "forced" if proposed and proposed != "risk_escalation" else "rule"
        return _result(plan, proposed, status, "high_risk_requires_escalation")

    request=case_state.get('delivery_turn_request') or {}
    if request.get('version')=='delivery_turn_requests_v1' and request.get('summary'):
        from agent.turn_requests import summary_plan
        return _result(summary_plan(plan,request),proposed,'forced','explicit_progress_request')

    contradictions = case_state.get("contradictions") or rule_plan.get("contradictions") or []
    conflict_field = (case_state.get("contradiction_fields") or rule_plan.get("contradiction_fields") or [""])[0]
    if contradictions and (case_state.get("followup_counts") or {}).get(conflict_field, 0) < 2:
        plan.update(next_action="clarify_conflict", next_focus=conflict_field,
                    question_targets=[], action_reason="先澄清尚未解决的信息冲突。")
        original_action = proposed or rule_plan.get("next_action")
        return _result(plan, proposed, "forced" if original_action != "clarify_conflict" else "rule", "unresolved_conflict")
    candidate = pending_confirmation(case_state)
    if candidate and not contradictions:
        return _result(confirmation_plan(plan, candidate), proposed, "forced", "uncertain_deferred_symptom")
    if proposal is None:
        if contradictions and plan.get("next_action") in {"final_advice", "request_pulse_input"}:
            plan["next_action"] = "clarify_conflict"
            plan["action_reason"] = "尚有未解决的信息冲突，先澄清。"
            return _result(plan, "", "forced", "unresolved_conflict")
        return _result(plan, "", "rule", "")
    if not isinstance(proposal, dict) or set(proposal) - {
        "next_action", "next_focus", "question_targets", "action_reason"
    }:
        return _result(plan, proposed, "fallback", "invalid_proposal_shape")
    if proposed not in ACTIONS:
        return _result(plan, proposed, "fallback", "invalid_action")
    if proposed == "risk_escalation":
        return _result(plan, proposed, "fallback", "non_high_cannot_escalate")
    if contradictions and proposed in {"final_advice", "request_pulse_input"}:
        plan["next_action"] = "clarify_conflict"
        plan["action_reason"] = "尚有未解决的信息冲突，先澄清。"
        return _result(plan, proposed, "forced", "unresolved_conflict")
    if proposed == "clarify_conflict" and not contradictions:
        return _result(plan, proposed, "fallback", "no_conflict")
    missing = list(rule_plan.get("missing_slots") or [])
    if proposed in FOLLOWUP_ACTIONS and not missing:
        return _result(plan, proposed, "fallback", "no_missing_slots")
    if proposed == "summarize_progress" and not missing:
        return _result(plan, proposed, "fallback", "no_missing_slots")
    if proposed == "request_pulse_input" and not rule_plan.get("should_request_pulse"):
        return _result(plan, proposed, "fallback", "pulse_gate_closed")
    if proposed == "final_advice" and missing and rule_plan.get("next_action") != "final_advice":
        return _result(plan, proposed, "fallback", "premature_final_advice")

    targets = proposal.get("question_targets")
    if targets is not None:
        if not isinstance(targets, list) or len(targets) > 3 or len(set(map(str, targets))) != len(targets):
            return _result(plan, proposed, "fallback", "invalid_question_targets")
        if any(not isinstance(target, str) or target not in missing for target in targets):
            return _result(plan, proposed, "fallback", "invalid_question_targets")
    elif proposed in FOLLOWUP_ACTIONS:
        targets = missing[:1] if proposed == "ask_followup_single" else missing[:3]
    else:
        targets = []
    if proposed in FOLLOWUP_ACTIONS and not targets:
        return _result(plan, proposed, "fallback", "empty_question_targets")
    if proposed == "ask_followup_single" and len(targets) != 1:
        return _result(plan, proposed, "fallback", "single_requires_one_target")
    if proposed not in FOLLOWUP_ACTIONS and targets:
        return _result(plan, proposed, "fallback", "non_followup_targets")

    focus = proposal.get("next_focus")
    if focus is not None:
        if not isinstance(focus, str) or len(focus) > 80:
            return _result(plan, proposed, "fallback", "invalid_next_focus")
        valid_focus = set(missing) | set(case_state.get("contradiction_fields") or [])
        valid_focus |= {"emergency_advice", "pattern_summary"}
        if focus and focus not in valid_focus:
            return _result(plan, proposed, "fallback", "invalid_next_focus")
    reason = proposal.get("action_reason")
    if reason is not None and (not isinstance(reason, str) or len(reason) > 200):
        return _result(plan, proposed, "fallback", "invalid_action_reason")

    plan["next_action"] = proposed
    if focus:
        plan["next_focus"] = focus
    if reason:
        plan["action_reason"] = reason
    if proposed in FOLLOWUP_ACTIONS:
        plan["question_targets"] = targets
        plan["missing_slots"] = targets + [slot for slot in missing if slot not in targets]
    else:
        plan["question_targets"] = []
    return _result(plan, proposed, "accepted", "")


def assert_response_action(action_name: str, risk_result: dict, case_state: dict) -> None:
    """Fail closed if graph routing somehow bypassed the action gate."""
    if risk_result.get("risk") == "HIGH" and action_name != "risk_escalation":
        raise ValueError("HIGH risk cannot produce a non-escalation response")
    if case_state.get("contradictions") and action_name == "final_advice":
        raise ValueError("unresolved conflict cannot produce final_advice")
    if (pending_confirmation(case_state) and risk_result.get('risk') != 'HIGH'
            and action_name in {'final_advice', 'request_pulse_input'}):
        raise ValueError('unconfirmed deferred symptom cannot produce final advice or request pulse')
