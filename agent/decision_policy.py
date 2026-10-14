"""Role-specific model decisions and deterministic enforcement for the shared graph."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from jsonschema import Draft202012Validator

from agent.planner import Planner, PLANNER_CONFIG
from agent.safety_gate import ACTIONS, FOLLOWUP_ACTIONS, apply_proposal
from knowledge.tcm_knowledge import TCM_SLOT_LABELS


PROMPT_DIR = Path(__file__).with_name("prompts")
PLANNER_PROMPT = (PROMPT_DIR / "planner_v1.txt").read_text(encoding="utf-8")
REVIEWER_PROMPT = (PROMPT_DIR / "reviewer_v1.txt").read_text(encoding="utf-8")
PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["next_action", "next_focus", "question_targets", "action_reason"],
    "properties": {
        "next_action": {"enum": sorted(ACTIONS)},
        "next_focus": {"type": "string", "maxLength": 80},
        "question_targets": {"type": "array", "maxItems": 3, "uniqueItems": True, "items": {"type": "string"}},
        "action_reason": {"type": "string", "minLength": 1, "maxLength": 200},
    },
}
REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["needs_replan", "suggested_action", "reason", "evidence"],
    "properties": {
        "needs_replan": {"type": "boolean"},
        "suggested_action": {"enum": [""] + sorted(ACTIONS)},
        "reason": {"type": "string", "minLength": 1, "maxLength": 200},
        "evidence": {"type": "array", "maxItems": 5, "uniqueItems": True, "items": {"type": "string", "maxLength": 100}},
    },
}
CASE_FIELDS = set(TCM_SLOT_LABELS) | {
    "chief_complaint", "symptoms", "accompanying_symptoms", "duration", "severity",
    "location", "age", "sex", "past_history", "red_flags", "risk_level",
    "contradictions", "contradiction_fields", "followup_counts", "last_action",
    "pulse_requested", "pulse_declined", "pulse_data",
    "semantic_contract", "current_focus", "current_focus_label", "chief_complaints",
    "chief_complaint_candidates", "intent_status",
    "fact_clarifications", "fact_confirmation_counts",
}


def digest(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compact(value):
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, list):
        return [_compact(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _compact(item) for key, item in value.items()}
    return value


def context_for(case_state, risk_result, plan):
    return {
        "policy_version": "policy_v1",
        "case_state": _compact({key: case_state[key] for key in sorted(CASE_FIELDS) if key in case_state}),
        "risk_result": _compact({key: risk_result.get(key) for key in ("risk", "matched_rules", "reasons")}),
        "missing_slots": list(plan.get("missing_slots") or []),
        "contradiction_fields": case_state.get("contradiction_fields") or [],
        "should_request_pulse": bool(plan.get("should_request_pulse")),
        "clarify_max_attempts": PLANNER_CONFIG["clarify_max_attempts"],
        "allowed_actions": sorted(ACTIONS),
    }


def _field_exists(data, path):
    parts = path.split(".")
    if len(parts) < 2 or parts[0] not in {"case_state", "risk_result", "plan", "action"}:
        return False
    for part in parts:
        if not isinstance(data, dict) or part not in data:
            return False
        data = data[part]
    return True


def _call(client, prompt, context, schema):
    result = client.call([
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))},
    ], require_json=True)
    audit = {"model": asdict(result), "input_sha256": digest(context), "prompt_sha256": digest(prompt),
             "schema_valid": False, "fallback_used": False}
    if not result.ok:
        audit["fallback_used"] = True
        return None, audit
    try:
        proposal = json.loads(result.content)
        if not Draft202012Validator(schema).is_valid(proposal):
            raise ValueError("schema")
    except (ValueError, TypeError):
        audit.update(fallback_used=True, validation_error="schema")
        return None, audit
    audit["schema_valid"] = True
    return proposal, audit


def _targets(plan):
    """Expose raw slot IDs while keeping full missing-state information available."""
    plan = deepcopy(plan)
    action = plan.get("next_action")
    missing = plan.get("missing_slots") or []
    proposed = plan.get("question_targets") or []
    if action in FOLLOWUP_ACTIONS:
        targets = proposed if proposed and all(x in missing for x in proposed) else missing[:1 if action == "ask_followup_single" else 3]
        plan["question_target_fields"] = targets
        plan["question_targets"] = targets
        plan["pending_questions"] = [TCM_SLOT_LABELS.get(x, x) for x in targets]
        if targets:
            plan["next_focus"] = targets[0]
    else:
        plan["question_target_fields"] = []
        plan["question_targets"] = []
    return plan


class DecisionPolicy:
    def __init__(self, *, planner_client=None, reviewer_client=None, name="rule_gate", prompt_version="v1"):
        if prompt_version not in {"v1", "v2", "v3"}:
            raise ValueError("unsupported decision prompt version")
        self.prompt_version = prompt_version
        self.planner_prompt = (PROMPT_DIR / f"planner_{prompt_version}.txt").read_text(encoding="utf-8")
        self.reviewer_prompt = (PROMPT_DIR / f"reviewer_{prompt_version}.txt").read_text(encoding="utf-8")
        self.rule = Planner()
        self.planner_client = planner_client
        self.reviewer_client = reviewer_client
        self.name = name

    def context(self, case_state, risk_result, plan):
        context = context_for(case_state, risk_result, plan)
        if self.prompt_version == 'v3':
            total = sum((case_state.get('followup_counts') or {}).values())
            context['interaction_policy'] = {
                'version':'progress_summary_prompt_v3',
                'progress_summary_due': total >= PLANNER_CONFIG['summarize_total_followups']
                    and len(plan.get('missing_slots') or []) >= PLANNER_CONFIG['summarize_min_missing'],
                'summary_followup_threshold':PLANNER_CONFIG['summarize_total_followups'],
                'summary_missing_threshold':PLANNER_CONFIG['summarize_min_missing']}
        return context

    def summary(self):
        return {"name": self.name, "safety_gate": True, "prompt_version": self.prompt_version,
                "planner": self.planner_client.config.public_summary() if self.planner_client else {"provider": "rule"},
                "reviewer": self.reviewer_client.config.public_summary() if self.reviewer_client else {"provider": "rule"}}

    def plan(self, case_state, risk_result, override_plan=None):
        rule_plan, rule_gate = apply_proposal(self.rule.create_plan(case_state), None, case_state, risk_result)
        audit = {"stage": "plan", "strategy": self.name, "rule_action": rule_plan["next_action"], "gate": rule_gate}
        proposal = None
        if override_plan:
            proposal = {key: override_plan[key] for key in PLAN_SCHEMA["required"]}
            audit["source"] = "review_override"
        elif risk_result.get("risk") == "HIGH":
            audit["source"] = "high_risk_short_circuit"
        elif self.planner_client:
            proposal, model_audit = _call(self.planner_client, self.planner_prompt, self.context(case_state, risk_result, rule_plan), PLAN_SCHEMA)
            audit.update(model_audit, source="model")
        else:
            audit["source"] = "rule"
        plan, gate = apply_proposal(rule_plan, proposal, case_state, risk_result)
        audit["gate"] = gate
        audit["proposal"] = proposal
        audit["fallback_used"] = bool(audit.get("fallback_used") or gate["status"] == "fallback")
        return _targets(plan), audit

    def review(self, case_state, plan, risk_result, action_result):
        hard = self.rule.review_action(case_state, plan, risk_result, action_result)
        audit = {"stage": "review", "strategy": self.name, "hard_review": deepcopy(hard), "source": "rule"}
        review = deepcopy(hard)
        context = self.context(case_state, risk_result, plan)
        context.update(plan={key: plan.get(key) for key in ("next_action", "question_targets", "missing_slots", "contradictions", "confidence", "should_request_pulse")},
                       action={"name": action_result.name, "response": action_result.response[:1600]})
        if not hard["needs_replan"] and risk_result.get("risk") != "HIGH" and self.reviewer_client:
            proposal, model_audit = _call(self.reviewer_client, self.reviewer_prompt, context, REVIEW_SCHEMA)
            audit.update(model_audit, source="model", proposal=proposal)
            if proposal is not None:
                valid = all(_field_exists(context, path) for path in proposal["evidence"])
                valid &= (bool(proposal["suggested_action"]) == proposal["needs_replan"])
                if proposal["needs_replan"]:
                    valid &= bool(proposal["evidence"]) and proposal["suggested_action"] != action_result.name
                if valid:
                    review.update(proposal)
                else:
                    audit.update(validation_error="review_evidence_or_action", fallback_used=True)
        elif risk_result.get("risk") == "HIGH":
            audit["source"] = "high_risk_short_circuit"
        override = None
        if review["needs_replan"]:
            action = review["suggested_action"]
            missing = plan.get("missing_slots") or []
            targets = missing[:1 if action == "ask_followup_single" else 3] if action in FOLLOWUP_ACTIONS else []
            proposal = {"next_action": action, "next_focus": targets[0] if targets else "pattern_summary",
                        "question_targets": targets, "action_reason": review["reason"][:200]}
            # Gate the review immediately, then again on entry to the next plan node.
            safe_base, _ = apply_proposal(self.rule.create_plan(case_state), None, case_state, risk_result)
            replacement, gate = apply_proposal(safe_base, proposal, case_state, risk_result)
            audit["gate"] = gate
            if gate["status"] == "fallback" or replacement["next_action"] == action_result.name:
                review.update(needs_replan=False, suggested_action="")
                audit["fallback_used"] = gate["status"] == "fallback"
            else:
                override = _targets(replacement)
                review["suggested_action"] = replacement["next_action"]
        return review, override, audit
