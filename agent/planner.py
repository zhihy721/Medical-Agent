# 中医知识模块导入
from knowledge.tcm_knowledge import (
    CHIEF_COMPLAINT_PRIORITIES,
    FOUR_DIAGNOSIS_GROUPS,
    GENERAL_TCM_PRIORITIES,
    TCM_SLOT_LABELS,
    get_syndrome_rule,
    infer_tcm_syndromes,
    normalize_term,
)

# Planner 决策阈值统一配置，避免魔法数字散落各处
# 调整策略时只改这里，不需要动决策逻辑本身
PLANNER_CONFIG = {
    # 置信度低于该值视为信息严重不足，即使收敛也要继续打包追问
    "confidence_low_threshold": 0.45,
    # 置信度达到该值视为关键信息基本齐全，可以进入最终建议
    "confidence_ready_threshold": 0.75,
    # 高风险且已有症状时的置信度下限
    "high_risk_confidence_floor": 0.82,
    # 存在信息冲突时的置信度惩罚
    "contradiction_confidence_penalty": 0.28,
    # 请求脉诊前问诊完整度必须达到的下限
    "pulse_completion_min": 0.6,
    # 请求脉诊时允许的最大缺失槽位数
    "pulse_max_missing_slots": 2,
    # 累计追问轮数达到该值且仍缺较多信息时，先总结进展
    "summarize_total_followups": 3,
    # 触发进展总结所需的最少缺失槽位数
    "summarize_min_missing": 2,
    # 证型得分达到该值时，其证据槽位进入定向追问加权
    "syndrome_focus_min_score": 5,
    # 缺失槽位达到该数量时切换打包追问
    "bundle_min_missing": 4,
}

# 每个slot最多允许被追问多少次
# 防止死循环追问
# 没问过：主动问
# 问过一次：换表达问
# 问过两次：停止追问
SLOT_ASK_LIMITS = {
    "chief_complaint": 2,
    "duration": 2,
    "severity": 2,
    "location": 2,
    "cold_heat": 2,
    "pain_character": 2,
    "age": 2,
    "past_history": 2,
    "sweating": 1,
    "thirst": 1,
    "appetite": 1,
    "sleep": 1,
    "stool_urine": 1,
    "emotion": 1,
    "complexion": 1,
    "voice_breath": 1,
    "odor": 1,
    "female_cycle": 1,
}


class Planner:
    # Planner 为纯规则决策器，不依赖 LLM，保证决策确定性与可测试性
    def __init__(self):
        pass

    # 根据当前的case_state，生成一个包含下一步行动建议的计划
    def create_plan(self, case_state):
        risk_level = case_state.get("risk_level") or "UNKNOWN"
        syndrome_candidates = infer_tcm_syndromes(case_state)
        missing_slots, deferred_slots = self._missing_slots(case_state, syndrome_candidates)
        completion_score = self._completion_score(case_state)
        contradictions = case_state.get("contradictions", [])
        contradiction_fields = case_state.get("contradiction_fields", [])
        followup_mode = self._followup_mode(case_state, missing_slots)
        confidence = self._confidence(case_state, risk_level, completion_score, contradictions)
        next_action = self._decide_next_action(case_state, missing_slots, risk_level, contradictions, confidence, followup_mode)
        next_focus = self._decide_next_focus(missing_slots, risk_level, contradiction_fields, next_action)
        hypotheses = [item["name"] for item in syndrome_candidates[:3]]
        pending_questions = [TCM_SLOT_LABELS.get(slot, slot) for slot in missing_slots[:3]]
        should_request_pulse = self._should_request_pulse(case_state, risk_level, completion_score, missing_slots)

        return {
            "intent": "tcm_triage_with_pulse",
            "four_diagnosis_focus": self._build_four_diagnosis_focus(missing_slots),
            "missing_slots": missing_slots,
            "deferred_slots": deferred_slots,
            "next_focus": next_focus,
            "next_action": next_action,
            "action_reason": self._build_action_reason(case_state, next_action, missing_slots, contradictions, should_request_pulse),
            "risk_level": risk_level,
            "followup_mode": followup_mode,
            "followup_reason": self._build_followup_reason(case_state, missing_slots, syndrome_candidates),
            "completion_score": completion_score,
            "completion_label": self._completion_label(completion_score),
            "confidence": confidence,
            "question_targets": pending_questions,
            "pending_questions": pending_questions,
            "syndrome_candidates": syndrome_candidates,
            "hypotheses": hypotheses,
            "contradictions": contradictions,
            "contradiction_fields": contradiction_fields,
            "should_request_pulse": should_request_pulse,
            "stop_condition": self._stop_condition(next_action, risk_level, missing_slots, contradictions),
            "goal_progress": self._goal_progress(next_action, risk_level, missing_slots, should_request_pulse),
            "chief_complaint": case_state.get("chief_complaint", ""),
        }

    # 自检当前行动结果是否合理，并给出改进建议
    # 防止Planner出现策略性错误
    def review_action(self, case_state, plan, risk_result, action_result):
        missing_slots = plan.get("missing_slots", [])
        contradictions = plan.get("contradictions", [])
        confidence = plan.get("confidence", 0.0)
        suggested_action = ""
        reason = "当前动作与状态一致，可直接执行。"

        if action_result.name == "final_advice" and contradictions:
            suggested_action = "clarify_conflict"
            reason = "存在前后不一致信息，先澄清再给建议更稳妥。"
        elif action_result.name == "final_advice" and missing_slots and risk_result.get("risk") != "HIGH" and confidence < PLANNER_CONFIG["confidence_ready_threshold"]:
            suggested_action = "ask_followup_bundle"
            reason = "仍缺少关键四诊信息，建议继续追问而不是过早收束。"
        elif action_result.name == "request_pulse_input" and risk_result.get("risk") == "HIGH":
            suggested_action = "risk_escalation"
            reason = "高风险场景下应优先安全分诊，不能等待脉诊补充。"
        elif action_result.name.startswith("ask_followup") and risk_result.get("risk") == "HIGH":
            suggested_action = "risk_escalation"
            reason = "高风险场景下应优先安全分诊，不宜继续常规追问。"
        elif action_result.name == "request_pulse_input" and plan.get("completion_score", 0) < PLANNER_CONFIG["pulse_completion_min"]:
            suggested_action = "ask_followup_bundle"
            reason = "问诊完整度仍较低，应先补充关键信息再考虑脉诊。"
        elif action_result.name.startswith("ask_followup") and not missing_slots and confidence >= PLANNER_CONFIG["confidence_ready_threshold"]:
            suggested_action = "final_advice"
            reason = "关键信息已基本齐全，可以进入整理与建议阶段。"

        return {
            "needs_replan": bool(suggested_action),
            "suggested_action": suggested_action,
            "reason": reason,
            "confidence": confidence,
            "open_questions": plan.get("pending_questions", []),
        }

    # 根据当前case_state中已收集的信息
    # 判断还缺哪些关键槽位，并根据优先级排序，返回待补充的槽位列表和可以暂缓的槽位列表
    # 排序规则：证型定向证据槽位优先 > 追问次数少的优先 > 原始优先级顺序
    def _missing_slots(self, case_state, syndrome_candidates=None):
        chief_complaint = case_state.get("chief_complaint", "")
        priorities = list(GENERAL_TCM_PRIORITIES)
        priorities.extend(self._chief_complaint_priorities(chief_complaint))
        followup_counts = case_state.get("followup_counts", {})
        boosted_slots = self._syndrome_focus_slots(syndrome_candidates or [])

        queue = []
        deferred = []
        for slot in priorities:
            if slot in queue:
                continue
            if self._has_value(case_state.get(slot)):
                continue
            if not self._should_ask_slot(slot, followup_counts):
                deferred.append(slot)
                continue
            queue.append(slot)

        sorted_queue = sorted(
            queue,
            key=lambda slot: (0 if slot in boosted_slots else 1, followup_counts.get(slot, 0), queue.index(slot)),
        )
        return sorted_queue, deferred

    # 主诉模糊匹配：先术语归一，再做包含匹配，替代精确查表
    # 用户主诉是自由文本（如“咳嗽得厉害”），精确匹配经常落空
    def _chief_complaint_priorities(self, chief_complaint):
        normalized = normalize_term((chief_complaint or "").strip())
        if not normalized:
            return []
        matched = []
        for key, slots in CHIEF_COMPLAINT_PRIORITIES.items():
            if key and (key in normalized or normalized in key):
                matched.extend(slots)
        return matched

    # 证型定向追问：高置信证型（score 达标）所需的证据槽位优先补充
    # 让追问朝辨证方向收敛，而不是只按通用顺序问
    def _syndrome_focus_slots(self, syndrome_candidates):
        boosted = set()
        for candidate in syndrome_candidates:
            if candidate.get("score", 0) < PLANNER_CONFIG["syndrome_focus_min_score"]:
                continue
            rule = get_syndrome_rule(candidate.get("name", ""))
            if rule:
                boosted.update(rule.get("evidence", {}).keys())
        return boosted

    # 把缺失的信息映射到中医四诊结构
    def _build_four_diagnosis_focus(self, missing_slots):
        focus = {}
        for diagnosis, slots in FOUR_DIAGNOSIS_GROUPS.items():
            labels = [TCM_SLOT_LABELS.get(slot, slot) for slot in missing_slots if slot in slots]
            if labels:
                focus[diagnosis] = labels[:3]
        return focus

    # 下一步行动的决策逻辑
    def _decide_next_focus(self, missing_slots, risk_level, contradiction_fields, next_action):
        if risk_level == "HIGH":
            return "emergency_advice"
        if next_action == "clarify_conflict" and contradiction_fields:
            return contradiction_fields[0]
        if not missing_slots:
            return "pattern_summary"
        return missing_slots[0]

    # 下一步行动决策：安全 > 冲突 > 补缺 > 脉诊 > 收束
    def _decide_next_action(self, case_state, missing_slots, risk_level, contradictions, confidence, followup_mode):
        if risk_level == "HIGH":
            return "risk_escalation"
        if contradictions:
            return "clarify_conflict"
        if missing_slots:
            total_followups = sum(case_state.get("followup_counts", {}).values())
            if total_followups >= PLANNER_CONFIG["summarize_total_followups"] and len(missing_slots) >= PLANNER_CONFIG["summarize_min_missing"]:
                return "summarize_progress"
            if followup_mode == "bundle":
                return "ask_followup_bundle"
            return "ask_followup_single"
        if self._should_request_pulse(case_state, risk_level, self._completion_score(case_state), missing_slots):
            return "request_pulse_input"
        if confidence < PLANNER_CONFIG["confidence_low_threshold"]:
            return "ask_followup_bundle"
        return "final_advice"

    def _followup_mode(self, case_state, missing_slots):
        if not missing_slots:
            return "none"

        followup_counts = case_state.get("followup_counts", {})
        first_slot = missing_slots[0]
        repeated_attempts = followup_counts.get(first_slot, 0)

        if repeated_attempts >= 2 or len(missing_slots) >= PLANNER_CONFIG["bundle_min_missing"]:
            return "bundle"
        return "single"

    # 计算当前信息收集的完整度，作为后续决策的参考
    def _confidence(self, case_state, risk_level, completion_score, contradictions):
        confidence = completion_score
        if risk_level == "HIGH" and case_state.get("symptoms"):
            confidence = max(confidence, PLANNER_CONFIG["high_risk_confidence_floor"])
        if case_state.get("pulse_summary"):
            confidence += 0.08
        if not case_state.get("chief_complaint"):
            confidence -= 0.12
        if contradictions:
            confidence -= PLANNER_CONFIG["contradiction_confidence_penalty"]
        return round(min(max(confidence, 0.05), 0.95), 2)

    # 判断是否满足请求补充脉诊输入的条件
    def _should_request_pulse(self, case_state, risk_level, completion_score, missing_slots):
        if risk_level == "HIGH":
            return False
        if case_state.get("pulse_summary"):
            return False
        if case_state.get("pulse_declined"):
            return False
        if case_state.get("pulse_prompt_count", 0) >= 1:
            return False
        if completion_score < PLANNER_CONFIG["pulse_completion_min"]:
            return False
        return len(missing_slots) <= PLANNER_CONFIG["pulse_max_missing_slots"]

    # 根据当前的case_state和计划中的下一步行动
    # 构建一个合理的行动理由，帮助用户理解为什么要这么做
    def _build_action_reason(self, case_state, next_action, missing_slots, contradictions, should_request_pulse):
        if next_action == "risk_escalation":
            return "已经识别到高风险信号，需要先完成安全分诊。"
        if next_action == "clarify_conflict":
            return f"已发现信息冲突，需要优先澄清：{'；'.join(contradictions)}。"
        if next_action == "summarize_progress":
            return "已经进行了多轮追问，先总结进展并集中补关键缺口。"
        if next_action == "ask_followup_bundle":
            return f"当前仍缺少{'、'.join(TCM_SLOT_LABELS.get(slot, slot) for slot in missing_slots[:3])}。"
        if next_action == "ask_followup_single" and missing_slots:
            return f"下一步优先补充{TCM_SLOT_LABELS.get(missing_slots[0], missing_slots[0])}。"
        if next_action == "request_pulse_input" and should_request_pulse:
            return "问诊信息已基本成型，可以尝试补充脉诊这一层证据。"
        return "关键信息已基本齐全，可以整理当前判断与建议。"

    def _stop_condition(self, next_action, risk_level, missing_slots, contradictions):
        if risk_level == "HIGH":
            return "触发高风险即停止常规追问，优先转入安全建议。"
        if contradictions:
            return "澄清冲突信息后再进入下一阶段。"
        if next_action == "request_pulse_input":
            return "等待脉诊输入或用户明确暂无脉诊信息。"
        if missing_slots:
            return "关键槽位补充到基本充分后再结束收集。"
        return "输出整理建议后结束当前轮次。"

    def _goal_progress(self, next_action, risk_level, missing_slots, should_request_pulse):
        if risk_level == "HIGH":
            return "已识别高风险，转入安全分诊"
        if next_action == "clarify_conflict":
            return "正在澄清前后不一致的信息"
        if next_action in {"ask_followup_single", "ask_followup_bundle", "summarize_progress"}:
            return f"正在补充{min(len(missing_slots), 3)}项关键四诊信息"
        if next_action == "request_pulse_input" and should_request_pulse:
            return "问诊信息已基本齐全，进入脉诊补充阶段"
        return "信息基本齐全，可整理输出建议"

    def _build_followup_reason(self, case_state, missing_slots, syndrome_candidates):
        if not missing_slots:
            return "四诊核心信息已经基本齐全，可以输出辨证线索与分诊建议。"

        summary = case_state.get("tcm_summary") or case_state.get("summary") or "当前证据有限"
        target_text = "、".join(TCM_SLOT_LABELS.get(slot, slot) for slot in missing_slots[:3])
        if syndrome_candidates:
            syndrome_text = "、".join(item["name"] for item in syndrome_candidates[:2])
            return f"当前中医问诊暂偏向{syndrome_text}等方向，但仍缺少{target_text}。现有证据：{summary}。"
        return f"当前四诊证据仍不足，尤其缺少{target_text}。现有证据：{summary}。"

    def _completion_score(self, case_state):
        key_slots = [
            "chief_complaint",
            "duration",
            "severity",
            "location",
            "cold_heat",
            "appetite",
            "sleep",
            "stool_urine",
        ]
        filled = sum(1 for slot in key_slots if self._has_value(case_state.get(slot)))
        return round(filled / len(key_slots), 2)

    def _completion_label(self, score):
        if score >= 0.85:
            return "充分"
        if score >= 0.55:
            return "基本充分"
        return "不足"

    def _has_value(self, value):
        if isinstance(value, list):
            return bool(value)
        return bool(value)

    def _should_ask_slot(self, slot_name, followup_counts):
        ask_limit = SLOT_ASK_LIMITS.get(slot_name, 1)
        return followup_counts.get(slot_name, 0) < ask_limit
