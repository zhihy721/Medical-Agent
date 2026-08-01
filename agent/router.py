# 动作驱动的对话管理
# 把动作结果变成一个强结构对象
from dataclasses import dataclass
# slot->中文语义标签的映射表
from knowledge.tcm_knowledge import TCM_SLOT_LABELS


@dataclass
class ActionResult:
    name: str
    response: str
    status: str
    is_final: bool
    missing_slots: list
    followup_slot: str = ""
    render_mode: str = "followup"

# Router负责根据Planner的决策和当前的case_state
# 生成一个具体的对用户的回复文本，以及一些额外的指令信息
# 它是连接Planner和实际对话输出的桥梁
class Router:
    FOLLOWUP_VARIANTS = {
        "chief_complaint": [
            "请先用一句话描述你这次最主要的不适。",
            "我还没识别到明确的主诉。你可以直接说，比如“我咳嗽得厉害”或“我肚子痛”。",
        ],
        "duration": [
            "这种不适大概持续多久了？",
            "我还需要了解病程长短，比如是刚开始，还是已经有几天了？",
        ],
        "severity": [
            "目前不适大概有多重？可以说轻、中、重，或者说是否已经影响日常活动。",
            "我还想确认严重程度，它现在只是隐约不舒服，还是已经明显难受了？",
        ],
        "location": [
            "不适主要在什么部位？请尽量说具体一点。",
            "我还需要确认部位，比如头部、胸口、腹部、咽喉，还是其他位置？",
        ],
        "cold_heat": [
            "这次不适偏怕冷、发热，还是有时冷有时热？",
            "我想确认一下寒热倾向，你是更怕冷，还是觉得发热明显？",
        ],
        "sweating": [
            "平时或发作时出汗情况怎么样？是无汗、容易出汗，还是夜里出汗？",
            "我还想了解汗出情况，这对判断表里寒热有帮助。",
        ],
        "thirst": [
            "最近口渴吗？更想喝热水还是冷饮？",
            "我还需要了解口渴和饮水偏好，比如是否明显口渴、喜欢冷饮还是热饮。",
        ],
        "appetite": [
            "这几天胃口怎么样？有没有纳差、食欲下降或吃了容易不舒服？",
            "我想再确认一下食欲和进食情况，这有助于判断脾胃状态。",
        ],
        "sleep": [
            "最近睡眠怎么样？有没有失眠、多梦或者总想睡？",
            "我还想了解睡眠情况，比如入睡难、梦多，还是容易困倦。",
        ],
        "stool_urine": [
            "最近大便和小便情况怎么样？有没有便溏、便秘、尿黄或尿清长？",
            "我还需要了解二便情况，这在中医问诊里很关键。",
        ],
        "pain_character": [
            "如果有疼痛，它更像刺痛、胀痛、隐痛，还是按着会更舒服或更痛？",
            "我想确认一下疼痛性质，比如胀、刺、隐隐作痛，或者喜按、拒按。",
        ],
        "emotion": [
            "最近情绪上有没有烦躁、焦虑、紧张、易怒或者明显郁闷？",
            "我还想了解一下情志变化，因为它可能影响辨证方向。",
        ],
        "complexion": [
            "最近面色和精神状态怎么样？是面白、发黄、发红，还是明显没精神？",
            "我还想了解面色神情，比如是否面色苍白、萎黄或精神差。",
        ],
        "voice_breath": [
            "说话声音和气息最近有变化吗？比如声音低弱、气短、说话没劲，或口气比较重？",
            "我还想确认一下声音和气息表现，这属于中医“闻诊”的一部分。",
        ],
        "female_cycle": [
            "如果方便的话，也可以补充一下月经或带下情况，比如周期变化、量和颜色。",
            "这一步主要是补充月经带下信息，女性问诊时常会用到。",
        ],
    }
    
    def route(self, case_state, plan, risk_result, guideline_result):
        missing_slots = plan.get("missing_slots", [])
        next_action = plan.get("next_action", "")

        if next_action == "risk_escalation" or risk_result.get("risk") == "HIGH":
            response = self._build_high_risk_response(case_state, risk_result, guideline_result, plan)
            return ActionResult("risk_escalation", response, "EMERGENCY_ESCALATION", True, [], render_mode="final")

        if next_action == "clarify_conflict":
            conflict_field = (plan.get("contradiction_fields") or [""])[0]
            response = self._build_conflict_question(case_state, conflict_field, plan)
            return ActionResult("clarify_conflict", response, "CLARIFYING_INFO", False, missing_slots, conflict_field)

        if next_action == "summarize_progress":
            response = self._build_progress_response(case_state, plan)
            return ActionResult("summarize_progress", response, "COLLECTING_INFO", False, missing_slots, missing_slots[0] if missing_slots else "")

        if next_action == "ask_followup_bundle":
            question = self._build_bundled_followup(case_state, missing_slots, plan)
            return ActionResult("ask_followup_bundle", question, "COLLECTING_INFO", False, missing_slots, missing_slots[0] if missing_slots else "")

        if next_action == "ask_followup_single" and missing_slots:
            slot_name = missing_slots[0]
            question = self._build_followup_question(case_state, slot_name, plan)
            return ActionResult("ask_followup_single", question, "COLLECTING_INFO", False, missing_slots, slot_name)

        if next_action == "request_pulse_input":
            response = self._build_pulse_request(case_state, plan)
            return ActionResult("request_pulse_input", response, "WAITING_PULSE_INPUT", False, missing_slots, render_mode="followup")

        response = self._build_final_response(case_state, risk_result, guideline_result, plan)
        return ActionResult("final_advice", response, "GENERATING_ADVICE", True, [], render_mode="final")

    def _build_followup_question(self, case_state, slot_name, plan):
        variants = self.FOLLOWUP_VARIANTS.get(slot_name, ["请再补充一些与这次不适相关的细节。"])
        counts = case_state.get("followup_counts", {})
        attempt = counts.get(slot_name, 0)
        question = variants[min(attempt, len(variants) - 1)]

        if slot_name == "location" and case_state.get("chief_complaint"):
            question = f"{case_state['chief_complaint']}主要出现在哪个部位？请尽量具体描述。"

        if attempt >= 2:
            question += " 如果方便，也可以把相关的寒热、食欲、睡眠或二便情况一起补充。"
        return question

    def _build_bundled_followup(self, case_state, missing_slots, plan):
        focus_slots = missing_slots[:3]
        labels = [TCM_SLOT_LABELS.get(slot, slot) for slot in focus_slots]
        chief_complaint = case_state.get("chief_complaint") or "这次不适"
        focus = "、".join(plan.get("four_diagnosis_focus", {}).keys()) or "问诊"
        return (
            f"为了把{chief_complaint}问得更完整，我还需要你补充"
            f"{'、'.join(labels)}。当前主要在完善{focus}这几部分信息。"
        )

    def _build_conflict_question(self, case_state, conflict_field, plan):
        labels = {
            "age": "年龄",
            "sex": "性别",
            "severity": "严重程度",
        }
        label = labels.get(conflict_field, "前后不一致的信息")
        contradictions = plan.get("contradictions", [])
        detail = contradictions[0] if contradictions else "我发现你前面的描述和后面的描述不完全一致。"
        return f"{detail}。为了避免判断偏掉，请你再确认一下这次的{label}。"

    def _build_progress_response(self, case_state, plan):
        summary = case_state.get("summary") or "目前已收集到部分信息"
        questions = "、".join(plan.get("pending_questions", [])[:3]) or "更多细节"
        return f"我先帮你整理一下：{summary}。接下来还想重点确认{questions}。"

    def _build_pulse_request(self, case_state, plan):
        chief_complaint = case_state.get("chief_complaint") or "这次不适"
        return (
            f"关于{chief_complaint}，目前问诊信息已经基本成型。"
            "如果你这边有脉诊设备结果，也可以继续补充脉象结论或信号质量；如果没有，我们也可以继续只根据问诊信息判断。"
        )

    def _build_high_risk_response(self, case_state, risk_result, guideline_result, plan):
        symptoms = "、".join(case_state.get("symptoms", [])) or "当前症状"
        lines = [
            f"根据你提供的信息，{symptoms}目前属于需要优先警惕的情况。",
            f"原因：{risk_result['reason']}。",
            f"建议：{risk_result['disposition']}。",
            "这一步先以安全分诊为主，不建议继续只依赖线上辨证或脉诊结果。",
        ]
        if plan.get("syndrome_candidates"):
            names = "、".join(item["name"] for item in plan["syndrome_candidates"])
            lines.append(f"中医问诊线索可暂参考：{names}，但应以线下评估安全性为先。")
        for advice in guideline_result.get("advice", []):
            lines.append(f"- {advice}")
        return "\n".join(lines)

    def _build_final_response(self, case_state, risk_result, guideline_result, plan):
        symptoms = "、".join(case_state.get("symptoms", [])) or "未明确"
        syndrome_candidates = plan.get("syndrome_candidates", [])
        syndrome_text = "、".join(item["name"] for item in syndrome_candidates) if syndrome_candidates else "证候线索暂不集中"
        parts = [
            f"已完成当前轮问诊整理，主要不适包括：{symptoms}。",
            f"当前分诊风险等级：{risk_result['risk']}。",
            f"四诊证据摘要：{case_state.get('tcm_summary') or '仍需继续补充'}。",
            f"目前中医问诊上可优先参考的证候方向：{syndrome_text}。",
            f"当前任务完成度：{plan.get('completion_label', '待评估')}，决策把握度约为{int(plan.get('confidence', 0) * 100)}%。",
            f"建议方向：{guideline_result['summary']}",
        ]
        for advice in guideline_result.get("advice", []):
            parts.append(f"- {advice}")
        return "\n".join(parts)
