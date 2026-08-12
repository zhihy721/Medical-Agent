import json

from knowledge.tcm_knowledge import KNOWLEDGE_VERSION, TCM_SLOT_LABELS, normalize_term
from llm.prompt import EXTRACTION_PROMPT, FINAL_RESPONSE_PROMPT, FOLLOWUP_PROMPT
from tools.knowledge_tool import search_knowledge
from tools.symptom_tool import extract_symptoms

# 降低LLM负担
# 保证医学规范，所有问题都是预定义的，不会乱问
# 提高用户体验，多轮不同问法
# 支持可控优化，之后可以根据实际情况增加更多变体，或者针对不同用户画像调整问法
# 优化转化率
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

# 信息抽取
# 先用规则抽取，再用LLM抽取，最后合并结果
# 保证规范的同时降低LLM负担
def extract_case_slots(llm, user_input):
    rule_result = extract_symptoms(user_input)
    llm_result = llm.extract_json(EXTRACTION_PROMPT.format(user_input=user_input))
    llm_result = _normalize_llm_result(llm_result)
    return merge_extracted_slots(rule_result, llm_result)

# LLM 抽取结果先过术语归一化，把口语别名统一为规范词（如“怕冷”→“恶寒”）
# 保证与规则层、辨证规则用同一套词表
def _normalize_llm_result(llm_result):
    if not isinstance(llm_result, dict):
        return llm_result
    normalized = {}
    for key, value in llm_result.items():
        if isinstance(value, str):
            normalized[key] = normalize_term(value)
        elif isinstance(value, list):
            normalized[key] = [normalize_term(item) if isinstance(item, str) else item for item in value]
        else:
            normalized[key] = value
    return normalized

# 规则和LLM怎么合并
# schema约束，防止LLM乱抽取
# 信息累积，不覆盖之前抽取到的信息，不丢信息
# 轻、重同时出现，直接变为重
def merge_extracted_slots(rule_result, llm_result):
    if not isinstance(llm_result, dict):
        return rule_result

    merged = dict(rule_result)
    for field, value in llm_result.items():
        if field not in merged:
            continue
        if isinstance(merged[field], list):
            combined = merged[field] + (value if isinstance(value, list) else [])
            merged[field] = list(dict.fromkeys(item for item in combined if item))
        elif value:
            merged[field] = value
    return merged

# 判断用户是否跳过了脉诊输入
def handle_post_pulse_reply(memory, user_input):
    case_state = memory.get_case_state()
    if case_state.get("last_action") != "request_pulse_input":
        return

    text = (user_input or "").strip()
    if not text:
        return

    memory.mark_pulse_skipped()

# 澄清回复的后处理：上一轮在澄清冲突字段，且本轮抽到了该字段的新值
# 则以最新值为准消除矛盾，避免对话永久卡在澄清环节
def resolve_clarified_contradiction(memory, extracted_slots):
    case_state = memory.get_case_state()
    if case_state.get("last_action") != "clarify_conflict":
        return
    field = case_state.get("last_followup_slot", "")
    if field and extracted_slots.get(field):
        memory.resolve_contradiction(field)

# 把计划和评审结果同步到记忆里，形成稳定的状态，供LLM调用
def sync_plan_to_memory(memory, plan, internal_step=None):
    # 写入分诊线索，供后续计划和行动调用
    memory.update_triage(
        syndrome_candidates=[item["name"] for item in plan.get("syndrome_candidates", [])]
    )
    # 写入计划结果，供后续评审和行动调用
    memory.update_task_state(
        goal_progress=plan.get("goal_progress", ""),
        pending_questions=plan.get("pending_questions", []),
        hypotheses=plan.get("hypotheses", []),
        confidence=plan.get("confidence"),
        contradictions=plan.get("contradictions", []),
        contradiction_fields=plan.get("contradiction_fields", []),
        stop_reason=plan.get("stop_condition", ""),
        internal_steps=internal_step,
    )

# 把评审结果同步到记忆里，形成稳定的状态，供LLM调用
# 是否需要重新规划、是否合理、是否需要结束、评审理由等
def sync_review_to_memory(memory, review_result, fallback_stop_reason=""):
    memory.update_task_state(
        self_check=review_result,
        stop_reason=review_result.get("reason", fallback_stop_reason),
    )

# 把行动结果同步到记忆里，形成稳定的状态，供LLM调用
def sync_action_to_memory(memory, action_result, plan, step):
    memory.update_triage(
        status=action_result.status,
        last_action=action_result.name,
        missing_slots=action_result.missing_slots,
        followup_slot=action_result.followup_slot,
    )
    # 记录行动结果，回溯分析行动效果，形成行动-结果的关联
    # 做行为建模，分析哪些行动更有效，哪些行动需要优化
    memory.record_action(
        action_name=action_result.name,
        status=action_result.status,
        step=step,
        reason=plan.get("action_reason", ""),
        missing_slots=action_result.missing_slots,
    )

# 输入：行动结果对象，包含name、response、status、is_final、missing_slots、followup_slot等字段
# 输出：可序列化的字典，方便存储和传输
# 把一个类对象转成标准JSON结构
# 把运行时对象变成协议数据
def serialize_action_result(action_result):
    if action_result is None:
        return {}
    return {
        "name": action_result.name,
        "response": action_result.response,
        "status": action_result.status,
        "is_final": action_result.is_final,
        "missing_slots": list(action_result.missing_slots),
        "followup_slot": action_result.followup_slot,
        "render_mode": action_result.render_mode,
    }

# 输入：行动结果的字典表示，包含name、response、status、is_final、missing_slots、followup_slot等字段
# 输出：行动结果对象，包含name、response、status、is_final、missing_slots、followup_slot等属性
# 把标准JSON结构转成一个类对象
# 把协议数据变成运行时对象
def deserialize_action_result(action_result_payload):
    if not action_result_payload:
        return None

    from agent.router import ActionResult

    return ActionResult(
        name=action_result_payload.get("name", ""),
        response=action_result_payload.get("response", ""),
        status=action_result_payload.get("status", ""),
        is_final=action_result_payload.get("is_final", False),
        missing_slots=list(action_result_payload.get("missing_slots", [])),
        followup_slot=action_result_payload.get("followup_slot", ""),
        render_mode=action_result_payload.get("render_mode", "followup"),
    )

# 高风险分诊升级的结果构建
# 安全优先级>中医辨证
def build_risk_escalation_action_result(case_state, risk_result, guideline_result, plan):
    from agent.router import ActionResult

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
    # 外部知识库增强，提供一些常见的高风险症状组合和处理建议，帮助用户理解风险的具体表现和应对措施
    for advice in guideline_result.get("advice", []):
        lines.append(f"- {advice}")
    # MCP 可选增强：附加附近医院清单；无位置来源、工具不可用或调用失败均静默跳过
    lines.extend(_nearby_hospital_lines(case_state))

    return ActionResult(
        name="risk_escalation",
        response="\n".join(lines),
        status="EMERGENCY_ESCALATION",
        is_final=True,
        missing_slots=[],
        render_mode="final",
    )

# 高风险升级回复的附近医院建议：仅当 hospital_locator MCP 工具已注册且有位置来源时附加
# 位置优先级：会话中抽取的城市（city 槽位） > MEDICAL_AGENT_LOCATION 环境变量（手动覆盖）
def _nearby_hospital_lines(case_state):
    import os

    from tools.registry import default_registry

    location = (case_state.get("city") or "").strip() or os.getenv("MEDICAL_AGENT_LOCATION", "").strip()
    if not location:
        return []
    tool = default_registry.get("hospital_locator_search_nearby_hospitals")
    if tool is None:
        return []
    result = tool(location=location)
    if result.get("status") != "ok":
        return []
    hospitals = (result.get("data") or {}).get("hospitals") or []
    if not hospitals:
        return []
    lines = [f"如你就近位于{location}，以下医院可前往（演示数据，以实际导航为准）："]
    for item in hospitals[:3]:
        emergency = "，有急诊" if item.get("has_emergency") else ""
        lines.append(f"- {item.get('name', '')}（{item.get('district', '')}，约 {item.get('distance_km', '?')} 公里{emergency}）")
    return lines


# 正常结束路径的结果构建
def build_final_advice_action_result(case_state, risk_result, guideline_result, plan):
    from agent.router import ActionResult

    symptoms = "、".join(case_state.get("symptoms", [])) or "未明确"
    syndrome_candidates = plan.get("syndrome_candidates", [])
    syndrome_text = (
        "、".join(item["name"] for item in syndrome_candidates)
        if syndrome_candidates
        else "证候线索暂不集中"
    )
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
    # RAG 注入：以 top 证型候选 + 主诉检索方剂/调护/FAQ 语料，附在建议末尾
    parts.extend(_knowledge_reference_lines(case_state, plan))

    return ActionResult(
        name="final_advice",
        response="\n".join(parts),
        status="GENERATING_ADVICE",
        is_final=True,
        missing_slots=[],
        render_mode="final",
    )

# 共享知识检索：query 构建（top 证型候选 + 主诉）+ 过滤 corpus 命中取前 3；
# 失败或无命中时静默返回空，绝不阻断主链路；回复注入与 LLM prompt 上下文共用
def _retrieve_corpus_hits(case_state, plan):
    syndrome_candidates = plan.get("syndrome_candidates") or []
    chief_complaint = case_state.get("chief_complaint", "")
    query_parts = [item["name"] for item in syndrome_candidates[:1]]
    if chief_complaint:
        query_parts.append(chief_complaint)
    query = " ".join(part for part in query_parts if part).strip()
    if not query:
        return []
    try:
        retrieval = search_knowledge(query, top_k=6)
    except Exception:
        return []
    return [
        hit for hit in retrieval.get("hits", [])
        if hit.get("type") == "corpus" and hit.get("content")
    ][:3]


# 回复注入：把方剂/调护/FAQ 语料参考行附在最终建议末尾
def _knowledge_reference_lines(case_state, plan):
    corpus_hits = _retrieve_corpus_hits(case_state, plan)
    if not corpus_hits:
        return []

    lines = ["知识库参考（演示语料，具体用药需医师辨证）："]
    for hit in corpus_hits:
        prefix = "参考方剂" if (hit.get("source") or "").endswith("formulas.json") else "调护参考"
        lines.append(f"- [{prefix}] {hit['name']}：{hit['content']}")
    version_text = "、".join(f"{key}={value}" for key, value in KNOWLEDGE_VERSION.items())
    lines.append(f"知识库版本：{version_text}")
    return lines


# LLM prompt 知识上下文：每条命中一行，无命中传"无"（prompt 要求忽略）
def _knowledge_context_text(case_state, plan):
    corpus_hits = _retrieve_corpus_hits(case_state, plan)
    if not corpus_hits:
        return "无"
    return "\n".join(f"【{hit['name']}】{hit['content']}" for hit in corpus_hits)


# 信息冲突澄清的结果构建
def build_clarify_conflict_action_result(case_state, plan):
    from agent.router import ActionResult

    labels = {
        "age": "年龄",
        "sex": "性别",
        "severity": "严重程度",
    }
    conflict_field = (plan.get("contradiction_fields") or [""])[0]
    label = labels.get(conflict_field, "前后不一致的信息")
    contradictions = plan.get("contradictions", [])
    detail = contradictions[0] if contradictions else "我发现你前面的描述和后面的描述不完全一致。"
    response = f"{detail}。为了避免判断偏掉，请你再确认一下这次的{label}。"

    return ActionResult(
        name="clarify_conflict",
        response=response,
        status="CLARIFYING_INFO",
        is_final=False,
        missing_slots=list(plan.get("missing_slots", [])),
        followup_slot=conflict_field,
        render_mode="followup",
    )

# 请求补充脉诊输入的结果构建
def build_request_pulse_input_action_result(case_state, plan):
    from agent.router import ActionResult

    chief_complaint = case_state.get("chief_complaint") or "这次不适"
    response = (
        f"关于{chief_complaint}，目前问诊信息已经基本成型。"
        "如果你这边有脉诊设备结果，也可以继续补充脉象结论或信号质量；如果没有，我们也可以继续只根据问诊信息判断。"
    )

    return ActionResult(
        name="request_pulse_input",
        response=response,
        status="WAITING_PULSE_INPUT",
        is_final=False,
        missing_slots=list(plan.get("missing_slots", [])),
        render_mode="followup",
    )

# 单变量补充的结果构建
def build_followup_single_action_result(case_state, plan):
    from agent.router import ActionResult

    missing_slots = list(plan.get("missing_slots", []))
    slot_name = missing_slots[0] if missing_slots else ""
    variants = FOLLOWUP_VARIANTS.get(slot_name, ["请再补充一些与这次不适相关的细节。"])
    counts = case_state.get("followup_counts", {})
    attempt = counts.get(slot_name, 0)
    question = variants[min(attempt, len(variants) - 1)]

    if slot_name == "location" and case_state.get("chief_complaint"):
        question = f"{case_state['chief_complaint']}主要出现在哪个部位？请尽量具体描述。"

    if attempt >= 2:
        question += " 如果方便，也可以把相关的寒热、食欲、睡眠或二便情况一起补充。"

    return ActionResult(
        name="ask_followup_single",
        response=question,
        status="COLLECTING_INFO",
        is_final=False,
        missing_slots=missing_slots,
        followup_slot=slot_name,
        render_mode="followup",
    )

# 多变量补充的结果构建
def build_followup_bundle_action_result(case_state, plan):
    from agent.router import ActionResult

    missing_slots = list(plan.get("missing_slots", []))
    focus_slots = missing_slots[:3]
    labels = [TCM_SLOT_LABELS.get(slot, slot) for slot in focus_slots]
    chief_complaint = case_state.get("chief_complaint") or "这次不适"
    focus = "、".join(plan.get("four_diagnosis_focus", {}).keys()) or "问诊"
    response = (
        f"为了把{chief_complaint}问得更完整，我还需要你补充"
        f"{'、'.join(labels)}。当前主要在完善{focus}这几部分信息。"
    )

    return ActionResult(
        name="ask_followup_bundle",
        response=response,
        status="COLLECTING_INFO",
        is_final=False,
        missing_slots=missing_slots,
        followup_slot=missing_slots[0] if missing_slots else "",
        render_mode="followup",
    )

# 根据行动结果的render_mode
# 如果是followup，生成一个针对性的追问，帮助用户补充信息，形成更完整的问诊上下文
# 如果是final，生成一个总结性的回复，帮助用户理解当前的问诊结果
def render_response(llm, memory, action_result, case_state, risk_result, guideline_result, plan):
    conversation_context = memory.get_prompt_context_text()
    profile_context = memory.get_profile_context_text()
    if llm.last_provider_used in {"deepseek", "openai"}:
        # 脉诊请求直出确定性文案：LLM 重写实测会把脉诊请求改写成雷同追问，语义丢失
        if action_result.name == "request_pulse_input":
            return action_result.response
        if action_result.render_mode == "followup":
            prompt = FOLLOWUP_PROMPT.format(
                case_state=json.dumps(case_state, ensure_ascii=False),
                plan=json.dumps(plan, ensure_ascii=False),
                planned_action=action_result.name,
                action_draft=action_result.response,
                profile_context=profile_context,
                conversation_context=conversation_context,
            )
            response = llm.call(prompt).strip()
            if response and response != "{}":
                return response

        if action_result.render_mode == "final":
            prompt = FINAL_RESPONSE_PROMPT.format(
                case_state=json.dumps(case_state, ensure_ascii=False),
                risk_result=json.dumps(risk_result, ensure_ascii=False),
                guideline_result=json.dumps(guideline_result, ensure_ascii=False),
                plan=json.dumps(plan, ensure_ascii=False),
                planned_action=action_result.name,
                action_draft=action_result.response,
                knowledge_context=_knowledge_context_text(case_state, plan),
                profile_context=profile_context,
                conversation_context=conversation_context,
            )
            response = llm.call(prompt).strip()
            if response and response != "{}":
                return response

    return action_result.response
