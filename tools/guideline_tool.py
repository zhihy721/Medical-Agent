from knowledge.tcm_knowledge import TCM_SLOT_LABELS
from tools.protocol import managed_tool

TOOL_VERSION = "1.0"

RISK_SUMMARIES = {
    "HIGH": "当前信息提示风险较高，应优先考虑线下医疗评估。",
    "MEDIUM": "当前信息仍有一定风险，建议尽快补齐关键四诊细节并提高警惕。",
    "LOW": "当前未见明显高风险信号，可在持续观察的同时完善中医四诊信息。",
    "UNKNOWN": "目前信息不足，暂时无法做出可靠的分诊判断。",
}


@managed_tool("guideline", TOOL_VERSION, "根据风险与问诊进展生成分诊指引与建议")
def get_guideline_tool(case_state, risk_result, plan=None):
    """协议版入口：返回标准 ToolResult，异常由 managed_tool 捕获。"""
    plan = plan or {}
    missing_slots = plan.get("missing_slots", case_state.get("missing_slots", []))
    syndrome_candidates = plan.get("syndrome_candidates", case_state.get("syndrome_candidates", []))
    risk = risk_result.get("risk", "UNKNOWN")

    if risk == "HIGH":
        return {
            "summary": risk_result["disposition"],
            "advice": [
                "优先线下就医，不建议继续仅在线辨证观察。",
                "如有脉诊手结果，可作为补充资料带给线下医生参考，但不要替代就医。",
                *risk_result.get("observation_points", []),
            ],
        }

    advice = []

    if syndrome_candidates:
        for candidate in syndrome_candidates[:2]:
            evidence = "、".join(candidate.get("matched_evidence", [])) or "现有症状组合"
            advice.append(f"当前问诊线索可暂参考{candidate['name']}，依据包括：{evidence}。")

    if case_state.get("pulse_summary"):
        pulse_quality = case_state.get("pulse_signal_quality")
        if pulse_quality:
            advice.append(f"已接入脉诊结果：{case_state['pulse_summary']}，当前信号质量为{pulse_quality}。")
        else:
            advice.append(f"已接入脉诊结果：{case_state['pulse_summary']}。")
    else:
        advice.append("后续如果接入脉诊手结果，可以再把脉诊结论和信号质量并入判断。")

    if missing_slots:
        slot_labels = [TCM_SLOT_LABELS.get(slot, slot) for slot in missing_slots[:3]]
        advice.append(f"目前还缺少{'、'.join(slot_labels)}，补充后更适合继续按“望闻问切”推进。")

    advice.extend(point for point in risk_result.get("observation_points", []) if point not in advice)
    advice.append("以上内容仅用于分诊与中医问诊整理，不能替代医生面诊。")

    return {
        "summary": RISK_SUMMARIES.get(risk, RISK_SUMMARIES["UNKNOWN"]),
        "advice": advice,
    }


def get_guideline(case_state, risk_result, plan=None):
    """兼容层：保持旧接口行为，返回 summary/advice dict。"""
    result = get_guideline_tool(case_state, risk_result, plan)
    if result["status"] != "ok":
        return {
            "summary": "指南生成异常，建议谨慎参考并补充信息。",
            "advice": ["以上内容仅用于分诊与中医问诊整理，不能替代医生面诊。"],
        }
    return result["data"]
