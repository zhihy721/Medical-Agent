from knowledge.tcm_knowledge import get_risk_rules
from tools.protocol import managed_tool

# 工具版本：风险规则迭代时升级，便于历史会话追溯当时使用的规则版本
# 1.1：规则从代码硬编码迁移至 knowledge/data/risk_rules.json（声明式解释执行）
TOOL_VERSION = "1.1"


@managed_tool("risk_assessment", TOOL_VERSION, "基于规则的安全风险评估，输出 HIGH/MEDIUM/LOW/UNKNOWN")
def risk_assessment_tool(case_state):
    """协议版入口：返回标准 ToolResult，异常由 managed_tool 捕获。"""
    red_flags = case_state.get("red_flags", [])
    symptoms = set(case_state.get("symptoms", []))
    accompanying = set(case_state.get("accompanying_symptoms", []))
    severity = case_state.get("severity", "")
    past_history = set(case_state.get("past_history", []))
    age = _safe_int(case_state.get("age"))

    # 红旗信号短路：命中即 HIGH，优先于所有声明式规则
    if red_flags:
        return _result(
            "HIGH",
            f"检测到红旗信号: {', '.join(red_flags)}",
            "建议尽快线下就医或急诊评估",
            ["red_flag_detected"],
            ["出现症状加重、意识改变或呼吸困难时立即就医"],
        )

    # 按 knowledge/data/risk_rules.json 的顺序逐条解释，命中即返回
    for rule in get_risk_rules():
        if _rule_matches(rule, symptoms, accompanying, severity, past_history, age):
            return _result(
                rule["risk"],
                rule["reason"],
                rule["disposition"],
                [rule["id"]],
                list(rule.get("observation_points", [])),
            )

    if symptoms:
        return _result(
            "LOW",
            "暂未识别到明确红旗信号",
            "可继续补充信息并进行居家观察",
            ["no_high_risk_signal"],
            ["若新增呼吸困难、持续高热、剧烈疼痛等情况，应及时就医"],
        )

    return _result(
        "UNKNOWN",
        "有效症状信息不足",
        "需要先补充主要不适和病程信息",
        ["insufficient_information"],
        ["先描述最主要的不适、出现多久以及是否明显加重"],
    )


# 声明式条件解释器：when 中所有条件同时满足才算命中
# 支持的症状池：symptoms / symptoms∪accompanying_symptoms（union_any）
def _rule_matches(rule, symptoms, accompanying, severity, past_history, age):
    when = rule.get("when", {})

    if "symptoms_all" in when and not all(item in symptoms for item in when["symptoms_all"]):
        return False
    if "symptoms_any" in when and not any(item in symptoms for item in when["symptoms_any"]):
        return False
    if "union_any" in when:
        pool = symptoms.union(accompanying)
        if not any(item in pool for item in when["union_any"]):
            return False
    if "severity_in" in when and severity not in when["severity_in"]:
        return False
    if "past_history_any" in when and not past_history.intersection(when["past_history_any"]):
        return False
    if "age_gte" in when and (age is None or age < when["age_gte"]):
        return False
    if when.get("require_symptoms") and not symptoms:
        return False
    return True


def _result(risk, reason, disposition, matched_rules, observation_points):
    return {
        "risk": risk,
        "reason": reason,
        "disposition": disposition,
        "matched_rules": matched_rules,
        "observation_points": observation_points,
    }


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def risk_assessment(case_state):
    """兼容层：保持旧接口行为，返回风险结果 dict。"""
    result = risk_assessment_tool(case_state)
    if result["status"] != "ok":
        return fallback_risk_result(result["error"])
    return result["data"]


def fallback_risk_result(reason=""):
    """工具异常时的降级结果，保证问诊链路不中断。"""
    return _result(
        "UNKNOWN",
        reason or "风险评估执行异常",
        "需要先补充主要不适和病程信息",
        ["tool_error"],
        ["先描述最主要的不适、出现多久以及是否明显加重"],
    )
