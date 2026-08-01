def risk_assessment(case_state):
    red_flags = case_state.get("red_flags", [])
    symptoms = set(case_state.get("symptoms", []))
    accompanying = set(case_state.get("accompanying_symptoms", []))
    severity = case_state.get("severity", "")
    past_history = set(case_state.get("past_history", []))
    age = _safe_int(case_state.get("age"))

    if red_flags:
        return _result(
            "HIGH",
            f"检测到红旗信号: {', '.join(red_flags)}",
            "建议尽快线下就医或急诊评估",
            ["red_flag_detected"],
            ["出现症状加重、意识改变或呼吸困难时立即就医"],
        )

    if "胸痛" in symptoms and "呼吸困难" in symptoms:
        return _result(
            "HIGH",
            "胸痛伴呼吸困难属于高风险组合",
            "建议立即前往急诊",
            ["chest_pain_with_dyspnea"],
            ["避免继续等待线上回复", "尽快由线下医生评估心肺问题"],
        )

    if "胸痛" in symptoms and past_history.intersection({"冠心病", "高血压"}):
        return _result(
            "HIGH",
            "胸痛合并心血管既往病史，需要优先排除急性心血管问题",
            "建议当天线下就医，若持续不缓解应急诊处理",
            ["chest_pain_with_cardiac_history"],
            ["若胸痛持续、加重或伴出汗气短，请立即就医"],
        )

    if severity == "重度":
        return _result(
            "HIGH",
            "症状描述为重度，需要尽快线下评估",
            "建议当天就医",
            ["severe_symptom"],
            ["如果已经影响正常活动，不建议继续仅做居家观察"],
        )

    if "发热" in symptoms and "咳嗽" in symptoms and "呼吸困难" in symptoms:
        return _result(
            "HIGH",
            "发热、咳嗽并伴呼吸困难，需要警惕较重呼吸系统问题",
            "建议尽快线下评估呼吸道情况",
            ["respiratory_high_risk"],
            ["关注呼吸频率、胸闷和精神状态变化"],
        )

    if "腹痛" in symptoms and {"呕吐", "便血"}.intersection(symptoms.union(accompanying).union(red_flags)):
        return _result(
            "HIGH",
            "腹痛伴呕吐或消化道出血提示消化系统高风险",
            "建议尽快线下就医",
            ["abdominal_high_risk"],
            ["若疼痛持续加重或无法进食，应尽快就医"],
        )

    if "头痛" in symptoms and severity in {"中度", "重度"} and "呕吐" in symptoms.union(accompanying):
        return _result(
            "MEDIUM",
            "头痛伴呕吐，需要继续排查神经系统风险信号",
            "建议尽快补充起病方式、持续时间和是否突发加重",
            ["headache_with_vomiting"],
            ["若出现意识异常、肢体无力或突然加重，应立即就医"],
        )

    if "发热" in symptoms and "咳嗽" in symptoms:
        return _result(
            "MEDIUM",
            "发热伴呼吸道症状，需要结合病程和伴随症状继续判断",
            "建议完善体温、病程和呼吸道相关伴随症状信息",
            ["fever_with_cough"],
            ["若合并气促、胸痛或高热持续不退，应及时就医"],
        )

    if "腹痛" in symptoms and "胃病" in past_history:
        return _result(
            "MEDIUM",
            "腹痛合并既往胃病史，需要继续排查消化系统问题",
            "建议关注疼痛位置、进食关系和呕吐排便情况",
            ["abdominal_pain_with_history"],
            ["若疼痛固定在右下腹或持续加重，应线下评估"],
        )

    if "胸痛" in symptoms or "呼吸困难" in symptoms:
        return _result(
            "MEDIUM",
            "当前症状涉及胸部或呼吸系统，仍需谨慎分诊",
            "建议尽快补充持续时间、严重程度和伴随症状",
            ["cardiopulmonary_needs_followup"],
            ["若症状进行性加重，应尽快线下就医"],
        )

    if age is not None and age >= 65 and symptoms:
        return _result(
            "MEDIUM",
            "高龄患者出现症状时建议提高警惕并尽快明确病情变化",
            "建议尽快补充病程、严重程度和基础病信息",
            ["older_patient"],
            ["如有多种慢病或症状波动明显，建议线下评估"],
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
