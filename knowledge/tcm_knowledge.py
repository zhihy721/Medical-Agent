TCM_SLOT_LABELS = {
    "chief_complaint": "主要不适",
    "duration": "病程长短",
    "severity": "严重程度",
    "location": "不适部位",
    "cold_heat": "寒热倾向",
    "sweating": "汗出情况",
    "thirst": "口渴与饮水偏好",
    "appetite": "食欲与纳食",
    "sleep": "睡眠情况",
    "stool_urine": "二便情况",
    "pain_character": "疼痛性质",
    "emotion": "情志变化",
    "complexion": "面色神情",
    "voice_breath": "声音气息",
    "odor": "口气或气味",
    "female_cycle": "月经带下",
    "past_history": "既往病史",
    "age": "年龄",
    "pulse_summary": "脉诊结论",
    "pulse_strength": "脉力",
    "pulse_rate": "脉率",
    "pulse_signal_quality": "脉诊信号质量",
}

FOUR_DIAGNOSIS_GROUPS = {
    "望": ["complexion"],
    "闻": ["voice_breath", "odor"],
    "问": [
        "chief_complaint",
        "duration",
        "severity",
        "location",
        "cold_heat",
        "sweating",
        "thirst",
        "appetite",
        "sleep",
        "stool_urine",
        "pain_character",
        "emotion",
        "female_cycle",
        "past_history",
        "age",
    ],
    "切": ["pulse_summary", "pulse_strength", "pulse_rate", "pulse_signal_quality"],
}

CHIEF_COMPLAINT_PRIORITIES = {
    "咳嗽": ["cold_heat", "sweating", "thirst"],
    "发热": ["cold_heat", "sweating", "thirst"],
    "胸痛": ["location", "pain_character", "cold_heat", "emotion"],
    "腹痛": ["pain_character", "appetite", "stool_urine", "cold_heat"],
    "头痛": ["cold_heat", "sleep", "emotion"],
    "头晕": ["thirst", "sleep", "appetite"],
    "乏力": ["appetite", "sleep", "stool_urine"],
    "恶心": ["appetite", "thirst", "stool_urine"],
    "腹泻": ["cold_heat", "appetite", "stool_urine"],
}

GENERAL_TCM_PRIORITIES = [
    "chief_complaint",
    "duration",
    "severity",
    "location",
    "cold_heat",
    "appetite",
    "sleep",
    "stool_urine",
]

SYNDROME_RULES = [
    {
        "name": "风寒束表",
        "evidence": {"cold_heat": ["恶寒"], "sweating": ["无汗"], "pulse_summary": ["浮紧", "浮"]},
        "symptoms": {"symptoms": ["咳嗽", "头痛"], "accompanying_symptoms": ["寒战"]},
    },
    {
        "name": "风热犯肺",
        "evidence": {"cold_heat": ["发热"], "thirst": ["口渴"]},
        "symptoms": {"symptoms": ["咳嗽", "咽痛"]},
    },
    {
        "name": "痰热壅肺",
        "evidence": {"thirst": ["口渴"], "pulse_summary": ["滑数", "滑"]},
        "symptoms": {"symptoms": ["咳嗽", "呼吸困难"], "accompanying_symptoms": ["胸闷"]},
    },
    {
        "name": "脾胃虚寒",
        "evidence": {"cold_heat": ["怕冷", "喜温"], "appetite": ["纳差"], "stool_urine": ["便溏"]},
        "symptoms": {"symptoms": ["腹痛", "腹泻"]},
    },
    {
        "name": "肝郁气滞",
        "evidence": {"emotion": ["焦虑", "易怒", "抑郁"], "pain_character": ["胀痛"], "pulse_summary": ["弦"]},
        "symptoms": {"symptoms": ["胸痛", "腹痛"]},
    },
    {
        "name": "气血两虚",
        "evidence": {"sleep": ["失眠"], "appetite": ["纳差"], "pulse_summary": ["细弱", "细"]},
        "symptoms": {"symptoms": ["乏力", "头晕"]},
    },
]


def infer_tcm_syndromes(case_state):
    ranked = []
    symptoms = set(case_state.get("symptoms", []))
    accompanying = set(case_state.get("accompanying_symptoms", []))

    for rule in SYNDROME_RULES:
        score = 0
        matched = []

        for symptom in rule.get("symptoms", {}).get("symptoms", []):
            if symptom in symptoms:
                score += 2
                matched.append(symptom)

        for symptom in rule.get("symptoms", {}).get("accompanying_symptoms", []):
            if symptom in accompanying:
                score += 1
                matched.append(symptom)

        for field, options in rule.get("evidence", {}).items():
            value = str(case_state.get(field, ""))
            if any(option in value for option in options):
                score += 2
                matched.append(f"{field}:{value}")

        if score >= 3:
            ranked.append(
                {
                    "name": rule["name"],
                    "score": score,
                    "matched_evidence": matched[:4],
                }
            )

    ranked.sort(key=lambda item: (-item["score"], item["name"]))
    return ranked[:3]
