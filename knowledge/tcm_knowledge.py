# 中医知识层：结构化定义 + knowledge/data 数据文件加载器
# 结构级定义（槽位标签、四诊分组、通用优先级）保留在代码中；
# 内容级知识（辨证规则、术语归一化、主诉追问、红旗信号）外置为 JSON，
# 模块加载时读取并做 schema 校验，校验失败直接抛出 KnowledgeLoadError。
# 注意：外置知识内容为演示级，需经中医专业审核后方可用于实际场景。
import json
import os

# 数据文件目录：knowledge/data/
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 红旗信号优先级枚举，校验时使用
_FLAG_PRIORITIES = {"HIGH", "MEDIUM"}


class KnowledgeLoadError(ValueError):
    """知识文件缺失或结构校验失败时抛出，带明确的定位信息。"""


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

# 所有主诉都会先按通用顺序补充基础信息，再叠加主诉专属优先级
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


# ---------- 数据加载与校验 ----------

def _load_json(filename):
    """读取 knowledge/data 下的 JSON 文件，要求带 version 字段。"""
    path = os.path.join(_DATA_DIR, filename)
    if not os.path.exists(path):
        raise KnowledgeLoadError(f"知识文件缺失: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeLoadError(f"知识文件读取失败 {filename}: {exc}") from exc
    if not isinstance(data, dict) or not data.get("version"):
        raise KnowledgeLoadError(f"知识文件缺少 version 字段: {filename}")
    return data


def _require_slot(slot, context):
    if slot not in TCM_SLOT_LABELS:
        raise KnowledgeLoadError(f"未知槽位 {slot!r}（出现在 {context}）")


def _load_syndrome_rules():
    data = _load_json("syndrome_rules.json")
    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise KnowledgeLoadError("syndrome_rules.json 缺少 rules 列表")

    for index, rule in enumerate(rules):
        context = f"syndrome_rules.json rules[{index}]"
        name = rule.get("name")
        if not name or not isinstance(name, str):
            raise KnowledgeLoadError(f"{context} 缺少 name")

        evidence = rule.get("evidence")
        if not isinstance(evidence, dict):
            raise KnowledgeLoadError(f"{context}（{name}）缺少 evidence 字典")
        for slot, options in evidence.items():
            _require_slot(slot, context)
            if not isinstance(options, list) or not options or not all(isinstance(o, str) for o in options):
                raise KnowledgeLoadError(f"{context}（{name}）evidence[{slot}] 必须是非空字符串列表")

        symptoms = rule.get("symptoms")
        if not isinstance(symptoms, dict):
            raise KnowledgeLoadError(f"{context}（{name}）缺少 symptoms 字典")
        for key in ("symptoms", "accompanying_symptoms"):
            value = symptoms.get(key, [])
            if not isinstance(value, list):
                raise KnowledgeLoadError(f"{context}（{name}）symptoms[{key}] 必须是列表")

        if not isinstance(rule.get("treatment_principle"), str):
            raise KnowledgeLoadError(f"{context}（{name}）缺少 treatment_principle")
        advice = rule.get("lifestyle_advice", [])
        if not isinstance(advice, list) or not all(isinstance(item, str) for item in advice):
            raise KnowledgeLoadError(f"{context}（{name}）lifestyle_advice 必须是字符串列表")

    return rules, data.get("version", "")


def _load_term_normalization():
    data = _load_json("term_normalization.json")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise KnowledgeLoadError("term_normalization.json 缺少 entries 列表")

    for index, entry in enumerate(entries):
        context = f"term_normalization.json entries[{index}]"
        canonical = entry.get("canonical")
        aliases = entry.get("aliases")
        category = entry.get("category")
        if not canonical or not isinstance(canonical, str):
            raise KnowledgeLoadError(f"{context} 缺少 canonical")
        if not isinstance(aliases, list) or not aliases or not all(isinstance(a, str) for a in aliases):
            raise KnowledgeLoadError(f"{context}（{canonical}）aliases 必须是非空字符串列表")
        _require_slot(category, context)

    return entries, data.get("version", "")


def _load_chief_complaint_followup():
    data = _load_json("chief_complaint_followup.json")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise KnowledgeLoadError("chief_complaint_followup.json 缺少 items 列表")

    priorities = {}
    for index, item in enumerate(items):
        context = f"chief_complaint_followup.json items[{index}]"
        complaint = item.get("chief_complaint")
        slots = item.get("priority_slots")
        if not complaint or not isinstance(complaint, str):
            raise KnowledgeLoadError(f"{context} 缺少 chief_complaint")
        if not isinstance(slots, list) or not slots:
            raise KnowledgeLoadError(f"{context}（{complaint}）priority_slots 必须是非空列表")
        for slot in slots:
            _require_slot(slot, context)
        priorities[complaint] = list(slots)

    return priorities, data.get("version", "")


def _load_red_flags():
    data = _load_json("red_flags.json")
    flags = data.get("flags")
    if not isinstance(flags, list) or not flags:
        raise KnowledgeLoadError("red_flags.json 缺少 flags 列表")

    for index, flag in enumerate(flags):
        context = f"red_flags.json flags[{index}]"
        label = flag.get("label")
        aliases = flag.get("aliases")
        if not label or not isinstance(label, str):
            raise KnowledgeLoadError(f"{context} 缺少 label")
        if not isinstance(aliases, list) or not aliases or not all(isinstance(a, str) for a in aliases):
            raise KnowledgeLoadError(f"{context}（{label}）aliases 必须是非空字符串列表")
        if flag.get("priority") not in _FLAG_PRIORITIES:
            raise KnowledgeLoadError(f"{context}（{label}）priority 必须是 {sorted(_FLAG_PRIORITIES)} 之一")
        if not flag.get("disposition") or not isinstance(flag["disposition"], str):
            raise KnowledgeLoadError(f"{context}（{label}）缺少 disposition")

    return flags, data.get("version", "")


# 模块加载时一次性载入并校验，失败直接暴露问题
SYNDROME_RULES, _SYNDROME_RULES_VERSION = _load_syndrome_rules()
_TERM_ENTRIES, _TERM_VERSION = _load_term_normalization()
CHIEF_COMPLAINT_PRIORITIES, _FOLLOWUP_VERSION = _load_chief_complaint_followup()
_RED_FLAGS, _RED_FLAGS_VERSION = _load_red_flags()

KNOWLEDGE_VERSION = {
    "syndrome_rules": _SYNDROME_RULES_VERSION,
    "term_normalization": _TERM_VERSION,
    "chief_complaint_followup": _FOLLOWUP_VERSION,
    "red_flags": _RED_FLAGS_VERSION,
}

# 别名 -> 规范词索引，按别名长度降序，保证长别名优先替换
_ALIAS_PAIRS = sorted(
    ((alias, entry["canonical"]) for entry in _TERM_ENTRIES for alias in entry["aliases"]),
    key=lambda pair: len(pair[0]),
    reverse=True,
)

# 证型名 -> 规则索引，供定向追问与知识检索使用
_SYNDROME_RULE_MAP = {rule["name"]: rule for rule in SYNDROME_RULES}


# ---------- 对外接口 ----------

def normalize_term(text):
    """把文本中的口语别名替换为规范中医术语，无匹配时原样返回。"""
    if not text or not isinstance(text, str):
        return text
    normalized = text
    for alias, canonical in _ALIAS_PAIRS:
        if alias in normalized:
            normalized = normalized.replace(alias, canonical)
    return normalized


def get_red_flags():
    """返回红旗信号表（副本），供抽取层构建匹配模式。"""
    return [dict(flag) for flag in _RED_FLAGS]


def get_term_entries():
    """返回术语归一化条目（副本），供知识检索工具使用。"""
    return [dict(entry) for entry in _TERM_ENTRIES]


def get_syndrome_rule(name):
    """按证型名返回规则条目，未命中返回 None。"""
    return _SYNDROME_RULE_MAP.get(name)


def get_syndrome_advice(name):
    """返回证型对应的治则方向与调理建议，未命中返回空结构。"""
    rule = _SYNDROME_RULE_MAP.get(name)
    if not rule:
        return {"treatment_principle": "", "lifestyle_advice": []}
    return {
        "treatment_principle": rule.get("treatment_principle", ""),
        "lifestyle_advice": list(rule.get("lifestyle_advice", [])),
    }


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
