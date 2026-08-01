# 正则表达式工具，用于从用户输入的文本中提取症状、伴随症状、病史、过敏史等信息
import re

SYMPTOM_SYNONYMS = {
    "发热": ["发热", "发烧", "高烧", "低烧", "体温高"],
    "咳嗽": ["咳嗽", "干咳", "咳痰"],
    "胸痛": ["胸痛", "胸口痛", "心前区痛"],
    "胸闷": ["胸闷"],
    "腹痛": ["腹痛", "肚子痛", "胃痛", "肚子不舒服"],
    "头痛": ["头痛", "头疼", "偏头痛"],
    "头晕": ["头晕", "眩晕", "眼前发黑"],
    "恶心": ["恶心", "反胃"],
    "呕吐": ["呕吐", "吐了"],
    "乏力": ["乏力", "没力气", "疲劳", "很累"],
    "心悸": ["心悸", "心慌", "心跳快"],
    "呼吸困难": ["呼吸困难", "喘不过气", "气短", "胸闷喘"],
    "腹泻": ["腹泻", "拉肚子", "稀便"],
    "咽痛": ["咽痛", "喉咙痛", "嗓子痛"],
}

ACCOMPANYING_SYMPTOMS = {
    "寒战": ["寒战", "发冷", "畏寒"],
    "出汗": ["出汗", "冷汗"],
    "流涕": ["流鼻涕", "鼻塞"],
    "食欲下降": ["没胃口", "食欲差", "吃不下", "纳差"],
}

LOCATION_PATTERNS = [
    (r"(头部|脑袋|太阳穴)", "头部"),
    (r"(胸口|左胸|右胸|心前区)", "胸部"),
    (r"(上腹|胃部)", "上腹部"),
    (r"(腹部|肚子|肚脐周围)", "腹部"),
    (r"(咽喉|嗓子)", "咽喉"),
    (r"(腰部|后腰)", "腰部"),
]

SEVERITY_PATTERNS = [
    ("轻度", ["轻微", "轻度", "一点点"]),
    ("中度", ["中度", "比较明显", "有点重"]),
    ("重度", ["严重", "剧烈", "很重", "受不了"]),
]

RED_FLAG_PATTERNS = {
    "持续胸痛": ["持续胸痛", "胸痛不缓解"],
    "呼吸困难": ["呼吸困难", "喘不过气", "无法呼吸"],
    "意识障碍": ["意识模糊", "昏迷", "晕厥", "失去意识"],
    "突发剧烈头痛": ["突发剧烈头痛", "突然头痛很厉害"],
    "便血": ["便血", "黑便"],
}

TCM_FIELD_PATTERNS = {
    "cold_heat": {
        "恶寒": ["恶寒", "怕冷", "发冷", "畏寒"],
        "发热": ["发热", "发烧", "身上发烫"],
        "寒热往来": ["一会冷一会热", "寒热往来"],
        "怕热": ["怕热", "身上燥热"],
        "喜温": ["喜温", "喜欢热敷", "热一点舒服"],
        "喜冷": ["喜冷", "喜欢冷饮", "凉一点舒服"],
    },
    "sweating": {
        "无汗": ["无汗", "不出汗"],
        "自汗": ["自汗", "容易出汗", "动一动就出汗"],
        "盗汗": ["盗汗", "夜里出汗"],
        "多汗": ["出汗多", "汗很多"],
    },
    "thirst": {
        "口渴": ["口渴", "总想喝水"],
        "不渴": ["不渴", "不太想喝水"],
        "喜冷饮": ["喜欢冷饮", "想喝凉水"],
        "喜热饮": ["喜欢热饮", "想喝热水"],
    },
    "appetite": {
        "纳差": ["纳差", "没胃口", "食欲差", "吃不下"],
        "食欲正常": ["胃口还行", "食欲正常"],
        "易饥": ["容易饿", "总想吃东西"],
    },
    "sleep": {
        "失眠": ["失眠", "睡不着"],
        "多梦": ["多梦", "梦多"],
        "嗜睡": ["嗜睡", "总想睡"],
    },
    "stool_urine": {
        "便溏": ["便溏", "大便稀", "拉稀"],
        "便秘": ["便秘", "大便干"],
        "小便黄": ["小便黄", "尿黄"],
        "小便清长": ["小便清", "尿清", "尿多"],
    },
    "pain_character": {
        "刺痛": ["刺痛", "针扎样痛"],
        "胀痛": ["胀痛", "发胀"],
        "隐痛": ["隐痛", "隐隐作痛"],
        "喜按": ["按着舒服", "喜欢按", "喜按"],
        "拒按": ["按着更痛", "不让按", "拒按"],
    },
    "emotion": {
        "焦虑": ["焦虑", "紧张"],
        "易怒": ["易怒", "爱生气"],
        "抑郁": ["郁闷", "情绪低落", "压抑"],
        "烦躁": ["烦躁", "坐立不安"],
    },
    "complexion": {
        "面色苍白": ["脸色白", "面色苍白"],
        "面色萎黄": ["面色萎黄", "脸色发黄"],
        "面红": ["面红", "脸红"],
        "精神差": ["没精神", "精神差"],
    },
    "voice_breath": {
        "气短懒言": ["气短懒言", "说话没劲", "不想说话"],
        "声音低弱": ["声音低弱", "声音小"],
        "口气重": ["口气重", "嘴里味大"],
    },
    "odor": {
        "口气重": ["口气重", "嘴里味大"],
        "痰味重": ["痰味重"],
    },
    "female_cycle": {
        "月经推迟": ["月经推迟"],
        "月经提前": ["月经提前"],
        "痛经": ["痛经"],
        "带下多": ["带下多", "白带多"],
    },
}

DURATION_REGEXES = [
    re.compile(r"(\d+\s*(分钟|小时|天|周|个月))"),
    re.compile(r"(半小时|半天|一天|两天|三天|一周|一个月)"),
]

AGE_PATTERN = re.compile(r"(\d{1,3})\s*岁")

SEX_KEYWORDS = {
    "男": ["男", "男性", "先生"],
    "女": ["女", "女性", "女士"],
}

HISTORY_KEYWORDS = {
    "高血压": ["高血压"],
    "糖尿病": ["糖尿病"],
    "冠心病": ["冠心病"],
    "哮喘": ["哮喘"],
    "胃病": ["胃病", "胃炎", "胃溃疡"],
}

ALLERGY_KEYWORDS = {
    "药物过敏": ["药物过敏", "青霉素过敏"],
    "食物过敏": ["食物过敏"],
}

MEDICATION_PATTERN = re.compile(r"(吃了|服用|正在用)([^，。；,\n]{1,20})")


def _find_labels(text, mapping):
    labels = []
    for label, keywords in mapping.items():
        if any(keyword in text for keyword in keywords):
            labels.append(label)
    return labels


def _extract_duration(text):
    for pattern in DURATION_REGEXES:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return ""


def _extract_location(text):
    for pattern, label in LOCATION_PATTERNS:
        if re.search(pattern, text):
            return label
    return ""


def _extract_severity(text):
    for label, keywords in SEVERITY_PATTERNS:
        if any(keyword in text for keyword in keywords):
            return label
    return ""


def _extract_age(text):
    match = AGE_PATTERN.search(text)
    return match.group(1) if match else ""


def _extract_sex(text):
    for label, keywords in SEX_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return label
    return ""


def _extract_medications(text):
    meds = []
    for _, med in MEDICATION_PATTERN.findall(text):
        med = med.strip()
        if med and med not in meds:
            meds.append(med)
    return meds


def _extract_tcm_fields(text):
    result = {}
    for field, label_mapping in TCM_FIELD_PATTERNS.items():
        for label, keywords in label_mapping.items():
            if any(keyword in text for keyword in keywords):
                result[field] = label
                break
        else:
            result[field] = ""
    return result


def extract_symptoms(text):
    normalized_text = text.strip().lower()
    symptoms = _find_labels(normalized_text, SYMPTOM_SYNONYMS)
    accompanying = _find_labels(normalized_text, ACCOMPANYING_SYMPTOMS)
    history = _find_labels(normalized_text, HISTORY_KEYWORDS)
    allergies = _find_labels(normalized_text, ALLERGY_KEYWORDS)
    red_flags = _find_labels(normalized_text, RED_FLAG_PATTERNS)
    tcm_fields = _extract_tcm_fields(normalized_text)

    return {
        "chief_complaint": symptoms[0] if symptoms else "",
        "symptoms": symptoms,
        "accompanying_symptoms": accompanying,
        "duration": _extract_duration(normalized_text),
        "severity": _extract_severity(normalized_text),
        "location": _extract_location(normalized_text),
        "age": _extract_age(normalized_text),
        "sex": _extract_sex(normalized_text),
        "past_history": history,
        "allergy_history": allergies,
        "medication_history": _extract_medications(normalized_text),
        "red_flags": red_flags,
        "description": text.strip(),
        **tcm_fields,
    }
