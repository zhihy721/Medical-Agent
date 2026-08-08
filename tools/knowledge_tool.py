# 中医知识检索工具：在证型规则、术语归一化、主诉追问优先级中做加权检索
# 纯 Python 实现（包含匹配 + 字段加权打分），零外部依赖，结果确定性强
from knowledge.tcm_knowledge import (
    CHIEF_COMPLAINT_PRIORITIES,
    KNOWLEDGE_VERSION,
    SYNDROME_RULES,
    get_term_entries,
)
from tools.protocol import managed_tool

TOOL_VERSION = "1.0"

# 加权配置：不同匹配位置的重要性不同
_SCORES = {
    "syndrome_name_exact": 10,
    "syndrome_name_partial": 6,
    "syndrome_evidence": 2,
    "syndrome_symptom": 2,
    "syndrome_principle": 1,
    "term_canonical": 4,
    "term_alias": 3,
    "chief_complaint": 5,
}


def _search_syndromes(query):
    hits = []
    for rule in SYNDROME_RULES:
        name = rule["name"]
        score = 0
        if name == query:
            score += _SCORES["syndrome_name_exact"]
        elif query in name or name in query:
            score += _SCORES["syndrome_name_partial"]

        for options in rule.get("evidence", {}).values():
            if any(query in option or option in query for option in options):
                score += _SCORES["syndrome_evidence"]

        symptom_group = rule.get("symptoms", {})
        symptom_pool = list(symptom_group.get("symptoms", [])) + list(symptom_group.get("accompanying_symptoms", []))
        if any(query in symptom or symptom in query for symptom in symptom_pool):
            score += _SCORES["syndrome_symptom"]

        if query in rule.get("treatment_principle", ""):
            score += _SCORES["syndrome_principle"]

        if score > 0:
            hits.append(
                {
                    "type": "syndrome",
                    "name": name,
                    "score": score,
                    "treatment_principle": rule.get("treatment_principle", ""),
                    "lifestyle_advice": list(rule.get("lifestyle_advice", []))[:2],
                    "source": "knowledge/data/syndrome_rules.json",
                }
            )
    return hits


def _search_terms(query):
    hits = []
    for entry in get_term_entries():
        canonical = entry["canonical"]
        if canonical == query or query in canonical or canonical in query:
            score = _SCORES["term_canonical"]
        elif any(query in alias or alias in query for alias in entry.get("aliases", [])):
            score = _SCORES["term_alias"]
        else:
            continue
        hits.append(
            {
                "type": "term",
                "name": canonical,
                "score": score,
                "category": entry.get("category", ""),
                "aliases": list(entry.get("aliases", []))[:4],
                "source": "knowledge/data/term_normalization.json",
            }
        )
    return hits


def _search_chief_complaints(query):
    hits = []
    for complaint, slots in CHIEF_COMPLAINT_PRIORITIES.items():
        if complaint == query or query in complaint or complaint in query:
            hits.append(
                {
                    "type": "chief_complaint",
                    "name": complaint,
                    "score": _SCORES["chief_complaint"],
                    "priority_slots": list(slots),
                    "source": "knowledge/data/chief_complaint_followup.json",
                }
            )
    return hits


@managed_tool("knowledge_retrieval", TOOL_VERSION, "检索中医知识库：证型规则、术语归一化与主诉追问优先级")
def search_knowledge_tool(query, top_k=5):
    """协议版入口：返回标准 ToolResult，异常由 managed_tool 捕获。"""
    query = (query or "").strip()
    if not query:
        return {"query": "", "hits": [], "total": 0, "knowledge_version": KNOWLEDGE_VERSION}

    hits = _search_syndromes(query) + _search_terms(query) + _search_chief_complaints(query)
    hits.sort(key=lambda item: (-item["score"], item["name"]))
    top_k = max(1, min(int(top_k), 20))
    return {
        "query": query,
        "hits": hits[:top_k],
        "total": len(hits),
        "knowledge_version": KNOWLEDGE_VERSION,
    }


def search_knowledge(query, top_k=5):
    """兼容层：保持旧式调用风格，返回检索结果 dict。"""
    result = search_knowledge_tool(query, top_k=top_k)
    if result["status"] != "ok":
        return {"query": query or "", "hits": [], "total": 0, "knowledge_version": KNOWLEDGE_VERSION}
    return result["data"]
