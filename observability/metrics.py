# 可观测性：事件流指标汇总
# 从 JSONL 事件聚合出节点耗时、LLM 调用、replan、风险分布等统计
# 供 /api/debug/trace 的 summary 与网页侧栏展示


def summarize_events(events):
    """汇总一段事件流，返回调试面板需要的统计摘要。"""
    node_stats = {}
    llm_calls = 0
    llm_latency_ms = 0.0
    mock_fallbacks = 0
    tool_calls = 0
    tool_errors = 0
    reviews = 0
    replans = 0
    risk_distribution = {}
    turn_ids = set()

    for event in events:
        event_type = event.get("event")
        if event.get("turn_id"):
            turn_ids.add(event["turn_id"])

        if event_type == "node_exit" and not event.get("error"):
            node = event.get("node", "unknown")
            stat = node_stats.setdefault(node, {"count": 0, "total_ms": 0.0})
            stat["count"] += 1
            stat["total_ms"] += float(event.get("elapsed_ms", 0) or 0)
        elif event_type == "llm_call":
            llm_calls += 1
            llm_latency_ms += float(event.get("elapsed_ms", 0) or 0)
            if event.get("fallback"):
                mock_fallbacks += 1
        elif event_type == "tool_call":
            tool_calls += 1
            if event.get("status") == "error":
                tool_errors += 1
        elif event_type == "review":
            reviews += 1
            if event.get("needs_replan"):
                replans += 1
        elif event_type == "risk_assessed":
            risk = event.get("risk") or "UNKNOWN"
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1

    node_summary = {}
    for node, stat in node_stats.items():
        node_summary[node] = {
            "count": stat["count"],
            "avg_ms": round(stat["total_ms"] / stat["count"], 1) if stat["count"] else 0.0,
        }

    return {
        "total_events": len(events),
        "turn_count": len(turn_ids),
        "node_stats": node_summary,
        "llm_calls": llm_calls,
        "llm_avg_latency_ms": round(llm_latency_ms / llm_calls, 1) if llm_calls else 0.0,
        "mock_fallbacks": mock_fallbacks,
        "mock_fallback_rate": round(mock_fallbacks / llm_calls, 3) if llm_calls else 0.0,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "reviews": reviews,
        "replans": replans,
        "replan_rate": round(replans / reviews, 3) if reviews else 0.0,
        "risk_distribution": risk_distribution,
    }
