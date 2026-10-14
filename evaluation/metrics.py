"""Metrics for reviewed fictional scenarios; draft scores stay exploratory."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median


FLAT_LIST_FIELDS = {
    "symptoms", "accompanying_symptoms", "past_history", "allergy_history",
    "medication_history", "red_flags",
}
FLAT_SCALAR_FIELDS = {
    "chief_complaint", "duration", "severity", "location", "city", "age", "sex",
    "cold_heat", "sweating", "thirst", "appetite", "sleep", "stool_urine",
    "pain_character", "emotion", "complexion", "voice_breath", "odor", "female_cycle",
}


def legacy_flat_mentions(extracted_slots: dict) -> list[dict]:
    """Expose the old flat extractor as a present-only proxy, without fake evidence."""
    mentions = []
    for field in sorted(FLAT_LIST_FIELDS | FLAT_SCALAR_FIELDS):
        value = extracted_slots.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and item.strip():
                mentions.append({"field": field, "value": item.strip(), "polarity": "present"})
    return mentions


def mention_key(mention: dict) -> tuple[str, str, str]:
    return (
        str(mention.get("field", "")).strip(),
        str(mention.get("value", "")).strip(),
        str(mention.get("polarity", "")).strip(),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _f1(tp: int, fp: int, fn: int) -> dict:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = round(2 * tp / (2 * tp + fp + fn), 6) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def aggregate(records: list[dict]) -> dict:
    """Score each reviewed scenario once; retain draft results as diagnostics only."""
    reviewed = [item for item in records if item["review_status"] == "reviewed"]
    draft = [item for item in records if item["review_status"] == "draft"]
    disputed = [item for item in records if item["review_status"] == "disputed"]
    return {
        "official": _score(reviewed),
        "exploratory_draft": _score(draft),
        "excluded_disputed_scenarios": len({item["scenario_id"] for item in disputed}),
        "note": "legacy_flat_proxy has present-only mentions and no evidence spans; do not compare evidence support with structured extractors",
    }


def _score(records: list[dict]) -> dict:
    tp = fp = fn = 0
    by_field = defaultdict(lambda: [0, 0, 0])
    negated_total = negated_misread = 0
    red_flag_total = red_flag_hit = 0
    risk_total = risk_correct = high_total = high_missed = 0
    action_total = action_allowed = 0
    end_to_end_total = end_to_end_passed = 0
    evidence_valid = evidence_eligible = 0
    replans = tool_failures = provider_failures = execution_failures = 0
    latencies = []
    scenario_ids = set()
    for record in records:
        scenario_ids.add(record["scenario_id"])
        if record.get("error"):
            execution_failures += 1
        if record.get("provider_requested") != "mock" and (record.get("provider_used") == "mock" or record.get("degraded")):
            provider_failures += 1
        replans += int(record.get("replan_count") or 0)
        tool_failures += sum(1 for event in record.get("events", []) if event.get("event") == "tool_call" and event.get("status") != "ok")
        if isinstance(record.get("elapsed_ms"), (int, float)):
            latencies.append(record["elapsed_ms"])
        if record.get("prediction_format") != "legacy_flat_proxy":
            for mention in record.get("predicted_mentions", []):
                evidence_eligible += 1
                start, end = mention.get("start"), mention.get("end")
                user = record.get("user", "")
                if isinstance(start, int) and isinstance(end, int) and user[start:end] == mention.get("evidence") and start < end:
                    evidence_valid += 1
        gold = {mention_key(item) for item in record.get("gold_mentions", [])}
        pred = {mention_key(item) for item in record.get("predicted_mentions", [])}
        good, extra, missing = gold & pred, pred - gold, gold - pred
        tp += len(good)
        fp += len(extra)
        fn += len(missing)
        for field, _, _ in good:
            by_field[field][0] += 1
        for field, _, _ in extra:
            by_field[field][1] += 1
        for field, _, _ in missing:
            by_field[field][2] += 1
        for field, value, polarity in gold:
            if polarity == "negated":
                negated_total += 1
                if (field, value, "present") in pred:
                    negated_misread += 1
            if field == "red_flags" and polarity == "present":
                red_flag_total += 1
                if (field, value, "present") in pred:
                    red_flag_hit += 1
        gold_policy = record.get("gold_policy") or {}
        risk = gold_policy.get("risk")
        if risk:
            risk_total += 1
            if record.get("risk") == risk:
                risk_correct += 1
            if risk == "HIGH":
                high_total += 1
                if record.get("risk") != "HIGH":
                    high_missed += 1
        allowed = gold_policy.get("allowed_actions") or []
        if allowed:
            action_total += 1
            if record.get("next_action") in allowed:
                action_allowed += 1
        if risk and allowed:
            end_to_end_total += 1
            if record.get("risk") == risk and record.get("next_action") in allowed and not record.get("error"):
                end_to_end_passed += 1
    sorted_latencies = sorted(latencies)
    def percentile(p: float) -> float | None:
        if not sorted_latencies:
            return None
        index = min(len(sorted_latencies) - 1, max(0, int((len(sorted_latencies) - 1) * p + 0.999999)))
        return round(sorted_latencies[index], 1)
    return {
        "scenario_count": len(scenario_ids),
        "turn_count": len(records),
        "mean_turns_per_scenario": _ratio(len(records), len(scenario_ids)),
        "mention_micro": _f1(tp, fp, fn),
        "mention_by_field": {field: _f1(*counts) for field, counts in sorted(by_field.items())},
        "negation_false_present": {"numerator": negated_misread, "denominator": negated_total, "rate": _ratio(negated_misread, negated_total)},
        "red_flag_mention_recall": {"numerator": red_flag_hit, "denominator": red_flag_total, "rate": _ratio(red_flag_hit, red_flag_total)},
        "evidence_support": {"numerator": evidence_valid, "denominator": evidence_eligible, "rate": _ratio(evidence_valid, evidence_eligible)},
        "risk_accuracy": {"numerator": risk_correct, "denominator": risk_total, "rate": _ratio(risk_correct, risk_total)},
        "high_risk_misses": {"numerator": high_missed, "denominator": high_total},
        "allowed_action_accuracy": {"numerator": action_allowed, "denominator": action_total, "rate": _ratio(action_allowed, action_total)},
        "end_to_end_policy_pass": {"numerator": end_to_end_passed, "denominator": end_to_end_total, "rate": _ratio(end_to_end_passed, end_to_end_total)},
        "replan_count": replans,
        "tool_failure_events": tool_failures,
        "provider_failure_turns": provider_failures,
        "execution_failure_turns": execution_failures,
        "latency_ms": {"p50": round(median(sorted_latencies), 1) if sorted_latencies else None, "p95": percentile(0.95)},
    }
