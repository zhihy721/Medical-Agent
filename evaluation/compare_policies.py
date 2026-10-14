"""Paired real-graph experiments; only Planner/Reviewer use real models.

The fixed rule/Mock extractor and template renderer isolate the strategy change.
Outputs append by scenario; a resumed run validates data, prompts, config and code.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from statistics import median
import yaml

from agent.decision_policy import DecisionPolicy, PLANNER_PROMPT, REVIEWER_PROMPT, digest
from agent.safety_gate import ACTIONS
from evaluation.harness import PROJECT_ROOT, POLICY_PATH, _git_head, _holdout_guard, _version, run_scenario
from evaluation.metrics import aggregate
from evaluation.validate_dataset import load_scenarios, sha256_file, validate_scenarios
from llm.clients import ModelClient, RoleConfig, RunBudget
from observability.events import setup_events

DEFAULT_CONFIG = PROJECT_ROOT / "evaluation/configs/b_comparison.yaml"
DEFAULT_DATA = PROJECT_ROOT / "evaluation/datasets/dev_reviewed_v1.jsonl"


def dump(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def append_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("extractor") != "rule_mock" or raw.get("renderer") != "deterministic_template":
        raise ValueError("This experiment must keep extractor/renderer fixed")
    if not 0 < float(raw["max_cost_cny"]) <= 100:
        raise ValueError("invalid run spending limit")
    if "rule_gate" not in raw["variants"]:
        raise ValueError("rule_gate baseline required")
    for roles in raw["variants"].values():
        if set(roles) != {"planner", "reviewer"}:
            raise ValueError("variant needs planner and reviewer")
        for role, model in roles.items():
            if model != "rule":
                RoleConfig.from_mapping(role, raw["models"][model])
    return raw


def make_policy(config, variant, budget, clients=None):
    selected = {}
    for role, model in config["variants"][variant].items():
        selected[role] = None if model == "rule" else (
            clients[(model, role)] if clients is not None else ModelClient(RoleConfig.from_mapping(role, config["models"][model]), budget=budget)
        )
    return DecisionPolicy(planner_client=selected["planner"], reviewer_client=selected["reviewer"], name=variant,
                          prompt_version=config.get("prompt_version", "v1"))


def stratified_cases(cases):
    families = defaultdict(list)
    for case in cases:
        families[case["family_id"]].append(case)
    ordered = []
    while any(families.values()):
        for family in sorted(families):
            if families[family]:
                ordered.append(families[family].pop(0))
    return ordered


def calls_for(record):
    return [a["model"] for a in record.get("decision_trace", []) if "model" in a]


def percentile(values, q):
    values = sorted(values)
    return round(values[min(len(values) - 1, int((len(values) - 1) * q + .999999))], 2) if values else None


def summarize(records):
    metrics = aggregate(records)["official"]
    audits = [a for r in records for a in r.get("decision_trace", [])]
    calls = [a["model"] for a in audits if "model" in a]
    active_times = [r["elapsed_ms"] for r in records if calls_for(r)]
    def violates(r):
        if r.get("error"):
            return False  # execution failures count separately, never as safe success
        name = (r.get("action_result") or {}).get("name", r.get("next_action"))
        return (name not in ACTIONS or (r.get("risk") == "HIGH" and name != "risk_escalation") or
                (bool(r.get("case_state", {}).get("contradictions")) and name == "final_advice"))
    metrics.update(
        model_calls=len(calls), model_statuses=dict(Counter(c["status"] for c in calls)),
        model_failure_calls=sum(c["status"] != "ok" for c in calls),
        schema_rejections=sum(a.get("validation_error") == "schema" for a in audits),
        semantic_rejections=sum(a.get("validation_error") == "review_evidence_or_action" for a in audits),
        gate_rejections=sum((a.get("gate") or {}).get("status") == "fallback" for a in audits),
        fallback_decisions=sum(bool(a.get("fallback_used")) for a in audits),
        gate_overrides=sum((a.get("gate") or {}).get("status") in {"forced", "fallback"} for a in audits),
        hard_constraint_violations=sum(violates(r) for r in records),
        escalation_misses=sum(r.get("gold_policy", {}).get("risk") == "HIGH" and r.get("next_action") != "risk_escalation" for r in records),
        replan_exhausted=sum(bool(a.get("replan_exhausted")) for a in audits),
        total_tokens=sum(c["usage"].get("total_tokens", 0) for c in calls),
        api_cost_estimate_cny=round(sum(c.get("estimated_cost_cny") or 0 for c in calls), 6),
        cost_unknown_calls=sum(c.get("estimated_cost_cny") is None for c in calls),
        model_active_turn_latency_ms={"count": len(active_times), "p50": round(median(active_times), 2) if active_times else None, "p95": percentile(active_times, .95)},
    )
    metrics["role_latency_ms"] = {}
    for stage in ("plan", "review"):
        durations = [a["model"]["latency_ms"] for a in audits if a["stage"] == stage and "model" in a]
        metrics["role_latency_ms"][stage] = {"count": len(durations), "p50": median(durations) if durations else None, "p95": percentile(durations, .95)}
    return metrics


def write_report(out, records, manifest):
    grouped = {v: [r for r in records if r["variant"] == v] for v in manifest["variants"]}
    metrics = {v: summarize(rows) for v, rows in grouped.items()}
    dump(out / "metrics.json", metrics)
    keyed = {v: {(r["scenario_id"], r["turn_id"]): r for r in rows} for v, rows in grouped.items()}
    paired = []
    for variant, mapping in keyed.items():
        if variant == "rule_gate":
            continue
        for key, row in mapping.items():
            base = keyed["rule_gate"].get(key)
            if base is None:
                continue
            allowed = row["gold_policy"].get("allowed_actions", [])
            paired.append({"scenario_id": key[0], "turn_id": key[1], "variant": variant,
                "baseline_action": base.get("next_action"), "candidate_action": row.get("next_action"),
                "baseline_correct": base.get("next_action") in allowed, "candidate_correct": row.get("next_action") in allowed,
                "response_changed": base.get("response") != row.get("response"),
                "extraction_identical": base.get("extracted_slots") == row.get("extracted_slots"),
                "latency_delta_ms": round(row.get("elapsed_ms", 0) - base.get("elapsed_ms", 0), 2)})
    (out / "paired_differences.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in paired), encoding="utf-8")
    lines = ["# 阶段 B：开发集策略对照", "", f"状态：{manifest['status']}；场景 {manifest['completed_scenarios']} / {len(manifest['scenario_ids'])}；数据 hash `{manifest['dataset_sha256']}`。", "",
        "所有变体共用原有规则/Mock 抽取、确定性回复模板、policy_v1、SafetyGate、医院 fixture；只有 Planner/Reviewer 使用真实模型。HIGH 由门禁短路，不计为模型规划成功。按固定用户轮次回放，尚未模拟用户根据追问动态回答。", "",
        "| 变体 | 动作符合政策 | 风险等级漏 HIGH | 动作漏升级 | 门禁违规 | 失败调用/全部 | 回退决策 | 总体 p50/p95 ms | API 估算元 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, m in metrics.items():
        a, h = m["allowed_action_accuracy"], m["high_risk_misses"]
        lines.append(f"| {variant} | {a['numerator']}/{a['denominator']} | {h['numerator']}/{h['denominator']} | {m['escalation_misses']} | {m['hard_constraint_violations']} | {m['model_failure_calls']}/{m['model_calls']} | {m['fallback_decisions']} | {m['latency_ms']['p50']}/{m['latency_ms']['p95']} | {m['api_cost_estimate_cny']:.6f} |")
    lines += ["", "高风险短路会拉低总体时延；model_active_turn_latency_ms 和 role_latency_ms 单列真实模型调用时延。费用按配置高峰单价与返回 usage/cache 估算，非账单；本地 API 费为零，硬件电费未计。本轮无专门性能预热、每例单次运行，不作为 D 阶段推理基准。开发集用于实现验证与错误分析，不是独立冻结测试成绩。", "",
        "配对差异见 paired_differences.jsonl，原始输出/门禁/回复见 predictions.jsonl。DeepSeek 同时作为云端候选和 Judge 时存在同模型偏好风险，硬指标及人工盲评优先。", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def runtime_hashes():
    paths = [PROJECT_ROOT / f"evaluation/{name}.py" for name in ("compare_policies", "harness", "metrics", "validate_dataset")]
    for folder in ("agent", "llm", "tools", "knowledge", "memory", "observability"):
        paths.extend(p for p in (PROJECT_ROOT / folder).rglob("*.py") if not p.name.startswith("test_"))
    paths.extend((PROJECT_ROOT / "knowledge/data").glob("*.json"))
    return {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sorted(paths)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=0, help="Pause after N newly completed cases, then --resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-manifest", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    cases = load_scenarios(args.dataset)
    errors = validate_scenarios(cases, require_reviewed=True)
    if errors:
        parser.error("; ".join(errors[:3]))
    _holdout_guard(args.dataset, cases, args.freeze_manifest)
    cases = stratified_cases(cases)
    args.out.mkdir(parents=True, exist_ok=True)
    prompt_policy = DecisionPolicy(prompt_version=config.get("prompt_version", "v1"))
    fingerprint = {"dataset_sha256": sha256_file(args.dataset), "config_sha256": sha256_file(args.config),
        "policy_sha256": sha256_file(POLICY_PATH), "planner_prompt_sha256": digest(prompt_policy.planner_prompt),
        "reviewer_prompt_sha256": digest(prompt_policy.reviewer_prompt), "seed": args.seed,
        "scenario_ids": [c["scenario_id"] for c in cases], "variants": list(config["variants"]), "code_sha256": runtime_hashes()}
    manifest_path = args.out / "run_manifest.json"
    records = read_jsonl(args.out / "predictions.jsonl")
    if manifest_path.exists():
        if not args.resume:
            parser.error("Output exists; use a new directory or --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in fingerprint.items()):
            parser.error("Data/config/code/prompts differ; create a new run")
        if manifest.get("in_flight"):
            parser.error("Interrupted scenario requires inspection; preserve this run and use a new directory")
    else:
        if records:
            parser.error("Predictions without a manifest")
        manifest = {**fingerprint, "started_at_utc": datetime.now(timezone.utc).isoformat(), "git_head": _git_head(),
            "config": config, "status": "running", "completed_scenarios": 0, "budget_accounted_cny": 0,
            "versions": {n: _version(n) for n in ("langgraph", "requests", "jsonschema", "PyYAML")},
            "note": "Development engineering policy evaluation; fixed rule/Mock extraction, not clinical validation."}
        dump(manifest_path, manifest)
    budget = RunBudget(float(config["max_cost_cny"]), manifest["budget_accounted_cny"])
    policies = {v: make_policy(config, v, budget) for v in config["variants"]}
    completed = Counter((r["variant"], r["scenario_id"]) for r in records)
    setup_events(args.out / "trace")
    stop = False
    new_cases = 0
    for index, case in enumerate(cases):
        variants = list(policies)
        random.Random(args.seed + index).shuffle(variants)
        case_new = False
        for variant in variants:
            key = (variant, case["scenario_id"])
            if completed[key]:
                if completed[key] != len(case["turns"]):
                    parser.error("Partial scenario requires inspection")
                continue
            case_new = True
            manifest["in_flight"] = {"variant": variant, "scenario_id": case["scenario_id"]}
            dump(manifest_path, manifest)
            try:
                rows = run_scenario(case, "mock", decision_policy=policies[variant])
            except Exception as exc:
                rows = [{"scenario_id": case["scenario_id"], "family_id": case["family_id"], "split": case["split"],
                    "review_status": case["review_status"], "turn_id": n, "user": turn["user"],
                    "gold_mentions": turn["gold_extract"]["mentions"], "gold_policy": turn["gold_policy"],
                    "predicted_mentions": [], "next_action": None, "risk": None,
                    "error": type(exc).__name__ + ": " + str(exc)} for n, turn in enumerate(case["turns"], 1)]
                stop = True
            for row in rows:
                row.update(variant=variant, input_history=[t["user"] for t in case["turns"][:row["turn_id"]]], tags=case["tags"])
            append_jsonl(args.out / "predictions.jsonl", rows)
            records.extend(rows)
            completed[key] = len(rows)
            manifest.update(in_flight=None, budget_accounted_cny=budget.accounted_cny)
            if any(c["status"] in {"budget_exceeded", "missing_key", "network_error", "http_error", "timeout"} for r in rows for c in calls_for(r)):
                stop = True
            dump(manifest_path, manifest)
            print(json.dumps({"variant": variant, "scenario": case["scenario_id"], "turns": len(rows), "budget_cny": round(budget.accounted_cny, 6)}, ensure_ascii=False), flush=True)
            if stop or budget.stopped:
                stop = True
                break
        manifest["completed_scenarios"] = sum(all(completed[(v, c["scenario_id"])] == len(c["turns"]) for v in policies) for c in cases)
        new_cases += int(case_new)
        if stop or (args.stop_after and new_cases >= args.stop_after):
            break
    status = "stopped" if stop else ("completed" if manifest["completed_scenarios"] == len(cases) else "paused")
    manifest.update(status=status, updated_at_utc=datetime.now(timezone.utc).isoformat())
    dump(manifest_path, manifest)
    write_report(args.out, records, manifest)
    return 1 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
