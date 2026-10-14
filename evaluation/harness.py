"""Replay fictional scenarios through the real LangGraph path.

The v1 runner supports the existing rule planner with Mock or DeepSeek extraction.
Future B/C variants can extend the provider/strategy registry without changing the
dataset or report shape. Holdout requires a reviewed freeze manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent.factory import create_agent
from evaluation.metrics import aggregate, legacy_flat_mentions
from evaluation.validate_dataset import SCHEMA_PATH, load_scenarios, sha256_file, validate_scenarios
from llm.llm import LLM
from llm.prompt import SYSTEM_PROMPT
from memory.memory import ConversationMemory
from memory.profile_store import InMemoryProfileStore
from memory.session_store import InMemorySessionStore
from observability.events import event_logger, setup_events
from tools.registry import default_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "evaluation" / "policy_v1.md"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


@contextmanager
def hospital_fixture(fixture_id: str):
    """Keep offline evaluation independent of MCP registration and live POI data."""
    original_get = default_registry.get

    def fixture_result(payload: dict):
        def call(**_kwargs):
            event_logger.emit(
                "tool_call", tool="hospital_locator_search_nearby_hospitals",
                version="fixture-v1", status=payload["status"], elapsed_ms=0.0,
                error=payload.get("error", ""), fixture_id=fixture_id,
            )
            return payload
        return call

    def fixed_get(name):
        if name != "hospital_locator_search_nearby_hospitals":
            return original_get(name)
        if fixture_id in {"none", "hospital_empty"}:
            if fixture_id == "none":
                return None
            return fixture_result({"status": "ok", "data": {"hospitals": [], "source": "fixture", "note": "虚构空结果"}})
        if fixture_id == "hospital_error":
            return fixture_result({"status": "error", "error": "synthetic tool failure", "data": {}})
        if fixture_id == "hospital_demo":
            return fixture_result({"status": "ok", "data": {"hospitals": [{"name": "虚构示例医院A", "district": "示例区", "has_emergency": True}], "source": "fixture", "note": "虚构测试数据"}})
        raise ValueError(f"unknown fixture: {fixture_id}")

    with patch.object(default_registry, "get", side_effect=fixed_get):
        yield


def _new_agent(scenario: dict, provider: str, strict_provider: bool = False, decision_policy=None):
    sid = scenario["scenario_id"]
    user_id = f"eval-{sid}-{uuid.uuid4().hex[:8]}"
    session_id = f"eval-session-{sid}-{uuid.uuid4().hex[:8]}"
    profiles = InMemoryProfileStore()
    sessions = InMemorySessionStore()
    memory = ConversationMemory(profile_store=profiles, session_store=sessions, user_id=user_id, session_id=session_id)
    if scenario.get("initial_profile"):
        memory.apply_manual_profile(scenario["initial_profile"])
    if scenario.get("initial_session_state"):
        illegal = set(scenario["initial_session_state"]) - set(memory.case_state)
        if illegal:
            raise ValueError(f"{sid}: unknown initial_session_state fields: {sorted(illegal)}")
        memory.case_state.update(scenario["initial_session_state"])
        sessions.set_case_state(session_id, memory.case_state)
    llm = LLM(system_prompt=SYSTEM_PROMPT, provider=provider, strict=strict_provider)
    agent = create_agent(llm, memory, runtime="langgraph", thread_id=session_id, decision_policy=decision_policy)
    return agent, llm, session_id


def _turn_record(scenario: dict, turn_index: int, turn: dict, agent, llm, session_id: str, response: str, elapsed_ms: float, prior_event_count: int, prior_llm_metrics: dict) -> dict:
    values = agent.get_graph_state().values
    events = event_logger.read_events(session_id=session_id, limit=10000)[prior_event_count:]
    risk_result = values.get("risk_result") or {}
    plan = values.get("plan") or {}
    extracted = values.get("extracted_slots") or {}
    node_timings = [
        {"node": event.get("node"), "elapsed_ms": event.get("elapsed_ms"), "error": event.get("error", False)}
        for event in events if event.get("event") == "node_exit"
    ]
    metrics_now = llm.get_metrics()
    count_keys = ("call_count", "error_count", "mock_fallback_count", "total_prompt_tokens", "total_completion_tokens", "total_latency_ms")
    metrics_turn = {key: round(metrics_now[key] - prior_llm_metrics[key], 1) for key in count_keys}
    replan_count = max(0, sum(event.get("event") == "plan" for event in events) - 1)
    return {
        "scenario_id": scenario["scenario_id"],
        "family_id": scenario["family_id"],
        "split": scenario["split"],
        "review_status": scenario["review_status"],
        "tool_fixture_id": scenario.get("tool_fixture_id", "none"),
        "turn_id": turn_index,
        "user": turn["user"],
        "gold_mentions": turn["gold_extract"]["mentions"],
        "gold_policy": turn["gold_policy"],
        "predicted_mentions": legacy_flat_mentions(extracted),
        "prediction_format": "legacy_flat_proxy",
        "extracted_slots": extracted,
        "case_state": agent.memory.get_case_state(),
        "risk": risk_result.get("risk"),
        "risk_result": risk_result,
        "next_action": plan.get("next_action"),
        "plan": plan,
        "action_result": values.get("action_result") or {},
        "review_result": values.get("review_result") or {},
        "decision_trace": values.get("decision_trace") or [],
        "decision_policy": agent.decision_policy.summary() if agent.decision_policy else None,
        "response": response,
        "provider_requested": llm.provider,
        "provider_used": llm.last_provider_used,
        "model_id": os.getenv("DEEPSEEK_MODEL", "deepseek-chat") if llm.last_provider_used == "deepseek" else "mock",
        "degraded": llm.degraded,
        "estimated_cost_cny": None,
        "llm_metrics_cumulative": llm.get_metrics(),
        "llm_metrics_turn": metrics_turn,
        "replan_count": replan_count,
        "elapsed_ms": round(elapsed_ms, 1),
        "node_timings_ms": node_timings,
        "events": events,
    }


def run_scenario(scenario: dict, provider: str, strict_provider: bool = False, decision_policy=None) -> list[dict]:
    agent, llm, session_id = _new_agent(scenario, provider, strict_provider, decision_policy)
    records = []
    with hospital_fixture(scenario.get("tool_fixture_id", "none")):
        for turn_index, turn in enumerate(scenario["turns"], start=1):
            if turn.get("pulse_data"):
                agent.ingest_pulse_data(turn["pulse_data"])
            prior_events = len(event_logger.read_events(session_id=session_id, limit=10000))
            prior_llm_metrics = llm.get_metrics()
            started = time.perf_counter()
            response = agent.run(turn["user"], user_coords=scenario.get("location_fixture"))
            elapsed_ms = (time.perf_counter() - started) * 1000
            records.append(_turn_record(scenario, turn_index, turn, agent, llm, session_id, response, elapsed_ms, prior_events, prior_llm_metrics))
    return records


def _holdout_guard(dataset: Path, scenarios: list[dict], manifest_path: Path | None) -> None:
    if not any(item["split"] == "holdout" for item in scenarios):
        return
    errors = validate_scenarios(scenarios, require_reviewed=True)
    if errors:
        raise ValueError("holdout is not fully reviewed: " + "; ".join(errors[:5]))
    if not manifest_path or not manifest_path.exists():
        raise ValueError("holdout run requires --freeze-manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("dataset_sha256") != sha256_file(dataset) or manifest.get("policy_sha256") != sha256_file(POLICY_PATH) or manifest.get("schema_sha256") != sha256_file(SCHEMA_PATH):
        raise ValueError("holdout/policy hash differs from freeze manifest")


def _failure_lines(records: list[dict]) -> list[str]:
    lines = ["# Evaluation mismatches", "", "Draft rows are exploratory; reviewed rows use policy v1.", ""]
    for item in records:
        gold = item.get("gold_policy") or {}
        problems = []
        if gold.get("risk") and gold["risk"] != item.get("risk"):
            problems.append(f"risk expected {gold['risk']} got {item.get('risk')}")
        allowed = gold.get("allowed_actions") or []
        if allowed and item.get("next_action") not in allowed:
            problems.append(f"action expected one of {allowed} got {item.get('next_action')}")
        if item.get("error"):
            problems.append(item["error"])
        if problems:
            lines.append(f"- `{item['scenario_id']}` turn {item['turn_id']} ({item['review_status']}): {'; '.join(problems)}")
    if len(lines) == 4:
        lines.append("No policy mismatches or execution errors in this run.")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--variant", default="rule_mock", choices=["rule_mock", "rule_deepseek"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fail-on-mock", "--strict-provider", action="store_true", dest="fail_on_mock")
    parser.add_argument("--freeze-manifest", type=Path)
    args = parser.parse_args(argv)
    random.seed(args.seed)
    scenarios = load_scenarios(args.dataset)
    errors = validate_scenarios(scenarios)
    if errors:
        parser.error("invalid dataset: " + "; ".join(errors[:5]))
    _holdout_guard(args.dataset, scenarios, args.freeze_manifest)
    provider = "mock" if args.variant == "rule_mock" else "deepseek"
    if provider == "mock" and args.fail_on_mock:
        parser.error("--fail-on-mock is incompatible with rule_mock")
    args.out.mkdir(parents=True, exist_ok=True)
    setup_events(args.out / "trace")
    records: list[dict] = []
    execution_errors = []
    for scenario in scenarios:
        try:
            case_records = run_scenario(scenario, provider, args.fail_on_mock)
            records.extend(case_records)
            if args.fail_on_mock and any(item["provider_used"] == "mock" or item["degraded"] for item in case_records):
                execution_errors.append(f"{scenario['scenario_id']}: real provider degraded to Mock")
        except Exception as exc:
            execution_errors.append(f"{scenario['scenario_id']}: {type(exc).__name__}: {exc}")
            for turn_index, turn in enumerate(scenario["turns"], start=1):
                records.append({"scenario_id": scenario["scenario_id"], "family_id": scenario["family_id"], "split": scenario["split"], "review_status": scenario["review_status"], "turn_id": turn_index, "user": turn["user"], "gold_mentions": turn["gold_extract"]["mentions"], "gold_policy": turn["gold_policy"], "predicted_mentions": [], "risk": None, "next_action": None, "error": str(exc)})
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    metrics = aggregate(records)
    metrics["execution_errors"] = execution_errors
    (args.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "failures.md").write_text("\n".join(_failure_lines(records)) + "\n", encoding="utf-8")
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "dataset": str(args.dataset), "dataset_sha256": sha256_file(args.dataset),
        "policy_sha256": sha256_file(POLICY_PATH),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "knowledge_sha256": {name: sha256_file(PROJECT_ROOT / "knowledge" / "data" / name) for name in ("risk_rules.json", "red_flags.json", "term_normalization.json")},
        "variant": args.variant, "provider_requested": provider,
        "seed": args.seed, "scenario_count": len(scenarios), "turn_count": len(records),
        "intentionally_mock": provider == "mock", "execution_error_count": len(execution_errors),
        "config_file_exists": (PROJECT_ROOT / "config.env").exists(),
        "versions": {name: _version(name) for name in ("Flask", "langgraph", "mcp", "requests", "jsonschema")},
        "note": "Mock baseline is a deterministic engineering regression, not clinical accuracy or LLM quality",
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_dir": str(args.out), "official": metrics["official"], "draft": metrics["exploratory_draft"], "errors": execution_errors}, ensure_ascii=False, indent=2))
    return 1 if execution_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
