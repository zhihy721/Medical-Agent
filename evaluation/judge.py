"""Read-only, blinded offline pair scoring with order reversal and resumable output."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from jsonschema import Draft202012Validator

from agent.decision_policy import digest
from evaluation.compare_policies import DEFAULT_CONFIG, append_jsonl, dump, load_config, read_jsonl
from evaluation.validate_dataset import sha256_file
from evaluation.metrics import FLAT_LIST_FIELDS, FLAT_SCALAR_FIELDS
from llm.clients import ModelClient, RoleConfig, RunBudget

RUBRIC_PATH = Path(__file__).with_name("judge_rubric_v1.md")
DIMENSIONS = ("relevance", "clarity", "groundedness")
SCORES = {"type": "object", "required": list(DIMENSIONS), "additionalProperties": False,
          "properties": {key: {"type": "integer", "minimum": 0, "maximum": 2} for key in DIMENSIONS}}
SCHEMA = {"type": "object", "required": ["A", "B", "winner", "reason"], "additionalProperties": False,
          "properties": {"A": SCORES, "B": SCORES, "winner": {"enum": ["A", "B", "tie"]},
                         "reason": {"type": "string", "minLength": 1, "maxLength": 500}}}


def system_prompt():
    return (RUBRIC_PATH.read_text(encoding="utf-8") +
            '\n只输出 JSON 对象 {"A":{"relevance":0,"clarity":0,"groundedness":0},"B":{"relevance":0,"clarity":0,"groundedness":0},"winner":"tie","reason":"简短依据"}。')


def winner(a, b):
    sa, sb = sum(a.values()), sum(b.values())
    return "A" if sa > sb else ("B" if sb > sa else "tie")


def validate_score(value):
    if not Draft202012Validator(SCHEMA).is_valid(value):
        raise ValueError("judge_schema")
    if value["winner"] != winner(value["A"], value["B"]):
        raise ValueError("winner_score_mismatch")
    return value


def blind_answer(row):
    return {"response": row.get("response", ""), "action": row.get("next_action"),
            "question_targets": row.get("plan", {}).get("question_targets", []),
            "trajectory": [{"stage": a["stage"], "effective_action": (a.get("gate") or {}).get("effective_action"),
                            "replan_exhausted": a.get("replan_exhausted", False)} for a in row.get("decision_trace", [])]}


def build_packet(records, count, seed):
    by_case = {}
    for row in records:
        by_case.setdefault(row["scenario_id"], {}).setdefault(row["turn_id"], {})[row["variant"]] = row
    families = {}
    for sid, turns in by_case.items():
        last = turns[max(turns)]
        if "rule_gate" not in last or any(r.get("error") for r in last.values()):
            continue
        family = last["rule_gate"]["family_id"]
        families.setdefault(family, []).append(sid)
    ordered = []
    while any(families.values()):
        for family in sorted(families):
            if families[family]:
                ordered.append(families[family].pop(0))
    packets, identities = [], []
    for index, sid in enumerate(ordered[:count]):
        turn = max(by_case[sid])
        mapping = by_case[sid][turn]
        candidates = sorted(set(mapping) - {"rule_gate"})
        if not candidates:
            continue
        candidate = candidates[index % len(candidates)]
        pair = [mapping["rule_gate"], mapping[candidate]]
        random.Random(seed + index).shuffle(pair)
        item_id = f"blind-{index + 1:03d}"
        packets.append({"item_id": item_id, "scenario_id": sid, "turn_id": turn,
                        "input_history": pair[0].get("input_history", [pair[0]["user"]]),
                        "observed_state": {key: value for key, value in mapping["rule_gate"].get("case_state", {}).items() if key in FLAT_LIST_FIELDS | FLAT_SCALAR_FIELDS},
                        "risk_result": mapping["rule_gate"].get("risk_result", {}),
                        "A": blind_answer(pair[0]), "B": blind_answer(pair[1])})
        # Separate key: neither the Judge prompt nor human-review endpoint reads it.
        identities.append({"item_id": item_id, "A": pair[0]["variant"], "B": pair[1]["variant"]})
    return packets, identities


def normalized_score(row):
    score = row["score"]
    if row["order"] == "AB":
        return score
    return {**score, "A": score["B"], "B": score["A"],
            "winner": {"A": "B", "B": "A", "tie": "tie"}[score["winner"]]}


def judge_summary(rows):
    paired = {}
    for row in rows:
        if row["status"] == "ok":
            paired.setdefault(row["item_id"], {})[row["order"]] = normalized_score(row)
    both = [v for v in paired.values() if set(v) == {"AB", "BA"}]
    consistent = sum(v["AB"]["winner"] == v["BA"]["winner"] for v in both)
    return {"calls": len(rows), "valid_calls": sum(r["status"] == "ok" for r in rows),
            "order_pairs": len(both), "winner_order_agreement": consistent / len(both) if both else None,
            "api_cost_estimate_cny": round(sum(r["model"].get("estimated_cost_cny") or 0 for r in rows), 6),
            "human_calibration": "pending", "note": "Judge opinion is auxiliary; raw AB and BA scores remain separate."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cost-cny", type=float, default=4)
    parser.add_argument("--seed", type=int, default=421)
    args = parser.parse_args(argv)
    if not 0 < args.cases <= 100 or not 0 < args.max_cost_cny <= 20:
        parser.error("invalid case count/budget")
    if args.out.resolve() == args.run.resolve():
        parser.error("Judge output must be separate from the Agent run")
    source = args.run / "predictions.jsonl"
    source_manifest = args.run / "run_manifest.json"
    source_hash, manifest_hash = sha256_file(source), sha256_file(source_manifest)
    run_meta = json.loads(source_manifest.read_text(encoding="utf-8"))
    if run_meta["status"] != "completed":
        parser.error("Complete Agent comparisons before offline judging")
    if "holdout" in {r["split"] for r in read_jsonl(source)}:
        parser.error("This calibration command accepts development runs only; freeze rubric before holdout")
    packet, identities = build_packet(read_jsonl(source), args.cases, args.seed)
    raw = dict(load_config(args.config)["models"]["deepseek"], max_tokens=800)
    cfg = RoleConfig.from_mapping("judge", raw)
    fingerprint = {"source_predictions_sha256": source_hash, "source_manifest_sha256": manifest_hash,
                   "packet_sha256": digest(packet), "prompt_sha256": digest(system_prompt()),
                   "judge_config": cfg.public_summary(), "max_cost_cny": args.max_cost_cny}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "run_manifest.json"
    rows = read_jsonl(args.out / "judgments.jsonl")
    if manifest_path.exists():
        if not args.resume:
            parser.error("Output exists; use --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in fingerprint.items()) or manifest.get("in_flight"):
            parser.error("Changed experiment or interrupted call; preserve outputs and inspect")
    else:
        manifest = {**fingerprint, "started_at_utc": datetime.now(timezone.utc).isoformat(), "budget_accounted_cny": 0}
        dump(args.out / "packet.json", {"packet_sha256": digest(packet), "rubric": RUBRIC_PATH.read_text(encoding="utf-8"), "items": packet})
        dump(args.out / "identity_key.json", identities)
        dump(manifest_path, manifest)
    budget = RunBudget(args.max_cost_cny, manifest["budget_accounted_cny"])
    client = ModelClient(cfg, budget=budget)
    seen = {(r["item_id"], r["order"]) for r in rows}
    stop, completed_now = False, 0
    for item in packet:
        new_item = False
        for order in ("AB", "BA"):
            if (item["item_id"], order) in seen:
                continue
            new_item = True
            context = {key: value for key, value in item.items() if key not in {"item_id", "scenario_id", "turn_id"}}
            if order == "BA":
                context["A"], context["B"] = context["B"], context["A"]
            manifest["in_flight"] = {"item_id": item["item_id"], "order": order}
            dump(manifest_path, manifest)
            result = client.call([{"role": "system", "content": system_prompt()}, {"role": "user", "content": json.dumps(context, ensure_ascii=False)}], require_json=True)
            record = {"item_id": item["item_id"], "order": order, "status": result.status, "model": asdict(result),
                      "source_predictions_sha256": source_hash, "input_sha256": digest(context)}
            if result.ok:
                try:
                    record["score"] = validate_score(json.loads(result.content))
                except ValueError as exc:
                    record.update(status="invalid_score", error=str(exc))
            append_jsonl(args.out / "judgments.jsonl", [record])
            rows.append(record)
            seen.add((item["item_id"], order))
            manifest.update(in_flight=None, budget_accounted_cny=budget.accounted_cny)
            dump(manifest_path, manifest)
            print(json.dumps({"item_id": item["item_id"], "order": order, "status": record["status"], "budget_cny": round(budget.accounted_cny, 6)}), flush=True)
            if result.status in {"budget_exceeded", "missing_key", "network_error", "http_error", "timeout"} or budget.stopped:
                stop = True
                break
        completed_now += int(new_item)
        if stop or (args.stop_after and completed_now >= args.stop_after):
            break
    if sha256_file(source) != source_hash or sha256_file(source_manifest) != manifest_hash:
        raise RuntimeError("Source Agent artifacts changed during offline judging")
    manifest.update(status="stopped" if stop else ("completed" if len(rows) == 2 * len(packet) else "paused"), updated_at_utc=datetime.now(timezone.utc).isoformat())
    dump(manifest_path, manifest)
    dump(args.out / "summary.json", judge_summary(rows))
    return int(stop)


if __name__ == "__main__":
    raise SystemExit(main())
