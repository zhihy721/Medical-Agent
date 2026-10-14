"""Validate fictional evaluation scenarios before review, replay, or freezing.

Usage: python -m evaluation.validate_dataset --dataset evaluation/datasets/dev_v1.jsonl
Install optional evaluation requirements first: pip install -r requirements-eval.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_PATH = Path(__file__).parent / "datasets" / "schema_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scenarios(path: Path) -> list[dict]:
    scenarios: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                scenarios.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not scenarios:
        raise ValueError(f"{path}: dataset is empty")
    return scenarios


def validate_scenarios(scenarios: list[dict], *, require_reviewed: bool = False) -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            errors.append(f"row {index}: expected JSON object")
            continue
        sid = scenario.get("scenario_id", f"row-{index}")
        for err in sorted(validator.iter_errors(scenario), key=lambda item: str(item.path)):
            at = ".".join(map(str, err.path)) or "root"
            errors.append(f"{sid}:{at}: {err.message}")
        if sid in seen_ids:
            errors.append(f"{sid}: duplicate scenario_id")
        seen_ids.add(sid)
        status = scenario.get("review_status")
        if require_reviewed and status != "reviewed":
            errors.append(f"{sid}: frozen dataset requires review_status=reviewed")
        if status == "reviewed" and not (scenario.get("reviewed_by") and scenario.get("reviewed_at")):
            errors.append(f"{sid}: reviewed scenario needs reviewed_by and reviewed_at")
        if scenario.get("split") == "holdout" and status == "reviewed":
            for turn_index, turn in enumerate(scenario.get("turns", []), start=1):
                if not turn.get("gold_policy"):
                    errors.append(f"{sid}:turn {turn_index}: reviewed holdout needs gold_policy")
        for turn_index, turn in enumerate(scenario.get("turns", []), start=1):
            user = turn.get("user", "")
            mentions = (turn.get("gold_extract") or {}).get("mentions") or []
            for mention_index, mention in enumerate(mentions, start=1):
                if not isinstance(mention, dict):
                    continue
                start, end = mention.get("start"), mention.get("end")
                if mention.get("turn_id") != turn_index:
                    errors.append(f"{sid}:turn {turn_index}:mention {mention_index}: turn_id mismatch")
                if isinstance(start, int) and isinstance(end, int) and isinstance(user, str):
                    if start >= end or user[start:end] != mention.get("evidence"):
                        errors.append(f"{sid}:turn {turn_index}:mention {mention_index}: evidence span mismatch")
            policy = turn.get("gold_policy") or {}
            if policy.get("risk") == "HIGH" and policy.get("allowed_actions") and "risk_escalation" not in policy["allowed_actions"]:
                errors.append(f"{sid}:turn {turn_index}: HIGH gold must allow risk_escalation")
    return errors


def cross_split_errors(named_sets: dict[str, list[dict]]) -> list[str]:
    """Reject shared IDs, families, and exact texts across datasets."""
    errors: list[str] = []
    seen_ids: dict[str, str] = {}
    seen_families: dict[str, str] = {}
    seen_texts: dict[str, str] = {}
    for name, scenarios in named_sets.items():
        for scenario in scenarios:
            sid = scenario.get("scenario_id", "")
            family = scenario.get("family_id", "")
            for value, seen, label in ((sid, seen_ids, "scenario_id"), (family, seen_families, "family_id")):
                if value and value in seen and seen[value] != name:
                    errors.append(f"{name}:{sid}: {label} shared with {seen[value]}: {value}")
                elif value:
                    seen[value] = name
            for turn in scenario.get("turns", []):
                normalized = "".join(str(turn.get("user", "")).split())
                if normalized and normalized in seen_texts and seen_texts[normalized] != name:
                    errors.append(f"{name}:{sid}: exact normalized user text shared with {seen_texts[normalized]}")
                elif normalized:
                    seen_texts[normalized] = name
    return errors


def cross_split_similarity(named_sets: dict[str, list[dict]], threshold: float = 0.8) -> list[dict]:
    """Surface near-duplicate utterances for human review; never auto-label leakage."""
    utterances = []
    for name, scenarios in named_sets.items():
        for scenario in scenarios:
            for turn_index, turn in enumerate(scenario.get("turns", []), start=1):
                normalized = re.sub(r"[^\w]", "", turn.get("user", "")).replace("想咨询一下", "")
                utterances.append((name, scenario.get("scenario_id"), turn_index, normalized, turn.get("user", "")))
    matches = []
    for left_index, left in enumerate(utterances):
        for right in utterances[left_index + 1:]:
            if left[0] == right[0] or not left[3] or not right[3]:
                continue
            score = SequenceMatcher(None, left[3], right[3]).ratio()
            if score >= threshold:
                matches.append({"left_set": left[0], "left_id": left[1], "left_turn": left[2], "left_text": left[4], "right_set": right[0], "right_id": right[1], "right_turn": right[2], "right_text": right[4], "similarity": round(score, 3)})
    return sorted(matches, key=lambda item: item["similarity"], reverse=True)


def dataset_summary(scenarios: list[dict]) -> dict:
    return {
        "scenarios": len(scenarios),
        "turns": sum(len(item.get("turns", [])) for item in scenarios),
        "statuses": dict(Counter(item.get("review_status", "missing") for item in scenarios)),
        "splits": dict(Counter(item.get("split", "missing") for item in scenarios)),
        "families": len({item.get("family_id") for item in scenarios}),
        "tags": dict(sorted(Counter(tag for item in scenarios for tag in item.get("tags", [])).items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--compare", action="append", type=Path, default=[])
    parser.add_argument("--policy", type=Path, default=Path("evaluation/policy_v1.md"))
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--similarity-report", type=Path)
    args = parser.parse_args(argv)
    if not args.policy.exists():
        parser.error(f"policy file missing: {args.policy}")
    named = {str(args.dataset): load_scenarios(args.dataset)}
    for path in args.compare:
        named[str(path)] = load_scenarios(path)
    errors = []
    for name, scenarios in named.items():
        errors.extend(f"{name}: {error}" for error in validate_scenarios(scenarios, require_reviewed=args.require_reviewed and name == str(args.dataset)))
    errors.extend(cross_split_errors(named))
    similar = cross_split_similarity(named) if len(named) > 1 else []
    if args.similarity_report:
        args.similarity_report.parent.mkdir(parents=True, exist_ok=True)
        with args.similarity_report.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["left_set", "left_id", "left_turn", "left_text", "right_set", "right_id", "right_turn", "right_text", "similarity"])
            writer.writeheader()
            writer.writerows(similar)
    print(json.dumps({"dataset_sha256": sha256_file(args.dataset), "policy_sha256": sha256_file(args.policy), "summary": dataset_summary(named[str(args.dataset)]), "near_duplicate_pairs_for_review": len(similar), "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
