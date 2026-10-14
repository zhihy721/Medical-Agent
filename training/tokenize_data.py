"""Tokenize authored messages with loss only on the assistant JSON completion."""
import argparse
import json
from pathlib import Path

from llm.extraction import messages as legacy_messages
from llm.fact_contract import VERSION as FACT_VERSION, messages as fact_messages


def tokenize_record(tokenizer, record, max_length=None):
    contract = record['input'].get('contract_version')
    if contract is None:
        messages = legacy_messages
        max_length = 768 if max_length is None else max_length
    elif contract == FACT_VERSION:
        messages = fact_messages
        if max_length is None:
            raise ValueError('semantic facts require an explicit measured max_length')
    else:
        raise ValueError('unsupported contract version')
    if type(max_length) is not int or max_length < 1:
        raise ValueError('max_length must be a positive integer')
    prompt = messages(record["input"])
    full = messages(record["input"], record["target"])
    # Explicitly verify prefix alignment; do not guess where the response begins.
    prefix_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    full_ids = tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False)
    if any(not isinstance(ids, list) or any(type(token) is not int for token in ids) for ids in (prefix_ids, full_ids)):
        raise ValueError("tokenizer must return flat integer token IDs")
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("chat template prefix mismatch; inspect assistant boundary before training")
    if len(full_ids) > max_length:
        raise ValueError(f"sample {record['sample_id']} exceeds max_length: {len(full_ids)}; no truncation")
    if len(full_ids) <= len(prefix_ids):
        raise ValueError("empty assistant target")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, help="Required for semantic facts; legacy default is 768")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    data = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    result, errors = [], []
    for r in data:
        try:
            tokens = tokenize_record(tokenizer, r, args.max_length)
            result.append({"sample_id": r["sample_id"], "total_tokens": len(tokens["input_ids"]),
                           "target_tokens": sum(x != -100 for x in tokens["labels"])})
        except ValueError as exc:
            errors.append({"sample_id": r["sample_id"], "error": str(exc)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scope": "tokenization probe, not training", "records": result, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
