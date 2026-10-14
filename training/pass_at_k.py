"""Offline extraction oracle pass@k; never a production response selector."""
import argparse
import json
from math import prod
from pathlib import Path
from llm.extraction import validate_delta


def estimate_pass_at_k(n, c, k):
    if any(type(x) is not int for x in (n, c, k)) or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('require integers 0 <= c <= n and 1 <= k <= n')
    if n - c < k:
        return 1.0
    return 1.0 - prod((n - c - i) / (n - i) for i in range(k))


ORACLE_VERSION = 'legacy_mentions_evidence_v2'


def strict_success(prediction, gold, input_data, acceptable_targets=None):
    """Joint fact/evidence match against gold or explicitly reviewed alternatives.

    Legal offsets alone do not establish evidence support. Alternatives are whole
    annotated targets, registered before sampling, never inferred from predictions.
    Invalid gold is a dataset error; invalid predictions are failed attempts.
    """
    validate_delta(gold, input_data)
    def facts(delta):
        return sorted((m['field'], m['value'], m['polarity'], m['operation']) for m in delta['mentions'])
    alternatives = acceptable_targets if acceptable_targets is not None else []
    if not isinstance(alternatives, list):
        raise ValueError('acceptable_targets must be a list of reviewed complete targets')
    for alternative in alternatives:
        validate_delta(alternative, input_data)
        if facts(alternative) != facts(gold):
            raise ValueError('alternative evidence cannot change target facts or operations')
    try:
        validate_delta(prediction, input_data)
    except (ValueError, TypeError, KeyError):
        return False
    def grounded_facts(delta):
        return sorted((m['field'], m['value'], m['polarity'], m['operation'],
                       m['evidence'], m['start'], m['end'], m['source_turn'])
                      for m in delta['mentions'])
    actual = grounded_facts(prediction)
    return any(actual == grounded_facts(target) for target in [gold, *alternatives])


def summarize(rows, ks=(1, 2, 4)):
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError('positive integer k required')
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique sample_id rows required')
    per_case = []
    for row in rows:
        attempts = row['attempts']
        n = len(attempts)
        if n < max(ks):
            raise ValueError('every case needs n >= max(k), including failed attempts')
        # Failure stays in the denominator; no silent resampling or survivor filtering.
        validate_delta(row['target'], row['input'])
        # Validate alternative gold even when every provider request failed.
        alternatives = row.get('acceptable_targets', [])
        strict_success(row['target'], row['target'], row['input'], alternatives)
        c = sum(a.get('status') == 'ok' and strict_success(a.get('prediction'), row['target'], row['input'], alternatives) for a in attempts)
        per_case.append({'sample_id': row['sample_id'], 'n': n, 'c': c,
                         'pass_at_k': {str(k): estimate_pass_at_k(n, c, k) for k in ks}})
    return {'oracle_version': ORACLE_VERSION, 'cases': len(rows), 'per_case': per_case,
            'macro_pass_at_k': {str(k): sum(r['pass_at_k'][str(k)] for r in per_case) / len(rows) for k in ks},
            'note': 'Oracle metric; requires independent draws from fixed sampling settings. Not deployment selection accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--k', type=int, nargs='+', default=[1, 2, 4])
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = summarize(rows, args.k)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
