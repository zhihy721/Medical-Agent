"""Versioned contract gate for generating formal teacher candidates in batches."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from llm.fact_contract import PROMPT
from training.family_registry import audit
from training.teacher_data import read_rows, sha

FILES = ('docs/SEMANTIC_FIELD_POLICY_V1.md', 'llm/fact_contract.py',
         'llm/fact_normalization.py', 'evaluation/fact_alias_metrics.py',
         'evaluation/fact_metrics.py', 'training/family_registry.py',
         'training/formal_data_plan_v1.json', 'evaluation/datasets/schema_v1.json',
         'training/formal_batch_gate.py')


def make_lock(seed_dir):
    root = Path(seed_dir)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    records, registry = root/'records.jsonl', root/'registry.jsonl'
    if manifest['records_sha256'] != sha(records) or manifest['registry_sha256'] != sha(registry):
        raise ValueError('reviewed seed provenance mismatch')
    if manifest['prompt_sha256'] != hashlib.sha256(PROMPT.encode()).hexdigest():
        raise ValueError('seed prompt differs from current contract')
    rows = read_rows(records)
    if len(rows) != 20 or Counter(r['split'] for r in rows) != {'train':10,'validation':10}:
        raise ValueError('first 20 reviewed rows required')
    if any(r.get('review_status') != 'reviewed' for r in rows):
        raise ValueError('unreviewed seed')
    groups = Counter((r['split'],r['primary_stratum']) for r in rows)
    if any(groups[split, stratum] < 1 for split in ('train','validation') for stratum in (
        'normalization','polarity_scope','uncertainty','attribute_binding',
        'correction_and_resolution','source_and_temporality','no_fact_and_injection','tcm_fields')):
        raise ValueError('first pilot lacks stratum coverage')
    registry_result = audit(read_rows(registry))
    if not registry_result['ready_for_generation']:
        raise ValueError('registry conflict')
    return dict(status='locked_for_candidate_generation_only', training_release_allowed=False,
                seed_records_sha256=sha(records), seed_registry_sha256=sha(registry),
                prompt_sha256=manifest['prompt_sha256'], files_sha256={f:sha(f) for f in FILES},
                reviewed_seed_rows=len(rows), next_batch_limit=80,
                note='Allows up to 80 new draft teacher candidates. Human review, later batch gates and SFT release are separate.')


def verify_lock(path):
    lock = json.loads(Path(path).read_text(encoding='utf-8'))
    if lock != make_lock('training/data/formal_seed_reviewed_v4'):
        raise ValueError('contract lock stale or tampered')
    return lock


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    lock = make_lock(args.seed_dir)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(lock,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'status':lock['status'],'reviewed_seed_rows':20,
                      'next_batch_limit':80,'training_release_allowed':False}))


if __name__ == '__main__':main()
