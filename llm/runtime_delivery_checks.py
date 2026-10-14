"""Delivery-only pre-dedup alarms; frozen V16 scores and facts stay unchanged."""
from llm.fact_transaction_guard import RESOLVED, BLOCK
from llm.deferred_scope import evidence_scope

VERSION='delivery_pre_dedup_checks_v1'


def check_discarded_transactions(inp, raw_delta, changes):
    discarded={c['fact_id'] for c in changes if c.get('kind')=='duplicate_alias'}
    for item in raw_delta['facts']:
        if item['id'] not in discarded or not (
                item['operation']=='assert' and item['field'] in {'symptoms','red_flags'}
                and item['subject']=='patient' and item['polarity']=='present'
                and item['temporality']=='current'):
            continue
        scope=evidence_scope(inp['current_turn'],item['evidence'])
        if RESOLVED.search(scope) and not BLOCK.search(scope):
            raise ValueError('dedup_must_not_hide_resolution_transaction')
