"""Versioned semantic-delta oracle; never repairs predictions or mutates history.

Scores one supplied input, not an end-to-end trajectory. Known IDs are fixed by
that input. Local IDs/order are irrelevant; their typed relations are not.
Evidence alternatives must be explicitly supplied as complete gold targets.
"""
from collections import Counter
import json

from llm.fact_contract import position_delta

ORACLE_VERSION = 'semantic_fact_relations_v1'
SEMANTICS = ('field', 'value', 'polarity', 'subject', 'temporality')


def signatures(delta, level):
    local = {f['id']: f for f in delta['facts']}

    def signature(f):
        result = tuple(f[k] for k in SEMANTICS)
        if level != 'semantics':
            ref = f['about']
            # Valid local targets are problems, never attributes: no cycles.
            binding = ('local', signature(local[ref])) if ref in local else ('known', ref)
            result += (binding,)
        if level in {'operations', 'evidence'}:
            result += (f['operation'], f['replaces'])
        if level == 'evidence':
            result += (f['evidence'],)
        return result

    return Counter(signature(f) for f in delta['facts'])


def score_delta(prediction, gold, inp, evidence_alternatives=()):
    """Cumulative exact metrics plus semantic multiset counts.

Invalid gold raises. Invalid model output is a failed attempt (never omitted).
Alternatives may change evidence only; each must pass the contract. Their
approval/provenance belongs in the dataset manifest, not this pure scorer.
"""
    targets = [gold, *evidence_alternatives]
    for target in targets:
        position_delta(target, inp)
        if signatures(target, 'operations') != signatures(gold, 'operations'):
            raise ValueError('evidence alternatives cannot change facts or relations')
    result = dict(oracle_version=ORACLE_VERSION, contract_valid=False,
                  semantics_exact=False, bindings_exact=False,
                  operations_exact=False, evidence_exact=False, strict_success=False,
                  semantic_counts=None, error='')
    try:
        pred = json.loads(prediction) if isinstance(prediction, str) else prediction
        position_delta(pred, inp)
    except (ValueError, TypeError, KeyError) as exc:
        result['error'] = str(exc)
        return result
    result['contract_valid'] = True
    p, g = signatures(pred, 'semantics'), signatures(gold, 'semantics')
    result['semantic_counts'] = {'matched': sum((p & g).values()),
                                 'predicted': sum(p.values()), 'gold': sum(g.values())}
    for key, level in [('semantics_exact', 'semantics'), ('bindings_exact', 'bindings'),
                       ('operations_exact', 'operations'), ('evidence_exact', 'evidence')]:
        result[key] = any(signatures(pred, level) == signatures(t, level) for t in targets)
    result['strict_success'] = result['evidence_exact']
    return result


def summarize(scores):
    """All attempts stay in the denominator, including invalid outputs."""
    rows = list(scores)
    keys = ('contract_valid', 'semantics_exact', 'bindings_exact',
            'operations_exact', 'evidence_exact', 'strict_success')
    return {'oracle_version': ORACLE_VERSION, 'attempts': len(rows),
            'passed': {k: sum(bool(r[k]) for r in rows) for k in keys},
            'rates': {k: sum(bool(r[k]) for r in rows) / len(rows) if rows else None for k in keys}}
