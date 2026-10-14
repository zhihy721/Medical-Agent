"""Alias-aware oracle, separate from the preserved literal v1 scorer."""
import json

from evaluation.fact_metrics import score_delta
from llm.fact_contract import position_delta
from llm.fact_normalization import VERSION, normalize_facts

ORACLE_VERSION = 'semantic_fact_relations_aliases_v3'


def score_delta_with_aliases(prediction, gold, inp, evidence_alternatives=()):
    normalized_gold, context, gold_changes = normalize_facts(gold, inp)
    alternatives = [normalize_facts(a, inp)[0] for a in evidence_alternatives]
    # Dataset errors still raise even if the model call failed.
    score_delta(normalized_gold, normalized_gold, context, alternatives)
    raw_valid, changes, raw_error = False, [], ''
    try:
        pred = json.loads(prediction) if isinstance(prediction, str) else prediction
        try:
            position_delta(pred, inp)
            raw_valid = True
        except (ValueError, TypeError, KeyError) as exc:
            raw_error = str(exc)
        normalized, _, changes = normalize_facts(pred, inp)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        normalized, raw_error = None, str(exc)
    result = score_delta(normalized, normalized_gold, context, alternatives)
    result.update(oracle_version=ORACLE_VERSION, normalization_version=VERSION,
                  raw_contract_valid=raw_valid, raw_contract_error=raw_error,
                  normalization_changes=changes, gold_normalization_changes=gold_changes)
    return result
