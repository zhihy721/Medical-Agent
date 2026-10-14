"""Opt-in redundant-assertion equivalence; original V2 oracle is unchanged.

This is a scoring view, not a clinical entailment checker or a label rewrite.
The ledger already retains repeated sources and aliases their object identity.
"""
from copy import deepcopy
import json
import re

from evaluation.runtime_fact_metrics import score_v2, _score, _identity
from llm.fact_contract_runtime import position_delta, OBSERVATION_FIELDS
from llm.fact_representation import normalize_representation
from llm.deferred_scope import evidence_scope, is_deferred_ambiguous

VERSION = 'explicit_known_reassertion_view_v1'
FIELDS = {'symptoms', 'red_flags'} | OBSERVATION_FIELDS
BLOCK = re.compile(r'如果|假设|要是|将来|以后可能|还没发生|不确定|不知道|说不准|可能|也许|[？?“”「」]|'
                   r'母亲|父亲|妈妈|爸爸|别人|医生说|忽略|只输出|只回答|清空|删除|说错|记错|缓解|已经好了')


def project_reassertions(delta, inp):
    position_delta(delta, inp)  # No invalid evidence/transaction is erased.
    output = deepcopy(delta)
    aliases, changes = {}, []
    for fact in output['facts']:
        if (fact['operation'] != 'assert' or fact['replaces'] is not None or fact['about'] is not None
                or fact['field'] not in FIELDS or fact['subject'] != 'patient'
                or fact['temporality'] != 'current' or fact['polarity'] not in {'present', 'negated'}):
            continue
        start = inp['current_turn'].find(fact['evidence'])
        scope = (re.split(r'[，,。；;\n]', inp['current_turn'][:start])[-1] + fact['evidence']
                 + re.split(r'[，,。；;\n]', inp['current_turn'][start+len(fact['evidence']):])[0])
        if BLOCK.search(scope) or is_deferred_ambiguous(inp['current_turn'], fact['evidence'], modern=True):
            continue
        matches = [k for k in inp['known_facts'] if k['status'] == 'active' and k['about'] is None
                   and all(k[key] == fact[key] for key in ('field', 'value', 'polarity', 'subject', 'temporality'))]
        if len(matches) != 1:
            continue
        aliases[fact['id']] = matches[0]['id']
        changes.append(dict(kind='redundant_explicit_assertion', original=deepcopy(fact),
                            known_id=matches[0]['id'], version=VERSION))
    output['facts'] = [f for f in output['facts'] if f['id'] not in aliases]
    for fact in output['facts']:
        if fact['about'] in aliases:
            before = fact['about']
            fact['about'] = aliases[before]
            changes.append(dict(kind='same_object_about', fact_id=fact['id'], before=before,
                                after=fact['about'], version=VERSION))
    position_delta(output, inp)
    return output, changes


def score_reassertions(prediction, gold, inp):
    original = score_v2(prediction, gold, inp)
    canonical_gold, _ = normalize_representation(gold, inp)
    gold_view, gold_changes = project_reassertions(canonical_gold, inp)
    changes = []
    try:
        pred = json.loads(prediction) if isinstance(prediction, str) else prediction
        canonical_pred, _ = normalize_representation(pred, inp)
        pred_view, changes = project_reassertions(canonical_pred, inp)
        # Field concept projection follows validation, as in the existing oracle.
        metrics = _score(_identity(pred_view), _identity(gold_view), inp, validated=True)
    except (ValueError, KeyError, TypeError) as exc:
        metrics = deepcopy(original['concept_equivalent'])
        metrics['error'] = str(exc)
    return dict(version=VERSION, original=original, explicit_reassertion_equivalent=metrics,
                prediction_changes=changes, gold_changes=gold_changes,
                scope='finite_current_patient_active_known_object_equivalence_not_entailment_or_relabeling')
