"""Narrow, auditable handling of deferred discussion, without inferring symptoms.

Only facts already emitted by the extractor are adjusted. Missing facts remain
a model-quality failure. The locked V1 contract and historical outputs stay intact.
"""
from copy import deepcopy
import re

VERSION = 'deferred_discussion_v1'
DEFER = re.compile(r'以后再|回头再|先搁置|到时候再|以后的事')
CONDITION = re.compile(r'如果|要是|假设|若|还没发生|未发生|以后.{0,10}(?:有了|出现|发生)|将来|日后|以后可能|假设以后')
EXPLICIT = re.compile(r'确实|已经|有过|一直|现在有|目前有|没有|没觉得|不疼|不痛')


def evidence_scope(text, evidence):
    """Use the literal evidence or its comma-bounded clause, never other clauses."""
    if DEFER.search(evidence):
        return evidence
    start = text.find(evidence)
    if start < 0:
        return evidence
    left = re.split(r'[，,。！？；;\n]', text[:start])[-1]
    right = re.split(r'[，,。！？；;\n]', text[start+len(evidence):])[0]
    return left + evidence + right


def is_deferred_ambiguous(text, evidence, *, modern=False):
    scope = evidence_scope(text, evidence)
    deferred = bool(DEFER.search(scope)) or (modern and bool(re.search(r'先放一放|先不谈|先不讨论|暂时不谈',scope)))
    return bool(deferred and not CONDITION.search(scope) and not EXPLICIT.search(scope))


def explicitly_current(fact, scope):
    from llm.fact_normalization import canonical_value,SYMPTOM_ALIASES
    concept = canonical_value(fact['field'], fact['value'])
    terms = {concept} | {k for k,v in SYMPTOM_ALIASES.items() if v == concept}
    return any(re.search(r'(?:现在|目前|眼下|当前)(?:就是|主要是|主要有|是|有)?\s*'+re.escape(term),scope)
               for term in terms) or bool(re.search(
        r'(?:耳朵|头|颈|肩|胸|肚子|腹部|膝盖|腿|腰|背)(?:现在|目前|眼下)(?:还|也)?(?:疼|痛|不舒服)',scope))


def adjust_delta(inp, delta):
    from llm.fact_contract_runtime import position_delta
    from llm.fact_normalization import normalize_facts
    normalized, normalized_input, _ = normalize_facts(delta, inp)
    position_delta(normalized, normalized_input)
    result = deepcopy(delta)
    changes = []
    for fact in result.get('facts', []):
        if (fact['field'] not in {'symptoms', 'red_flags'} or fact['subject'] != 'patient'
                or fact['operation'] != 'assert'
                or not is_deferred_ambiguous(inp['current_turn'], fact['evidence'])
                or explicitly_current(fact, evidence_scope(inp['current_turn'], fact['evidence']))):
            continue
        # Deferral is not a retraction of an already established patient fact.
        canonical = next(f for f in normalized['facts'] if f['id'] == fact['id'])
        if any(k['field'] == canonical['field'] and k['value'] == canonical['value']
               and k['subject'] == 'patient' and k['polarity'] == 'present'
               and k['status'] == 'active' and k['temporality'] == 'current'
               for k in normalized_input['known_facts']):
            continue
        before = {k: fact[k] for k in ('polarity', 'temporality', 'evidence')}
        fact.update(polarity='uncertain', temporality='current')
        fact['evidence'] = evidence_scope(inp['current_turn'], fact['evidence'])
        if before != {k: fact[k] for k in before}:
            changes.append(dict(fact_id=fact['id'], before=before,
                                after={k: fact[k] for k in before}, policy_version=VERSION))
    return result, changes


def pending_confirmation(case_state):
    counts = case_state.get('fact_confirmation_counts', {})
    return next((c for c in case_state.get('fact_clarifications', [])
                 if counts.get(c['fact_id'], 0) < 2), None)


def confirmation_plan(plan, candidate):
    result = deepcopy(plan)
    result.update(next_action='ask_followup_single', next_focus='fact_confirmation',
                  question_targets=[], question_target_fields=[],
                  fact_confirmation=candidate, pending_questions=[candidate['question']],
                  action_reason='症状是否实际发生尚不明确，先确认再完善属性。',
                  stop_condition='', goal_progress='等待症状存在性确认')
    if candidate.get('kind') == 'heart_interpretation':
        result.update(action_reason='心慌的具体含义尚不明确，保留原词并澄清感受。',
                      goal_progress='等待口语含义澄清')
    return result
