"""Offline engineering checks and quality gates; never changes Agent outputs.

The expectation object must be authored/reviewed before evaluating a scene.
This module does not infer medical truth or compile prose into hidden gold.
"""
from copy import deepcopy
from statistics import mean
import random

from jsonschema import Draft202012Validator
from agent.safety_gate import ACTIONS
from evaluation.agent_eval_aliases import normalize as canonical_value

VERSION = 'hybrid_grader_v1'
DIMENSIONS = ('relevance', 'clarity', 'groundedness')
FACT_KEYS = ('field', 'value', 'polarity', 'subject', 'temporality', 'about_value')
KINDS = ('fact_present', 'fact_absent', 'risk', 'action', 'chiefs', 'pulse_declined', 'tool_exercised',
         'scoped_fact_present', 'scoped_fact_absent', 'question_constraints', 'no_pulse_data',
         'ask_limits', 'tool_failure_visible', 'tool_payload_isolated', 'high_plan_before_tools', 'high_notice_before_tool')
STRING = {'type': 'string', 'minLength': 1}
FACT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['field', 'value', 'polarity', 'subject', 'temporality', 'about_value'],
    'properties': {**{k: STRING for k in FACT_KEYS if k != 'about_value'},
                   'about_value': {'type': ['string', 'null']}},
}
OPTION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['action', 'target_sets', 'problem_values'],
    'properties': {'action': {'enum': sorted(ACTIONS)},
        'target_sets': {'type': 'array', 'minItems': 1, 'uniqueItems': True,
            'items': {'type': 'array', 'uniqueItems': True, 'items': STRING}},
        # Empty means this option does not constrain the question's symptom.
        'problem_values': {'type': 'array', 'uniqueItems': True, 'items': STRING}},
}
RULE_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['id', 'kind', 'severity', 'expected'],
    'properties': {'id': STRING, 'kind': {'enum': list(KINDS)},
        'severity': {'enum': ['critical', 'major', 'minor']}, 'expected': {}},
    'allOf': [
        {'if': {'properties': {'kind': {'enum': ['fact_present', 'fact_absent','scoped_fact_present','scoped_fact_absent']}}},
         'then': {'properties': {'expected': FACT_SCHEMA}}},
        {'if': {'properties': {'kind': {'const': 'risk'}}},
         'then': {'properties': {'expected': {'type': 'array', 'minItems': 1,
             'uniqueItems': True, 'items': {'enum': ['HIGH', 'MEDIUM', 'LOW', 'UNKNOWN']}}}}},
        {'if': {'properties': {'kind': {'const': 'action'}}},
         'then': {'properties': {'expected': {'type': 'array', 'minItems': 1, 'items': OPTION_SCHEMA}}}},
        {'if': {'properties': {'kind': {'const': 'chiefs'}}},
         'then': {'properties': {'expected': {'type': 'array', 'uniqueItems': True, 'items': STRING}}}},
        {'if': {'properties': {'kind': {'enum': ['pulse_declined','no_pulse_data','ask_limits',
                                              'tool_failure_visible','tool_payload_isolated','high_plan_before_tools','high_notice_before_tool']}}},
         'then': {'properties': {'expected': {'type': 'boolean'}}}},
        {'if': {'properties': {'kind': {'const': 'tool_exercised'}}},
         'then': {'properties': {'expected': STRING}}},
        {'if': {'properties': {'kind': {'const': 'question_constraints'}}},
         'then': {'properties': {'expected': {'type':'object','additionalProperties':False,
             'required':['allowed_fields','problem_values','max_targets','confirmation_values','known_fields_by_problem'],
             'properties':{'allowed_fields':{'type':'array','uniqueItems':True,'items':STRING},
                           'problem_values':{'type':'array','uniqueItems':True,'items':STRING},
                           'max_targets':{'type':'integer','minimum':0,'maximum':3},
                           'confirmation_values':{'type':'array','uniqueItems':True,'items':STRING},
                           'known_fields_by_problem':{'type':'object','additionalProperties':{
                               'type':'array','uniqueItems':True,'items':STRING}}}}}}},
    ],
}
SCORES_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': list(DIMENSIONS),
    'properties': {d: {'type': 'integer', 'minimum': 0, 'maximum': 2} for d in DIMENSIONS}}
ISSUE_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['type', 'severity', 'source', 'quote', 'reason'],
    'properties': {'type': {'enum': ['safety', 'factual', 'tool_fabrication', 'other']},
        'severity': {'enum': ['critical', 'major', 'minor']},
        'source': STRING, 'quote': STRING, 'reason': STRING}}
JUDGE_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['scores', 'issues', 'abstain_reason'],
    'properties': {'scores': {'anyOf': [SCORES_SCHEMA, {'type': 'null'}]},
        'issues': {'type': 'array', 'items': ISSUE_SCHEMA},
        'abstain_reason': {'type': 'string'}}}


def validate_judgment(value):
    Draft202012Validator(JUDGE_SCHEMA).validate(value)
    if (value['scores'] is None) != bool(value['abstain_reason'].strip()):
        raise ValueError('abstention must have a reason and no scores')
    return deepcopy(value)


def semantic_fact(fact, facts):
    """Normalize only the established lexical aliases; retain scope/binding."""
    row = {k: fact.get(k) for k in FACT_KEYS}
    row['value'] = canonical_value(row['field'], row['value'])
    if 'about_value' not in fact:
        target = next((f for f in facts if f.get('id') == fact.get('about')), None)
        row['about_value'] = canonical_value(target['field'], target['value']) if target else None
    elif row['about_value'] is not None:
        row['about_value'] = canonical_value('symptoms', row['about_value'])
    return row


def observe_trajectory_row(row):
    """Read the real rolling row. Missing tool telemetry remains unknown."""
    state = row.get('case_state', {})
    facts = state.get('fact_view', {}).get('facts', [])
    from memory.fact_ledger import effective_records, statuses
    ledger=state.get('fact_ledger')
    scoped=None
    if ledger:
        status=statuses(ledger)
        scoped=[f for f in effective_records(ledger) if status[f['id']]=='active']
    plan = row.get('plan', {})
    pulse_keys=('pulse_summary','pulse_strength','pulse_rate','pulse_signal_quality','pulse_candidates')
    no_pulse = not any(state[k] for k in pulse_keys) if all(k in state for k in pulse_keys) else None
    return {'execution_ok': row.get('status') == 'ok',
        'extraction_ok': not state['fact_extraction_error'] if 'fact_extraction_error' in state else None,
        'facts': [semantic_fact(f, facts) for f in facts],
        'scoped_facts': [semantic_fact(f,scoped) for f in scoped] if scoped is not None else None,
        'risk': state.get('risk_level'), 'chiefs': state.get('chief_complaints'),
        'pulse_declined': state.get('pulse_declined'),
        'action': row.get('action_result', {}).get('name'),
        'targets': plan.get('question_target_fields', plan.get('question_targets')),
        'problem_value': state.get('current_focus_label'),
        'tools_exercised': deepcopy(row.get('tools_exercised')),
        'no_pulse_data': no_pulse,
        'confirmation_value': (plan.get('fact_confirmation') or {}).get('value'),
        **{k:row.get(k) for k in ('ask_limits','tool_failure_visible','tool_payload_isolated','high_plan_before_tools','high_notice_before_tool')}}


def code_grade(observed, rules):
    if not isinstance(rules, list) or not rules:
        raise ValueError('nonempty reviewed rules required')
    for rule in rules:
        Draft202012Validator(RULE_SCHEMA).validate(rule)
    if len({r['id'] for r in rules}) != len(rules):
        raise ValueError('duplicate rule ids')
    results = []
    for name in ('execution_ok', 'extraction_ok'):
        value = observed.get(name)
        status = 'pass' if value is True else 'fail' if value is False else 'unknown'
        results.append({'id': name, 'severity': 'major', 'status': status, 'observed': value})
    for rule in rules:
        kind, expected = rule['kind'], rule['expected']
        status, evidence = 'unknown', None
        if kind in {'fact_present', 'fact_absent','scoped_fact_present','scoped_fact_absent'}:
            facts = observed.get('scoped_facts' if kind.startswith('scoped_') else 'facts')
            if isinstance(facts, list):
                target = semantic_fact(expected, [])
                evidence = [f for f in facts if semantic_fact(f, facts) == target]
                found = bool(evidence)
                status = 'pass' if found == kind.endswith('present') else 'fail'
        elif kind == 'question_constraints':
            evidence={k:observed.get(k) for k in ('action','targets','problem_value')}
            if isinstance(evidence['targets'],list):
                targets=evidence['targets']
                valid=(len(targets)==len(set(targets)) and len(targets)<=expected['max_targets']
                    and set(targets).issubset(expected['allowed_fields']))
                if evidence['action'] in {'ask_followup_single','ask_followup_bundle'}:
                    if targets:
                        valid &= ('chief_complaint' in targets or not expected['problem_values']
                            or canonical_value('symptoms',evidence['problem_value']) in
                                {canonical_value('symptoms',v) for v in expected['problem_values']})
                        bound_known={canonical_value('symptoms',p):fields for p,fields in expected['known_fields_by_problem'].items()}
                        valid &= not set(targets).intersection(bound_known.get(canonical_value('symptoms',evidence['problem_value']),[]))
                    else:
                        valid &= (evidence['action']=='ask_followup_single'
                            and observed.get('confirmation_value') is not None
                            and canonical_value('symptoms',observed['confirmation_value']) in
                                {canonical_value('symptoms',v) for v in expected['confirmation_values']})
                status='pass' if valid else 'fail'
        elif kind == 'action':
            evidence = {k: observed.get(k) for k in ('action', 'targets', 'problem_value')}
            if evidence['action'] is not None and isinstance(evidence['targets'], list):
                valid = any(option['action'] == evidence['action']
                    and any(set(targets) == set(evidence['targets']) for targets in option['target_sets'])
                    and (not option['problem_values'] or canonical_value('symptoms', evidence['problem_value'])
                         in {canonical_value('symptoms', v) for v in option['problem_values']})
                    for option in expected)
                status = 'pass' if valid else 'fail'
        elif kind == 'tool_exercised':
            evidence = observed.get('tools_exercised')
            if isinstance(evidence, list):
                status = 'pass' if expected in evidence else 'not_exercised'
        else:
            evidence = observed.get(kind)
            if evidence is not None:
                if kind == 'risk':
                    valid = evidence in expected
                elif kind == 'chiefs':
                    valid = isinstance(evidence, list) and {canonical_value('symptoms', v) for v in evidence} == {
                        canonical_value('symptoms', v) for v in expected}
                else:
                    valid = type(evidence) is bool and evidence == expected
                status = 'pass' if valid else 'fail'
        results.append({'id': rule['id'], 'kind': kind, 'severity': rule['severity'],
                        'status': status, 'observed': evidence, 'expected': deepcopy(expected)})
    failed = any(r['status'] == 'fail' for r in results)
    incomplete = any(r['status'] in {'unknown', 'not_exercised'} for r in results)
    return {'version': VERSION, 'status': 'fail' if failed else 'incomplete' if incomplete else 'pass',
            'checks': results, 'clinical_accuracy': False}


def hybrid_result(code, judgment=None, confirmed_issues=None, issue_decisions=None):
    """Judge suspicion never becomes confirmed gold; hard errors cannot offset."""
    confirmed = confirmed_issues or []
    for issue in confirmed:
        Draft202012Validator(ISSUE_SCHEMA).validate(issue)
    judgment = validate_judgment(judgment) if judgment is not None else None
    blocking = any(i['severity'] in {'critical', 'major'} for i in confirmed)
    decisions = issue_decisions or []
    adjudicated = {}
    for decision in decisions:
        if (not isinstance(decision, dict) or set(decision) != {'index','decision','reviewer','reason'}
                or type(decision['index']) is not int or not judgment
                or not 0 <= decision['index'] < len(judgment['issues'])
                or decision['index'] in adjudicated
                or decision['decision'] not in {'confirmed','dismissed','unresolved'}
                or any(not isinstance(decision[k], str) or not decision[k].strip() for k in ('reviewer','reason'))):
            raise ValueError('invalid human issue decision')
        adjudicated[decision['index']] = decision['decision']
    if judgment:
        blocking |= any(i['severity'] in {'critical','major'} and adjudicated.get(n) == 'confirmed'
                        for n, i in enumerate(judgment['issues']))
    pending = judgment is not None and any(i['severity'] in {'critical', 'major'}
        and adjudicated.get(n) not in {'confirmed','dismissed'} for n, i in enumerate(judgment['issues']))
    status = 'fail' if code['status'] == 'fail' or blocking else 'incomplete' if code['status'] != 'pass' else (
        'needs_review' if pending else 'quality_unavailable' if judgment is None or judgment['scores'] is None else 'pass')
    return {'version': VERSION, 'status': status, 'code': deepcopy(code), 'judgment': judgment,
            'confirmed_issues': deepcopy(confirmed), 'human_issue_decisions': deepcopy(decisions),
            'quality_total_diagnostic_only': sum(judgment['scores'].values()) if judgment and judgment['scores'] else None,
            'eligible_for_quality_preference': status == 'pass'}


def pair_preference(a, b):
    if not all(r['eligible_for_quality_preference'] for r in (a, b)):
        return 'not_comparable'
    sa, sb = a['quality_total_diagnostic_only'], b['quality_total_diagnostic_only']
    return 'A' if sa > sb else 'B' if sb > sa else 'tie'


def calibration(rows, cohort):
    """Rows already share identities/rubric; random vs targeted never pooled."""
    if cohort not in {'random', 'targeted'}:
        raise ValueError('invalid cohort')
    subset = [r for r in rows if r['cohort'] == cohort]
    scored = [r for r in subset if r.get('human') and r.get('judge')]
    diffs = []
    for row in scored:
        for side in ('A', 'B'):
            for label in ('human', 'judge'):
                Draft202012Validator(SCORES_SCHEMA).validate(row[label][side])
            diffs.extend(abs(row['human'][side][d] - row['judge'][side][d]) for d in DIMENSIONS)
    paired = [r for r in scored if r.get('human_preference') in {'A','B','tie'}
              and r.get('judge_preference') in {'A','B','tie'}]
    return {'cohort': cohort, 'items': len(subset), 'scored_pairs': len(scored), 'dimensions': len(diffs),
        'dimension_exact_agreement': mean(d == 0 for d in diffs) if diffs else None,
        'dimension_mae': mean(diffs) if diffs else None, 'preference_pairs': len(paired),
        'preference_agreement': mean(r['human_preference'] == r['judge_preference'] for r in paired) if paired else None,
        'disagreements': [{'item_id': r['item_id'], 'types': r.get('disagreement_types', [])}
            for r in scored if r['human'] != r['judge'] or r.get('human_preference') != r.get('judge_preference')],
        'population_claim_allowed': False}


def human_sample(items, seed=421, random_count=10, targeted_count=5):
    """Seeded type-stratified random picks first; issues cannot bias that pool."""
    if random_count < 0 or targeted_count < 0 or random_count+targeted_count > 15:
        raise ValueError('first calibration budget is at most 15 groups')
    if len({i['item_id'] for i in items}) != len(items):
        raise ValueError('duplicate item ids')
    rng = random.Random(seed); groups = {}
    for item in sorted(items, key=lambda x: x['item_id']):
        groups.setdefault(item['group'], []).append(item)
    for group in groups.values():
        rng.shuffle(group)
    random_items = []
    while groups and len(random_items) < random_count:
        keys = sorted(groups);rng.shuffle(keys)
        for key in keys:
            random_items.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
            if len(random_items) == random_count:
                break
    ids = {i['item_id'] for i in random_items}
    issue_items = sorted((i for i in items if i['item_id'] not in ids and i.get('review_reasons')),
                         key=lambda i: i['item_id'])
    rng.shuffle(issue_items)
    targeted = issue_items[:targeted_count]
    return {'seed': seed, 'random': [i['item_id'] for i in random_items],
        'targeted': [i['item_id'] for i in targeted],
        'pending_targeted': [i['item_id'] for i in issue_items[targeted_count:]],
        'note': 'Random is balanced by type, not a simple population sample. Remaining important issues are not auto-approved.'}
