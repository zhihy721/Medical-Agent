"""Pure, append-only evidence ledger for the draft fact contract.

This module deliberately returns a structured view, not legacy flat slots. The
online adapter must handle disputed/resolved safety facts before consuming it.
"""
from copy import deepcopy
from memory import fact_ledger as legacy

from llm.fact_contract_runtime import (ATTRIBUTES, KNOWN_KEYS, VERSION, V2, OBSERVATION_FIELDS, build_input,
                               derived_fields, position_delta)


def empty_ledger(*, interpretation=False, contract_version=VERSION):
    if contract_version == VERSION:
        return legacy.empty_ledger(interpretation=interpretation)
    if contract_version not in {VERSION, V2}:
        raise ValueError('unsupported fact contract')
    ledger = {'contract_version': contract_version, 'turns': [], 'records': [], 'events': [], 'aliases': {}}
    from memory.fact_temporal import VERSION as TEMPORAL_VERSION
    ledger['temporal_policy_version'] = TEMPORAL_VERSION
    from memory.fact_identity import VERSION as IDENTITY_VERSION
    ledger['concept_identity_version'] = IDENTITY_VERSION
    if interpretation:
        from llm.fact_normalization import VERSION as ALIAS_VERSION
        from llm.fact_clarification import VERSION as INTERPRETATION_VERSION
        ledger.update(normalization_version=ALIAS_VERSION, interpretation_version=INTERPRETATION_VERSION)
    return ledger


def canonical_id(ledger, fact_id):
    aliases = ledger.get('aliases', {})
    seen = set()
    while fact_id in aliases:
        if fact_id in seen:
            raise ValueError('cyclic fact alias')
        seen.add(fact_id)
        fact_id = aliases[fact_id]
    return fact_id


def effective_records(ledger):
    state = statuses(ledger)
    rows = []
    for raw in ledger['records']:
        if canonical_id(ledger, raw['id']) != raw['id'] or state[raw['id']] == 'superseded':
            continue
        row = deepcopy(raw)
        row['about'] = canonical_id(ledger, row['about'])
        rows.append(row)
    return rows


def update_profile(ledger, fields, source='manual_profile'):
    """Explicit form edits replace their fields; prefill only seeds a new ledger."""
    allowed = {'age', 'sex', 'past_history', 'allergy_history', 'medication_history'}
    if source not in {'manual_profile', 'profile_prefill'} or set(fields) - allowed:
        raise ValueError('unsupported profile source or field')
    if source == 'profile_prefill' and ledger['records']:
        raise ValueError('prefill only allowed before facts exist')
    result = deepcopy(ledger)
    turn = len(result['turns'])
    state = statuses(result)
    for old in result['records']:
        if old['field'] in fields and state[old['id']] != 'superseded':
            result['events'].append({'fact_id': old['id'], 'status': 'superseded',
                                     'cause': source, 'turn_id': turn})
    for field, value in fields.items():
        values = value if isinstance(value, list) else [value]
        for item in dict.fromkeys(str(v).strip() for v in values if v):
            result['records'].append(dict(id=f'profile:{len(result["records"])+1}', field=field,
                value=item, polarity='present', evidence=item, subject='patient', temporality='current',
                operation='assert', about=None, replaces=None, source=source, source_turn=0,
                start=None, end=None))
    return result


def statuses(ledger):
    result = {record['id']: 'active' for record in ledger['records']}
    for event in ledger['events']:
        result[event['fact_id']] = event['status']
    return result


def next_input(ledger, text, last_question='', current_focus=None):
    if ledger['contract_version'] == VERSION:
        return legacy.next_input(ledger, text, last_question, current_focus)
    if ledger['contract_version'] not in {VERSION, V2}:
        raise ValueError('ledger contract mismatch')
    state = statuses(ledger)
    known = []
    for record in effective_records(ledger):
        if state[record['id']] in {'active', 'disputed'}:
            row = {**record, 'status': state[record['id']]}
            known.append({key: row[key] for key in KNOWN_KEYS})
    if ledger['contract_version'] == V2:
        from memory.fact_context import select_facts
        known, _ = select_facts(known, current_focus)
    return build_input(text, len(ledger['turns']) + 1, known, last_question, current_focus, contract_version=ledger['contract_version'])


def _conflicts(a, b, *, identity_version=None):
    from memory.fact_identity import same_concept
    same_field = a['field'] == b['field'] or same_concept(a,b,version=identity_version)
    if not same_field or (a['subject'], a['temporality'], a['about']) != (
            b['subject'], b['temporality'], b['about']):
        return False
    if a['field'] in ATTRIBUTES | {'age', 'sex', 'city'}:
        return (a['value'], a['polarity']) != (b['value'], b['polarity'])
    return a['value'] == b['value'] and a['polarity'] != b['polarity']


def apply_delta(ledger, inp, delta):
    if ledger['contract_version'] == VERSION:
        return legacy.apply_delta(ledger, inp, delta)
    """Atomic transition: validate everything first, return a new ledger.

    A conflicting assertion never silently erases older evidence. Only an explicit
    well-formed replacement supersedes a known record. Semantic support for that
    replacement is a model/gold quality obligation, not proven by this function.
    """
    expected = next_input(ledger, inp['current_turn'], inp['last_question'], inp['current_focus'])
    if expected != inp:
        raise ValueError('stale or foreign ledger context')
    from memory.fact_identity import VERSION as IDENTITY_VERSION
    identity_version = ledger.get('concept_identity_version')
    if identity_version not in {None, IDENTITY_VERSION}:
        raise ValueError('unsupported concept identity version')
    from llm.fact_normalization import VERSION as ALIAS_VERSION, normalize_facts
    from llm.fact_lexical_policy import ALIAS_VERSION as NEW_ALIAS_VERSION
    from llm.fact_projection_policy import ALIAS_VERSION as V5
    normalization_version = ledger.get('normalization_version')
    raw_delta = deepcopy(delta)
    changes = []
    if ledger.get('normalization_version'):
        if normalization_version not in {ALIAS_VERSION, NEW_ALIAS_VERSION, V5}:
            raise ValueError('unsupported normalization version')
        delta, normalized_input, changes = normalize_facts(delta, inp, version=normalization_version)
    else:
        normalized_input = inp
    records = position_delta(delta, normalized_input)
    result = deepcopy(ledger)
    audit = {'input': deepcopy(inp), 'raw_delta': raw_delta}
    if ledger.get('normalization_version'):
        audit.update(normalized_delta=deepcopy(delta), normalization_changes=changes,
                     normalization_version=normalization_version)
    result['turns'].append(audit)
    state = statuses(result)

    def transition(fact_id, status, cause):
        root = canonical_id(result, fact_id)
        for member in result['records']:
            if canonical_id(result, member['id']) == root and state.get(member['id']) != status:
                result['events'].append({'fact_id': member['id'], 'status': status,
                                         'cause': cause, 'turn_id': inp['turn_id']})
                state[member['id']] = status

    # Apply replacements as a transaction before checking conflicts, so order of
    # JSON facts cannot make a valid replacement itself appear contradictory.
    old_by_id = {r['id']: r for r in effective_records(result)}
    if result.get('temporal_policy_version'):
        from memory.fact_temporal import SUPPORTED_VERSIONS, prior_denials_ended
        if result['temporal_policy_version'] not in SUPPORTED_VERSIONS:
            raise ValueError('unsupported temporal policy version')
        lexical_aliases=None
        if normalization_version==V5:
            from llm.fact_projection_policy import BREATHING_WORDS
            from llm.fact_lexical_policy import PAIN_ALIASES,APPROVED_SYMPTOM_ALIASES
            lexical_aliases={**PAIN_ALIASES,**APPROVED_SYMPTOM_ALIASES,**BREATHING_WORDS}
        transitions = prior_denials_ended(inp, records, old_by_id.values(), state,
                                         policy_version=result['temporal_policy_version'],
                                         identity_version=identity_version,lexical_aliases=lexical_aliases)
        for update in transitions:
            if identity_version:
                update['concept_identity_version'] = identity_version
        for update in transitions:
            transition(update['prior_id'], 'past', update['reason'] + ':' + update['new_id'])
        result['turns'][-1]['temporal_transitions'] = transitions
    # A direct answer to a pending existence question refines an uncertain
    # candidate. It cannot retract an established positive or an unrelated fact.
    for record in records:
        if record['operation'] != 'assert' or record['polarity'] not in {'present', 'negated'}:
            continue
        for old in old_by_id.values():
            if (state[old['id']] == 'active' and old['polarity'] == 'uncertain'
                    and old['field'] in ({'symptoms', 'red_flags'} | (OBSERVATION_FIELDS if ledger['contract_version'] == V2 else set()))
                    and all(old[k] == record[k] for k in ('field', 'value', 'subject', 'temporality', 'about'))
                    and f"你现在有{old['value']}吗" in inp['last_question']):
                transition(old['id'], 'superseded', 'presence_answer:' + record['id'])
    reaffirmed = set()
    for record in records:
        old = old_by_id.get(record['replaces'])
        if (old and state[old['id']] == 'active' and record['operation'] == 'correct'
                and old['field'] in {'symptoms', 'red_flags'}
                and all(old[k] == record[k] for k in ('field', 'value', 'polarity', 'subject', 'temporality', 'about'))):
            reaffirmed.add(record['id'])
    replacements = {canonical_id(result, r['replaces']) for r in records
                    if r['replaces'] is not None and r['id'] not in reaffirmed}
    for record in records:
        if record['replaces'] and record['id'] not in reaffirmed:
            old = old_by_id[record['replaces']]
            # A targeted correction of a disputed slot settles that slot's
            # competing assertions, not other symptoms or unrelated attributes.
            if state[old['id']] == 'disputed':
                for peer in old_by_id.values():
                    if _conflicts(old, peer, identity_version=identity_version):
                        transition(peer['id'], 'superseded', record['id'])
                        replacements.add(peer['id'])
            transition(record['replaces'], 'superseded', record['id'])
    for old in result['records']:
        if canonical_id(result, old['about']) in replacements:
            transition(old['id'], 'superseded', 'problem_replaced')
    # Repeated assertions share a stable identity but keep every original record.
    # Establish problem aliases before normalizing their attribute references.
    aliases = result.setdefault('aliases', {})
    for record in sorted(records, key=lambda r: r['about'] is not None):
        about = canonical_id(result, record['about'])
        key = (record['field'], record['value'], record['polarity'], record['subject'], record['temporality'], about)
        candidates = result['records'] + [r for r in records if r is not record and r['id'] in state]
        peer = next((r for r in candidates if state.get(r['id']) in {'active', 'disputed'}
            and (r['field'], r['value'], r['polarity'], r['subject'], r['temporality'], canonical_id(result, r['about'])) == key), None)
        if peer:
            aliases[record['id']] = canonical_id(result, peer['id'])
        state[record['id']] = state.get(peer['id'], 'active') if peer else 'active'
    for record in records:
        if canonical_id(result, record['about']) in replacements:
            raise ValueError('attribute points to a problem replaced in the same turn')
        result['records'].append(record)
        if state[record['id']] == 'disputed':
            result['events'].append({'fact_id': record['id'], 'status': 'disputed',
                                     'cause': 'repeated_disputed_fact', 'turn_id': inp['turn_id']})
    active = [r for r in effective_records(result) if state[r['id']] in {'active','disputed'}]
    for index, a in enumerate(active):
        for b in active[index + 1:]:
            if _conflicts(a, b, identity_version=identity_version):
                transition(a['id'], 'disputed', b['id'])
                transition(b['id'], 'disputed', a['id'])
    # Do not auto-resolve existing disputes merely because a third fact arrived.
    if ledger.get('interpretation_version'):
        result['turns'][-1]['interpretations'] = interpretation_annotations(result)
    return result


def interpretation_annotations(ledger):
    """Derived annotations with stable fact IDs; old ledgers are not migrated."""
    from llm.fact_clarification import VERSION as INTERPRETATION_VERSION, breathing_clarifications
    if not ledger.get('interpretation_version'):
        return []
    if ledger['interpretation_version'] != INTERPRETATION_VERSION:
        raise ValueError('unsupported interpretation policy version')
    live = effective_records(ledger)
    result = breathing_clarifications(live)
    by_id = {r['id']: r for r in ledger['records']}
    # An episode truly happened then resolved: preserve its candidate safety
    # signal even when the new evidence no longer repeats the colloquial words.
    for record in live:
        if record['temporality'] != 'resolved' or record['polarity'] != 'present' or record['subject'] != 'patient':
            continue
        ancestor = by_id.get(record.get('replaces'))
        if ancestor and breathing_clarifications([ancestor]):
            annotation = breathing_clarifications([ancestor])[0]
            result.append({**annotation, 'fact_id': record['id'],
                           'interpretation_status': 'resolved_episode', 'question': ''})
    return result


def project(ledger):
    """Audit/consumer view preserving bindings, polarity and temporal scope.

    No patient chief intent is inferred. Historical/resolved red flags remain in
    safety_history for the runtime's explicit clearance policy, never flattened
    into a claim that the episode did not occur.
    """
    if ledger['contract_version'] == VERSION:
        return legacy.project(ledger)
    state = statuses(ledger)
    active = [r for r in effective_records(ledger) if state[r['id']] == 'active']
    disputed = [r for r in effective_records(ledger) if state[r['id']] == 'disputed']
    current = [r for r in active if r['subject'] == 'patient' and r['temporality'] == 'current']
    facts = []
    for record in current:
        facts.append({**record, 'derived_from': None})
        facts.extend({**record, 'field': field, 'value': value, 'derived_from': record['id']}
                     for field, value in derived_fields(record))
    problems = []
    for record in current:
        if record['field'] in {'symptoms', 'red_flags'} and record['polarity'] == 'present':
            problems.append({'id': record['id'], 'value': record['value'],
                             'attributes': [r for r in current if r['about'] == record['id']]})
    observations = [{'id': r['id'], 'field': r['field'], 'value': r['value'],
                     'attributes': [a for a in current if a['about'] == r['id']]}
                    for r in current if r['field'] in OBSERVATION_FIELDS and r['polarity'] == 'present']
    scoped_slots = {}
    if ledger['contract_version'] == V2:
        from memory.runtime_fact_projection import derive_identical_concept_fields
        scoped, trace = derive_identical_concept_fields(active)
        scoped_slots = dict(scoped_facts=scoped, scoped_projection_trace=trace)
    return {**scoped_slots, 'contract_version': ledger['contract_version'], 'facts': facts, 'problems': problems,
            'previous_current_facts': [r for r in effective_records(ledger) if state[r['id']]=='past'],
            'observations': observations,
            'unbound_attributes': [r for r in current if r['field'] in ATTRIBUTES and r['about'] is None],
            'disputed': disputed,
            'safety_history': [r for r in active if r['field'] == 'red_flags'
                               and r['subject'] == 'patient' and r['polarity'] == 'present'
                               and r['temporality'] in {'historical', 'resolved'}],
            'chief_complaint_candidates': sorted({r['value'] for r in facts
                if r['field'] == 'symptoms' and r['polarity'] == 'present'}),
            'intent_status': 'not_evaluated'}


def risk_input(ledger):
    """Project facts for engineering risk policy, never filter by chief/focus.

    Explicitly corrected positives are superseded. Disputed current positives and
    resolved red flags remain conservative signals; historical-only events are
    separately exposed for clarification, not silently declared current episodes.
    """
    if ledger['contract_version'] == VERSION:
        return legacy.risk_input(ledger)
    state = statuses(ledger)
    retained = [r for r in effective_records(ledger) if state[r['id']] in {'active', 'disputed'}
                and r['subject'] == 'patient' and r['polarity'] == 'present']
    current = [r for r in retained if r['temporality'] == 'current']
    flat = [(r['field'], r['value']) for r in current]
    flat += [item for r in current for item in derived_fields(r)]
    # Same literal concept, same existing red-flag policy. Do not let its
    # serialization field decide whether an affirmed patient fact escalates.
    equivalent = [r for r in retained if r['field']=='symptoms' and r['value']=='呼吸困难']
    flat += [('red_flags','呼吸困难') for r in equivalent if r['temporality']=='current']
    result = {field: sorted({value for f, value in flat if f == field})
              for field in ('symptoms', 'red_flags', 'accompanying_symptoms', 'past_history')}
    # Past medical history describes established background conditions even when
    # the utterance labels their time as historical.
    result['past_history'] = sorted(set(result['past_history']) | {
        r['value'] for r in retained if r['field'] == 'past_history'
        and r['temporality'] == 'historical'})
    resolved = sorted({r['value'] for r in retained if (r['field'] == 'red_flags' or r in equivalent)
                       and r['temporality'] == 'resolved'})
    result['risk_representation_trace'] = dict(version='identical_dyspnea_risk_v2',
        mapped=[dict(id=r['id'],subject=r['subject'],temporality=r['temporality'])
                for r in equivalent if r['temporality'] in {'current','resolved'}],raw_ledger_modified=False)
    result['resolved_red_flags'] = resolved
    result['red_flags'] = sorted(set(result['red_flags']) | {value + '（已缓解，未解除安全升级）' for value in resolved})
    interpretation = interpretation_annotations(ledger)
    result['interpretation_safety_candidates'] = interpretation
    if interpretation:
        result['red_flags'] = sorted(set(result['red_flags']) | {'呼吸困难（口语候选，含义待澄清或已缓解）'})
    result['historical_red_flags'] = sorted({r['value'] for r in retained
        if r['field'] == 'red_flags' and r['temporality'] == 'historical'})
    result['fact_conflicts'] = [r['id'] for r in ledger['records'] if state[r['id']] == 'disputed']
    age_values = [int(r['value']) for r in current if r['field'] == 'age' and r['value'].isdigit()]
    result['age'] = str(max(age_values)) if age_values else ''
    severity_order = {'轻度': 1, '中度': 2, '重度': 3}
    severity_values = [r['value'] for r in current if r['field'] == 'severity' and r['value'] in severity_order]
    result['severity'] = max(severity_values, key=severity_order.get, default='')
    # Preserve the object binding for rules that combine a named symptom and its
    # severity. Global severe_symptom still considers every current severity.
    by_id = {r['id']: r for r in current}
    severity_by_symptom = {}
    for r in current:
        if r['field'] != 'severity' or r['value'] not in severity_order or r['about'] not in by_id:
            continue
        problem = by_id[r['about']]
        names = ([problem['value']] if problem['field'] == 'symptoms' else
                 [v for f, v in derived_fields(problem) if f == 'symptoms'])
        for name in names:
            previous = severity_by_symptom.get(name, '')
            if severity_order[r['value']] > severity_order.get(previous, 0):
                severity_by_symptom[name] = r['value']
    result['severity_by_symptom'] = severity_by_symptom
    return result
