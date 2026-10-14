"""Projection from immutable evidence to the existing UI/planner slot interface.

Flat problem attributes are a view of current_focus only. Risk consumes the
separate full ledger projection, never these focus-specific scalars.
"""
from llm.fact_contract_runtime import ATTRIBUTES, FACT_FIELDS
from memory.fact_ledger_runtime import project, risk_input, interpretation_annotations
from llm.deferred_scope import is_deferred_ambiguous, VERSION as SCOPE_VERSION
from llm.fact_contract_runtime import V2


LIST_FIELDS = {'symptoms', 'accompanying_symptoms', 'red_flags', 'past_history',
               'allergy_history', 'medication_history'}


def case_projection(ledger, previous):
    view = project(ledger)
    projection_slots = {}
    if previous.get('runtime_fact_projection_version'):
        from memory.runtime_fact_projection import VERSION, derive_denied_red_flags
        if previous['runtime_fact_projection_version'] != VERSION:
            raise ValueError('unsupported runtime fact projection')
        raw_view = view
        view, trace = derive_denied_red_flags(raw_view)
        projection_slots = dict(raw_fact_view=raw_view, runtime_fact_projection_trace=trace)
    slots = {field: [] if field in LIST_FIELDS else '' for field in FACT_FIELDS}
    slots['accompanying_symptoms'] = []
    grouped = {}
    for fact in view['facts']:
        if fact['polarity'] == 'present' and fact['field'] not in ATTRIBUTES:
            grouped.setdefault(fact['field'], set()).add(fact['value'])
    for field, values in grouped.items():
        slots[field] = sorted(values) if field in LIST_FIELDS else (next(iter(values)) if len(values) == 1 else '')
        if ledger['contract_version'] == V2 and field == 'sleep':
            slots[field] = '、'.join(sorted(values))
    slots['past_history'] = risk_input(ledger)['past_history']
    problems = view['problems']
    valid_ids = {p['id'] for p in problems}
    selected = [i for i in previous.get('complaint_intent_selected', []) if i['problem_id'] in valid_ids]
    selected_ids = {i['problem_id'] for i in selected}
    chiefs = sorted({p['value'] for p in problems if p['id'] in selected_ids})
    def incomplete(problem):
        filled = {f['field'] for f in problem['attributes'] if f['polarity'] == 'present'}
        return not {'duration', 'severity', 'location'}.issubset(filled)
    existing = next((p for p in problems if p['id'] == previous.get('current_focus')), None)
    focus = existing if existing and incomplete(existing) else next((p for p in problems if incomplete(p)), existing)
    focus = focus or (problems[0] if problems else None)
    focus = next((p for p in problems if p['id'] in selected_ids and incomplete(p)), focus)
    if focus:
        for field in ATTRIBUTES:
            values = {f['value'] for f in focus['attributes'] if f['field'] == field and f['polarity'] == 'present'}
            slots[field] = next(iter(values)) if len(values) == 1 else ''
    slots.update(semantic_contract=view['contract_version'], fact_view=view,
                 interpretation_annotations=interpretation_annotations(ledger),
                 chief_complaint='、'.join(chiefs), chief_complaints=chiefs,
                 complaint_intent_selected=selected,
                 chief_complaint_candidates=sorted(set(view['chief_complaint_candidates']) - set(chiefs)),
                 intent_status=previous.get('intent_status', 'not_evaluated'),
                 current_focus=focus['id'] if focus else None, current_focus_label=focus['value'] if focus else '',
                 fact_extraction_error='',
                 contradictions=[f"{f['field']}：{f['value']}（{f['polarity']}），需确认" for f in view['disputed']],
                 contradiction_fields=sorted({f['field'] for f in view['disputed']}))
    slots['fact_clarifications'] = [dict(fact_id=f['id'], candidate_concept=f['value'],
        policy_version=SCOPE_VERSION,
        question=f"你现在有{f['value']}吗，还是只是想以后讨论这个症状？如果之前有过，也请说明发生的时间。")
        for f in view['facts'] if f['derived_from'] is None and f['field'] in {'symptoms', 'red_flags'}
        and f['polarity'] == 'uncertain' and is_deferred_ambiguous(f['evidence'], f['evidence'])]
    if ledger['contract_version'] == V2:
        slots['observation_attributes'] = view['observations']
        slots['fact_clarifications'] = [dict(fact_id=f['id'], candidate_concept=f['value'],
            policy_version='fact_clarification_v2', kind='presence',
            question=f"你现在有{f['value']}吗？如果之前有过，也请说明发生的时间。")
            for f in view['facts'] if f['derived_from'] is None
            and f['field'] in {'symptoms', 'red_flags', 'sleep', 'sweating', 'appetite', 'thirst'}
            and f['polarity'] == 'uncertain'
            and not any(g['field'] == f['field'] and g['value'] == f['value']
                        and g['polarity'] == 'present' for g in view['facts'])]
        for f in view['facts']:
            if (f['derived_from'] is not None or f['field'] != 'cold_heat'
                    or f['value'] not in {'正常', '和平时一样'} or f['polarity'] != 'present'):
                continue
            missing = [value for value in ('发热', '畏寒') if not any(
                g['field']=='symptoms' and g['value']==value and g['polarity'] in {'present','negated'}
                for g in view['facts'])]
            if missing:
                slots['fact_clarifications'].append(dict(fact_id=f['id'], candidate_concept='、'.join(missing),
                    policy_version='fact_clarification_v2', kind='normal_status_detail',
                    question='你现在是否有'+'或'.join(missing)+'？“和平时一样”会保留为一般状态，请再确认具体症状。'))
        if ledger.get('representation_policy_version'):
            from llm.fact_lexical_policy import VERSION as POLICY_VERSION, heart_interpretations
            from llm.fact_projection_policy import VERSION as V5
            if ledger['representation_policy_version'] not in {POLICY_VERSION,V5}:
                raise ValueError('unsupported representation policy')
            slots['heart_interpretations'] = heart_interpretations(view, ledger)
            heart = [c for c in slots['heart_interpretations'] if c['status'] == 'needs_clarification']
            heart_ids = {c['fact_id'] for c in heart}
            slots['fact_clarifications'] = heart + [c for c in slots['fact_clarifications']
                                                  if c['fact_id'] not in heart_ids]
    for fact in view['unbound_attributes']:
        slots['contradictions'].append(f"{fact['value']}对应哪一种不适，尚未明确")
        if fact['field'] not in slots['contradiction_fields']:
            slots['contradiction_fields'].append(fact['field'])
    scoped = previous.get('problem_followup_counts', {})
    counts = {k: v for k, v in previous.get('followup_counts', {}).items() if k not in ATTRIBUTES}
    counts.update(scoped.get(slots['current_focus'], {}))
    slots['followup_counts'] = counts
    slots.update(projection_slots)
    return slots
