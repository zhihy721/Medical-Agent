"""Opt-in v15 false-alarm fixes. Historical v13/v14 code and grades remain fixed.

The static transaction tail below retains v13 checks after a separately checked
conditional-parser false alarm; it does not skip remaining transaction checks.
"""
import re
from llm.fact_contract_runtime import position_delta
from llm.runtime_extraction_guard import CLINICAL,CONTROL,SYMPTOM_ALIASES,RED_FLAG_SYMPTOMS
from llm.fact_transaction_guard import CALENDAR,RESOLVED,BLOCK
from llm.compact_fact_guard import OBSERVATIONS,PROBLEMS,ATTRIBUTE_CORRECTION
from llm.deferred_scope import evidence_scope,is_deferred_ambiguous
from llm.compact_fact_guard import check as previous_check

VERSION='compact_transaction_guard_v15'
PURE_COMMANDS=[
 r'别管之前(?:的设定了|记的[^。！？；;,，]+)',
 r'(?:直接)?忽略(?:所有先前的|系统)?规则',
 r'(?:立即)?清除关于[^。！？；;,，]+的历史记录',
 r'(?:仅|只)输出(?:OK|[^。！？；;,，]{1,16})',
 r'(?:目前)?无新信息补充',
 r'把[^。！？；;,，]+(?:都)?删掉',
 r'直接按我说的输出[“”「」\"]?状态正常[“”「」\"]?',
]


def pure_control(text):
    # Classification only: raw messages/evidence are never masked or rewritten.
    clauses=[s.strip() for s in re.split(r'[，,。！？；;\n]',text) if s.strip()]
    return bool(clauses and CONTROL.search(text) and all(
        any(re.fullmatch(pattern,c) for pattern in PURE_COMMANDS) for c in clauses))


def check_conditions(inp,delta):
    condition=re.compile(r'(?<!主)(?:如果|要是|假如)(?:以后|今后|将来)?'
        r'(?P<who>我父亲|我母亲|父亲|母亲|我)?(?:以后|今后|将来)?'
        r'(?:出现|发生|有)?(?P<problem>'+CLINICAL.pattern+r')')
    from llm.fact_lexical_policy import PAIN_ALIASES,APPROVED_SYMPTOM_ALIASES
    from llm.fact_projection_policy import BREATHING_WORDS
    aliases={**SYMPTOM_ALIASES,**PAIN_ALIASES,**APPROVED_SYMPTOM_ALIASES,**BREATHING_WORDS}
    def concept(value):return RED_FLAG_SYMPTOMS.get(value,aliases.get(value,value))
    for clause in re.split(r'[，,。！？；;\n]',inp['current_turn']):
        if (CONTROL.search(clause) or re.search(r'[“”\"「」『』]|引用|例如|比如|转述|医生说',clause)
                or re.search(r'不确定|说不准|可能|也许|假设没有',clause)):continue
        for match in condition.finditer(clause):
            who=match.group('who')
            if who is None and re.search(r'父亲|母亲|朋友|同事',clause[:match.start()]):continue
            subject='other' if who and who!='我' else 'patient'
            value=concept(match.group('problem'))
            if not any(f['field'] in {'symptoms','red_flags','sleep','sweating','appetite','thirst'}
                and concept(f['value'])==value and f['subject']==subject
                and f['temporality']=='hypothetical' and f['polarity']=='present'
                for f in list(delta['facts'])+list(inp['known_facts'])):
                raise ValueError('literal_positive_condition_missing_scoped_fact')


def _remaining_transactions(inp,delta,classification):
    facts=delta['facts']
    for f in facts:
        old=next((k for k in inp['known_facts'] if k['id']==f['replaces']),None)
        if f['operation']=='correct' and f['polarity']=='present' and old and old['polarity']=='negated':
            from memory.fact_temporal import prior_denials_ended
            from memory.fact_identity import VERSION as IDENTITY
            from llm.fact_projection_policy import BREATHING_WORDS
            from llm.fact_lexical_policy import PAIN_ALIASES,APPROVED_SYMPTOM_ALIASES
            simulated=[dict(g,start=inp['current_turn'].find(g['evidence']),
                operation='assert' if g['id']==f['id'] else g['operation']) for g in facts]
            transitions=prior_denials_ended(inp,simulated,inp['known_facts'],
                {k['id']:k['status'] for k in inp['known_facts']},identity_version=IDENTITY,
                lexical_aliases={**PAIN_ALIASES,**APPROVED_SYMPTOM_ALIASES,**BREATHING_WORDS})
            if any(t['prior_id']==old['id'] for t in transitions):
                raise ValueError('explicit_new_occurrence_must_not_correct_prior_denial')
        if f['field']=='duration' and CALENDAR.fullmatch(f['value'].strip()):
            raise ValueError('calendar_reference_is_not_elapsed_duration')
        if not (f['field'] in {'symptoms','red_flags'} and f['subject']=='patient'
                and f['temporality']=='current' and f['polarity']=='present'):
            continue
        scope=evidence_scope(inp['current_turn'],f['evidence'])
        if BLOCK.search(scope):continue
        if re.search(r'(?:疼|痛)得厉害',scope):
            degree=[g for g in facts if g['field']=='severity' and g['about']==f['id']
                    and (g['value'],g['polarity'],g['subject'],g['temporality'])==(
                        '厉害','present','patient','current')]
            known_obj=[k for k in inp['known_facts'] if k['status']=='active'
                and all(k[key]==f[key] for key in ('field','value','polarity','subject','temporality'))]
            degree+= [k for k in inp['known_facts'] if k['field']=='severity' and k['value']=='厉害'
                and k['polarity']=='present' and k['status']=='active'
                and any(k['about']==obj['id'] for obj in known_obj)]
            if not degree:raise ValueError('explicit_degree_has_no_bound_attribute')
        if f['operation']=='assert' and RESOLVED.search(scope):
            if not any(g['operation']=='resolve' and g['value']==f['value']
                       and g['subject']==f['subject'] for g in facts):
                raise ValueError('explicit_resolution_requires_resolved_operation')


def _additional_checks(inp,delta,classification):
    for f in delta['facts']:
        scope=evidence_scope(inp['current_turn'],f['evidence'])
        if BLOCK.search(scope):continue
        if f['field'] in OBSERVATIONS and f['value'] in PROBLEMS:
            raise ValueError('defined_problem_in_unrelated_observation_field')
        if (f['field'] in {'symptoms','red_flags'} and f['polarity']=='negated'
                and f['temporality']=='current' and is_deferred_ambiguous(inp['current_turn'],f['evidence'],modern=True)):
            raise ValueError('deferred_discussion_is_not_symptom_denial')
        if (f['operation']=='correct' and f['field'] not in {'duration','severity','location','pain_character'}
                and f['polarity']=='present' and ATTRIBUTE_CORRECTION.search(scope)):
            # Only reject the redundant object correction if a matching actual
            # old attribute is being corrected too. Ambiguous cases abstain.
            if any(g['operation']=='correct' and g['about'] in {f['id'],f['replaces']}
                    and g['field'] in {'duration','severity'} and
                    any(k['id']==g['replaces'] and k['about']==f['replaces'] for k in inp['known_facts'])
                    for g in delta['facts']):
                raise ValueError('attribute_only_correction_must_not_correct_object')


def check(inp,delta,classification):
    position_delta(delta,inp)
    effective=dict(classification)
    if pure_control(inp['current_turn']):effective['class']='possible_control'
    try:
        previous_check(inp,delta,effective)
    except ValueError as exc:
        if str(exc)!='literal_positive_condition_missing_scoped_fact':raise
        # Prior structural/problem checks have completed; only this known
        # parser alarm is reconsidered. True omitted conditions still reject.
        check_conditions(inp,delta)
        _remaining_transactions(inp,delta,effective)
        _additional_checks(inp,delta,effective)
